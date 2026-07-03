module gpu::vk;

import gpu;
import vk;
import std::core::mem;

<* One command pool per queue family per frame-in-flight. Command buffers are
   freed by pool reset at frame retire, so there is no per-command-list free
   path. Two pool arrays are kept even when the families are equal. *>
fn void? create_command_pools(VkDeviceState* state) {
    state.graphics_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    state.transfer_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    for (uint i = 0; i < state.frames_in_flight; i++) {
        vk::CommandPoolCreateInfo gfx = vk::command_pool_create_info()
            .set_queue_family_index(state.graphics_family)
            .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
        check(vk::create_command_pool(state.device, &gfx, null, &state.graphics_pools[i]))!;

        vk::CommandPoolCreateInfo xfer = vk::command_pool_create_info()
            .set_queue_family_index(state.transfer_family)
            .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
        check(vk::create_command_pool(state.device, &xfer, null, &state.transfer_pools[i]))!;
    }
}

fn void destroy_command_pools(VkDeviceState* state) {
    if (state.graphics_pools.ptr != null) {
        foreach (pool : state.graphics_pools) {
            if (pool != null) vk::destroy_command_pool(state.device, pool, null);
        }
        mem::free(state.graphics_pools.ptr);
        state.graphics_pools = {};
    }
    if (state.transfer_pools.ptr != null) {
        foreach (pool : state.transfer_pools) {
            if (pool != null) vk::destroy_command_pool(state.device, pool, null);
        }
        mem::free(state.transfer_pools.ptr);
        state.transfer_pools = {};
    }
}

fn void? vk_begin_commands(gpu::Device* device, gpu::QueueKind queue, gpu::CommandList* out_commands) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    vk::CommandPool pool = queue == gpu::QueueKind.TRANSFER
        ? state.transfer_pools[state.frame_index]
        : state.graphics_pools[state.frame_index];

    vk::CommandBufferAllocateInfo alloc = vk::command_buffer_allocate_info()
        .set_command_pool(pool)
        .set_level(vk::CommandBufferLevel.PRIMARY)
        .set_command_buffer_count(1);
    vk::CommandBuffer cmd;
    check(vk::allocate_command_buffers(state.device, &alloc, &cmd))!;

    vk::CommandBufferBeginInfo begin = vk::command_buffer_begin_info()
        .set_flags(vk::CommandBufferUsageFlagBits.ONE_TIME_SUBMIT_BIT);
    check(vk::begin_command_buffer(cmd, &begin))!;

    *out_commands = (gpu::CommandList){
        .device  = device,
        .backend = (void*)cmd,
        .queue   = queue,
        .state   = gpu::CommandState.RECORDING,
    };
}

fn void? vk_end_commands(gpu::Device* device, gpu::CommandList* commands) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    check(vk::end_command_buffer((vk::CommandBuffer)commands.backend))!;
    commands.state = gpu::CommandState.EXECUTABLE;
}

fn void? vk_cmd_copy_buffer(gpu::CommandList* commands, gpu::BufferCopyDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    BufferSlot* src = state.buffers.get(desc.src)!;
    BufferSlot* dst = state.buffers.get(desc.dst)!;

    vk::BufferCopy region = {
        .src_offset = (vk::DeviceSize)desc.src_offset,
        .dst_offset = (vk::DeviceSize)desc.dst_offset,
        .size       = (vk::DeviceSize)desc.size,
    };
    vk::cmd_copy_buffer((vk::CommandBuffer)commands.backend, src.buffer, dst.buffer, 1, &region);
}

fn void? vk_cmd_fill_buffer(gpu::CommandList* commands, gpu::BufferHandle buffer, usz offset, usz size, uint value) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    BufferSlot* slot = state.buffers.get(buffer)!;
    vk::cmd_fill_buffer((vk::CommandBuffer)commands.backend, slot.buffer, (vk::DeviceSize)offset, (vk::DeviceSize)size, value);
}

<* Validate a copy region against the texture, the mip's clamped extent,
   Vulkan's buffer-offset alignment, and the span's bounds, then translate it.
   Zero defaults resolve here: extent 0 = full mip, layer_count 0 = 1. *>
fn vk::BufferImageCopy? resolve_texture_copy_region(gpu::GpuSpan* span, TextureSlot* slot, uint row_length_texels, uint mip, uint base_layer, uint layer_count, uint x, uint y, uint width, uint height) @private {
    if (mip >= slot.mip_levels) return gpu::INVALID_ARGUMENT~;
    uint layers = layer_count == 0 ? 1 : layer_count;
    if (base_layer + layers > slot.array_layers) return gpu::INVALID_ARGUMENT~;

    MipExtent extent = texture_mip_extent(slot, mip);
    if (width == 0)  width = extent.width;
    if (height == 0) height = extent.height;
    if (x + width > extent.width || y + height > extent.height) return gpu::INVALID_ARGUMENT~;
    if (row_length_texels != 0 && row_length_texels < width) return gpu::INVALID_ARGUMENT~;

    usz texel_size = format_texel_size(slot.format);
    if (texel_size == 0) return gpu::INVALID_ARGUMENT~;
    if (span.offset % texel_size != 0 || span.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;

    usz row_texels = row_length_texels == 0 ? width : row_length_texels;
    usz copy_bytes = row_texels * texel_size * height * layers;
    if (copy_bytes > span.size) return gpu::INVALID_ARGUMENT~;

    return (vk::BufferImageCopy){
        .buffer_offset      = (vk::DeviceSize)span.offset,
        .buffer_row_length  = row_length_texels,
        .buffer_image_height = 0,
        .image_subresource  = {
            .aspect_mask      = format_aspect(slot.format),
            .mip_level        = mip,
            .base_array_layer = base_layer,
            .layer_count      = layers,
        },
        .image_offset = { .x = (int)x, .y = (int)y, .z = 0 },
        .image_extent = { .width = width, .height = height, .depth = 1 },
    };
}

fn void? vk_cmd_copy_buffer_to_texture(gpu::CommandList* commands, gpu::BufferTextureCopyDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TextureSlot* texture = state.textures.get(desc.texture)!;
    BufferSlot* buffer = state.buffers.get(desc.src.buffer)!;
    if (!texture.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (texture.current_layout != gpu::TextureLayout.TRANSFER_DST) return gpu::INVALID_RESOURCE_STATE~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        span:              &desc.src,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_buffer_to_image((vk::CommandBuffer)commands.backend, buffer.buffer, texture.image, layout_to_vk(texture.current_layout), 1, &region);
}

fn void? vk_cmd_copy_texture_to_buffer(gpu::CommandList* commands, gpu::TextureBufferCopyDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TextureSlot* texture = state.textures.get(desc.texture)!;
    BufferSlot* buffer = state.buffers.get(desc.dst.buffer)!;
    if (!texture.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    if (texture.current_layout != gpu::TextureLayout.TRANSFER_SRC) return gpu::INVALID_RESOURCE_STATE~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        span:              &desc.dst,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_image_to_buffer((vk::CommandBuffer)commands.backend, texture.image, layout_to_vk(texture.current_layout), buffer.buffer, 1, &region);
}

<* The heap is rebound per dispatch: pipeline layouts differ in push-constant
   range, which breaks descriptor set compatibility across pipelines. *>
fn void? vk_cmd_dispatch(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, gpu::Vec3u groups) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    vk::cmd_bind_pipeline(cb, vk::PipelineBindPoint.COMPUTE, slot.pipeline);
    bind_descriptor_heap(state, cb, slot.layout, vk::PipelineBindPoint.COMPUTE);
    ulong addr = (ulong)root;
    vk::cmd_push_constants(cb, slot.layout, vk::ShaderStageFlagBits.COMPUTE_BIT, 0, 8, &addr);
    vk::cmd_dispatch(cb, groups.x, groups.y, groups.z);
}
