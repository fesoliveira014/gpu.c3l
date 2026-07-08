module gpu::vk;

import gpu;
import vk;
import std::thread;
import std::core::mem;

// Concrete (non-generic) slot table for recording contexts. Monomorphized
// twin of gpu::slots, same c3c 0.8.0 rationale as PipelineTable.
const uint RECORDING_CONTEXT_FREE_NONE = 0;

struct RecordingContextState {
    vk::CommandPool[] graphics_pools;
    // Aliases graphics_pools when the compute queue shares the graphics
    // family; owned storage only when distinct (compute_pools_distinct).
    vk::CommandPool[] compute_pools;
    vk::CommandPool[] transfer_pools;
    bool              compute_pools_distinct;
}

struct RecordingContextCell {
    RecordingContextState value;
    uint                  generation;
    uint                  next_free;
    bool                  used;
}

struct RecordingContextTable {
    RecordingContextCell[] slots;
    uint                   free_head;
    uint                   next_new;
}

fn gpu::RecordingContextHandle? RecordingContextTable.alloc(&self, RecordingContextState value) {
    uint idx;
    if (self.free_head != RECORDING_CONTEXT_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation++;
    self.slots[idx].used = true;
    return (gpu::RecordingContextHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn RecordingContextState*? RecordingContextTable.get(&self, gpu::RecordingContextHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? RecordingContextTable.free(&self, gpu::RecordingContextHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* One command pool per queue family per frame-in-flight, owned by a
   recording context (confined to one thread at a time). Command buffers are
   freed by pool reset at frame retire. *>
fn void? create_context_pools(VkDeviceState* state, RecordingContextState* context) {
    context.graphics_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    context.transfer_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    context.compute_pools_distinct = state.compute_family != state.graphics_family;
    if (context.compute_pools_distinct) {
        context.compute_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    }
    for (uint i = 0; i < state.frames_in_flight; i++) {
        vk::CommandPoolCreateInfo gfx = vk::command_pool_create_info()
            .set_queue_family_index(state.graphics_family)
            .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
        check(vk::create_command_pool(state.device, &gfx, null, &context.graphics_pools[i]))!;

        if (context.compute_pools_distinct) {
            vk::CommandPoolCreateInfo comp = vk::command_pool_create_info()
                .set_queue_family_index(state.compute_family)
                .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
            check(vk::create_command_pool(state.device, &comp, null, &context.compute_pools[i]))!;
        }

        vk::CommandPoolCreateInfo xfer = vk::command_pool_create_info()
            .set_queue_family_index(state.transfer_family)
            .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
        check(vk::create_command_pool(state.device, &xfer, null, &context.transfer_pools[i]))!;
    }
    if (!context.compute_pools_distinct) context.compute_pools = context.graphics_pools;
}

fn void destroy_context_pools(VkDeviceState* state, RecordingContextState* context) {
    if (context.graphics_pools.ptr != null) {
        foreach (pool : context.graphics_pools) {
            if (pool != null) vk::destroy_command_pool(state.device, pool, null);
        }
        mem::free(context.graphics_pools.ptr);
        context.graphics_pools = {};
    }
    if (context.compute_pools_distinct && context.compute_pools.ptr != null) {
        foreach (pool : context.compute_pools) {
            if (pool != null) vk::destroy_command_pool(state.device, pool, null);
        }
        mem::free(context.compute_pools.ptr);
    }
    context.compute_pools = {};
    if (context.transfer_pools.ptr != null) {
        foreach (pool : context.transfer_pools) {
            if (pool != null) vk::destroy_command_pool(state.device, pool, null);
        }
        mem::free(context.transfer_pools.ptr);
        context.transfer_pools = {};
    }
}

fn gpu::RecordingContextHandle? vk_create_recording_context(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    RecordingContextState context;
    create_context_pools(state, &context)!;
    if (try handle = state.recording_contexts.alloc(context)) {
        return handle;
    }
    destroy_context_pools(state, &context);
    return gpu::SLOT_TABLE_FULL~;
}

<* The built-in default context cannot be destroyed; a context's command
   lists must be retired first (device teardown sweeps leftovers). *>
fn void? vk_destroy_recording_context(gpu::Device* device, gpu::RecordingContextHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if ((ulong)handle == (ulong)state.default_context) return gpu::INVALID_ARGUMENT~;
    RecordingContextState* context = state.recording_contexts.get(handle)!;
    destroy_context_pools(state, context);
    state.recording_contexts.free(handle)!;
}

<* The pool whose family matches the queue the list will submit to. *>
fn vk::CommandPool pool_for(RecordingContextState* context, gpu::QueueKind queue, uint frame_index) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return context.transfer_pools[frame_index];
        case gpu::QueueKind.COMPUTE:  return context.compute_pools[frame_index];
        default:                      return context.graphics_pools[frame_index];
    }
}

fn void? vk_begin_commands(gpu::Device* device, gpu::QueueKind queue, gpu::RecordingContextHandle ctx, gpu::CommandList* out_commands) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    gpu::RecordingContextHandle resolved = ctx.is_valid() ? ctx : state.default_context;
    RecordingContextState* context = state.recording_contexts.get(resolved)!;
    vk::CommandPool pool = pool_for(context, queue, state.frame_index);

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
        .device        = device,
        .backend       = (void*)cmd,
        .queue         = queue,
        .state         = gpu::CommandState.RECORDING,
        .last_pipeline = gpu::PIPELINE_HANDLE_INVALID,
        .last_entry    = PIPELINE_ENTRY_NONE,
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
    if (!src.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    if (!dst.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (desc.size == 0) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(src.size, desc.src_offset, desc.size)) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(dst.size, desc.dst_offset, desc.size)) return gpu::INVALID_ARGUMENT~;
    // Vulkan forbids overlapping regions within one buffer.
    if (desc.src == desc.dst
        && desc.src_offset < desc.dst_offset + desc.size
        && desc.dst_offset < desc.src_offset + desc.size) {
        return gpu::INVALID_ARGUMENT~;
    }

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
    if (!slot.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    // Vulkan: offset and size must be 4-byte multiples; size 0 is the
    // whole-buffer convention (translated to WHOLE_SIZE below).
    if (offset % 4 != 0 || size % 4 != 0) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize fill_size = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    vk::cmd_fill_buffer((vk::CommandBuffer)commands.backend, slot.buffer, (vk::DeviceSize)offset, fill_size, value);
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
    if (!buffer.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    if (!texture.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (texture_layout(texture) != gpu::TextureLayout.TRANSFER_DST) return gpu::INVALID_RESOURCE_STATE~;

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
    vk::cmd_copy_buffer_to_image((vk::CommandBuffer)commands.backend, buffer.buffer, texture.image, layout_to_vk(texture_layout(texture)), 1, &region);
}

fn void? vk_cmd_copy_texture_to_buffer(gpu::CommandList* commands, gpu::TextureBufferCopyDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TextureSlot* texture = state.textures.get(desc.texture)!;
    BufferSlot* buffer = state.buffers.get(desc.dst.buffer)!;
    if (!buffer.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (!texture.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    if (texture_layout(texture) != gpu::TextureLayout.TRANSFER_SRC) return gpu::INVALID_RESOURCE_STATE~;

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
    vk::cmd_copy_image_to_buffer((vk::CommandBuffer)commands.backend, texture.image, layout_to_vk(texture_layout(texture)), buffer.buffer, 1, &region);
}

<* Resolve an indirect argument span: indirect usage, 4-byte offset alignment,
   and count*arg_size within bounds. *>
fn BufferSlot*? resolve_indirect_span(VkDeviceState* state, gpu::GpuSpan* span, usz arg_size, uint count) {
    BufferSlot* slot = state.buffers.get(span.buffer)!;
    if (!slot.usage.indirect) return gpu::INVALID_ARGUMENT~;
    if (span.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;
    if ((usz)count * arg_size > span.size) return gpu::INVALID_ARGUMENT~;
    return slot;
}

fn void? bind_compute_and_push(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, vk::CommandBuffer cb) @private {
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if ((ulong)handle != (ulong)commands.last_pipeline) {
        vk::cmd_bind_pipeline(cb, vk::PipelineBindPoint.COMPUTE, slot.pipeline);
        bind_descriptor_heap(state, cb, slot.layout, vk::PipelineBindPoint.COMPUTE);
        commands.last_pipeline = handle;
        commands.last_entry = slot.entry;
    }
    gpu::RootPush push = { .root_gpu = root };
    vk::cmd_push_constants(cb, slot.layout, vk::ShaderStageFlagBits.COMPUTE_BIT, 0, gpu::RootPush::size, &push);
}

<* Pipeline and heap binds are skipped when the handle is unchanged since the
   last bind; the heap otherwise rebinds because compute layouts differ in
   push-constant range, which breaks descriptor set compatibility. *>
fn void? vk_cmd_dispatch(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, gpu::Vec3u groups) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.COMPUTE)!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_compute_and_push(
        commands: commands,
        handle:   handle,
        root:     root,
        cb:       cb,
    )!;
    vk::cmd_dispatch(cb, groups.x, groups.y, groups.z);
}

fn void? vk_cmd_dispatch_indirect(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, gpu::GpuSpan args) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.COMPUTE)!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DispatchIndirectCommand::size,
        count:    1,
    )!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_compute_and_push(
        commands: commands,
        handle:   handle,
        root:     root,
        cb:       cb,
    )!;
    vk::cmd_dispatch_indirect(cb, args_buffer.buffer, (vk::DeviceSize)args.offset);
}
