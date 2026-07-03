module gpu::vk;

import gpu;
import vk;
import std::core::mem;

struct GraphicsRootPush @private {
    ulong vertex_root;
    ulong fragment_root;
}
$assert GraphicsRootPush::size == GRAPHICS_ROOT_PUSH_SIZE;

fn vk::AttachmentLoadOp load_op_to_vk(gpu::LoadOp op) @private {
    switch (op) {
        case gpu::LoadOp.LOAD:      return vk::AttachmentLoadOp.LOAD;
        case gpu::LoadOp.CLEAR:     return vk::AttachmentLoadOp.CLEAR;
        case gpu::LoadOp.DONT_CARE: return vk::AttachmentLoadOp.DONT_CARE;
        default:                    return vk::AttachmentLoadOp.DONT_CARE;
    }
}

fn vk::AttachmentStoreOp store_op_to_vk(gpu::StoreOp op) @private {
    switch (op) {
        case gpu::StoreOp.STORE:     return vk::AttachmentStoreOp.STORE;
        case gpu::StoreOp.DONT_CARE: return vk::AttachmentStoreOp.DONT_CARE;
        default:                     return vk::AttachmentStoreOp.DONT_CARE;
    }
}

<* Begin dynamic rendering. Attachment textures must carry color_attach usage
   and sit in the COLOR_ATTACHMENT tracked layout; the attachment view for the
   requested mip/layer comes from the texture's view cache. Viewport and
   scissor are set to the full pass dimensions. *>
fn void? vk_cmd_begin_render_pass(gpu::CommandList* commands, gpu::RenderPassDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (desc.colors.len == 0 || desc.width == 0 || desc.height == 0) return gpu::INVALID_ARGUMENT~;
    if (desc.depth != null) return gpu::INVALID_ARGUMENT~;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;

    @pool() {
        vk::RenderingAttachmentInfo[] attachments = mem::talloc_array(vk::RenderingAttachmentInfo, desc.colors.len);
        foreach (i, &target : desc.colors) {
            TextureSlot* slot = state.textures.get(target.texture)!;
            if (!slot.usage.color_attach) return gpu::INVALID_ARGUMENT~;
            if (slot.current_layout != gpu::TextureLayout.COLOR_ATTACHMENT) return gpu::INVALID_RESOURCE_STATE~;

            gpu::TextureViewDesc view_desc = {
                .base_mip    = target.mip_level,
                .mip_count   = 1,
                .base_layer  = target.array_layer,
                .layer_count = 1,
            };
            vk::ImageView view = resolve_texture_view(state, slot, &view_desc)!;

            attachments[i] = vk::rendering_attachment_info();
            attachments[i].image_view = view;
            attachments[i].image_layout = vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
            attachments[i].load_op = load_op_to_vk(target.load_op);
            attachments[i].store_op = store_op_to_vk(target.store_op);
            attachments[i].clear_value.color.float32 = target.clear.rgba;
        }

        vk::RenderingInfo rendering = vk::rendering_info();
        rendering.render_area = { .offset = { .x = 0, .y = 0 }, .extent = { .width = desc.width, .height = desc.height } };
        rendering.layer_count = 1;
        rendering.color_attachment_count = (uint)attachments.len;
        rendering.p_color_attachments = attachments.ptr;
        vk::cmd_begin_rendering(cb, &rendering);
    };

    vk::Viewport viewport = {
        .x = 0.0, .y = 0.0,
        .width = (float)desc.width, .height = (float)desc.height,
        .min_depth = 0.0, .max_depth = 1.0,
    };
    vk::cmd_set_viewport(cb, 0, 1, &viewport);
    vk::Rect2D scissor = { .offset = { .x = 0, .y = 0 }, .extent = { .width = desc.width, .height = desc.height } };
    vk::cmd_set_scissor(cb, 0, 1, &scissor);

    commands.state = gpu::CommandState.RECORDING_RENDER_PASS;
}

fn void? vk_cmd_end_render_pass(gpu::CommandList* commands) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering((vk::CommandBuffer)commands.backend);
    commands.state = gpu::CommandState.RECORDING;
}

<* Apply the slot's raster/depth snapshot through the core-1.3 dynamic-state
   commands; graphics pipelines bake none of this state. *>
fn void replay_dynamic_state(vk::CommandBuffer cb, PipelineSlot* slot) @private {
    vk::cmd_set_cull_mode(cb, (vk::CullModeFlags)cull_mode_to_vk(slot.raster.cull_mode));
    vk::cmd_set_front_face(cb, slot.raster.front_face == gpu::FrontFace.CLOCKWISE
        ? vk::FrontFace.CLOCKWISE : vk::FrontFace.COUNTER_CLOCKWISE);
    vk::cmd_set_depth_test_enable(cb, slot.depth.test_enable ? vk::TRUE : vk::FALSE);
    vk::cmd_set_depth_write_enable(cb, slot.depth.write_enable ? vk::TRUE : vk::FALSE);
    vk::cmd_set_depth_compare_op(cb, compare_op_to_vk(slot.depth.compare));
}

<* Two-level redundant-bind filter: an unchanged handle skips everything but
   the push constants; a changed handle over the same cache entry skips the
   pipeline/heap bind but replays the slot's dynamic-state snapshot, which may
   differ between aliases of one entry. *>
fn void? bind_graphics_and_push(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, vk::CommandBuffer cb) @private {
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if ((ulong)handle != (ulong)commands.last_pipeline) {
        if (slot.entry != commands.last_entry) {
            vk::cmd_bind_pipeline(cb, vk::PipelineBindPoint.GRAPHICS, slot.pipeline);
            bind_descriptor_heap(state, cb, slot.layout, vk::PipelineBindPoint.GRAPHICS);
            commands.last_entry = slot.entry;
        }
        replay_dynamic_state(cb, slot);
        commands.last_pipeline = handle;
    }
    GraphicsRootPush push = { .vertex_root = (ulong)vertex_root, .fragment_root = (ulong)fragment_root };
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(cb, slot.layout, (vk::ShaderStageFlags)push_stages, 0, GRAPHICS_ROOT_PUSH_SIZE, &push);
}

fn void? vk_cmd_draw(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, uint vertex_count, uint instance_count) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(commands, handle, vertex_root, fragment_root, cb)!;
    vk::cmd_draw(cb, vertex_count, instance_count, 0, 0);
}

fn void? vk_cmd_draw_indexed(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan index_span, uint index_count, uint instance_count, gpu::IndexType index_type) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    BufferSlot* buffer = state.buffers.get(index_span.buffer)!;

    usz index_size = index_type == gpu::IndexType.U16 ? 2 : 4;
    if ((usz)index_count * index_size > index_span.size) return gpu::INVALID_ARGUMENT~;

    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(commands, handle, vertex_root, fragment_root, cb)!;
    vk::cmd_bind_index_buffer(cb, buffer.buffer, (vk::DeviceSize)index_span.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
    vk::cmd_draw_indexed(cb, index_count, instance_count, 0, 0, 0);
}
