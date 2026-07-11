module gpu::vk;

import gpu;
import vk;
import std::core::mem;
import std::math;

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

<* Begin dynamic rendering. Attachments must sit in their attachment layouts,
   have matching usage, and cover the pass dimensions. Color views honor the
   target's mip/layer; the depth view uses mip zero. *>
fn void? vk_cmd_begin_render_pass(gpu::CommandList* commands, gpu::RenderPassDesc* desc) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (desc.colors.len == 0 && desc.depth == null) return gpu::INVALID_ARGUMENT~;
    if (desc.width == 0 || desc.height == 0) return gpu::INVALID_ARGUMENT~;
    if (desc.colors.len > state.caps.max_color_attachments) return gpu::INVALID_ARGUMENT~;
    vk::CommandBuffer cb = record.command_buffer;

    @pool() {
        TextureSlot*[] color_slots = {};
        vk::RenderingAttachmentInfo[] attachments = {};
        sz max_created_views = desc.colors.len + (desc.depth != null ? 1 : 0);
        CachedViewCreation[] creations = mem::talloc_array(CachedViewCreation, max_created_views);
        uint creation_count = 0;
        if (desc.colors.len > 0) {
            color_slots = mem::talloc_array(TextureSlot*, desc.colors.len);
            attachments = mem::talloc_array(vk::RenderingAttachmentInfo, desc.colors.len);
            foreach (i, &target : desc.colors) {
                TextureSlot* slot = state.textures.get(target.texture)!;
                color_slots[i] = slot;
                if (!slot.usage.color_attach) return gpu::INVALID_ARGUMENT~;
                if (target.mip_level >= slot.mip_levels || target.array_layer >= slot.array_layers) return gpu::INVALID_ARGUMENT~;
                MipExtent extent = texture_mip_extent(slot, target.mip_level);
                if (desc.width > extent.width || desc.height > extent.height) return gpu::INVALID_ARGUMENT~;
                if (effective_layout(record, slot, target.texture) != gpu::TextureLayout.COLOR_ATTACHMENT) return gpu::INVALID_RESOURCE_STATE~;
            }
        }

        TextureSlot* depth_slot = null;
        if (desc.depth != null) {
            depth_slot = state.textures.get(desc.depth.texture)!;
            if (!depth_slot.usage.depth_attach) return gpu::INVALID_ARGUMENT~;
            if (desc.width > depth_slot.width || desc.height > depth_slot.height) return gpu::INVALID_ARGUMENT~;
            if (effective_layout(record, depth_slot, desc.depth.texture) != gpu::TextureLayout.DEPTH_STENCIL) return gpu::INVALID_RESOURCE_STATE~;
        }

        defer catch rollback_texture_view_creations(state, creations[:creation_count]);

        foreach (i, &target : desc.colors) {
            gpu::TextureViewDesc view_desc = {
                .base_mip    = target.mip_level,
                .mip_count   = 1,
                .base_layer  = target.array_layer,
                .layer_count = 1,
            };
            CachedViewCreation creation;
            vk::ImageView view = resolve_texture_view_tracked(
                state:    state,
                slot:     color_slots[i],
                view:     &view_desc,
                creation: &creation,
            )!;
            if (creation.cell != null) creations[creation_count++] = creation;

            attachments[i] = vk::rendering_attachment_info();
            attachments[i].image_view = view;
            attachments[i].image_layout = vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
            attachments[i].load_op = load_op_to_vk(target.load_op);
            attachments[i].store_op = store_op_to_vk(target.store_op);
            attachments[i].clear_value.color.float32 = target.clear.rgba;
        }

        vk::RenderingAttachmentInfo depth_attach;
        vk::RenderingInfo rendering = vk::rendering_info();
        rendering.render_area = { .offset = { .x = 0, .y = 0 }, .extent = { .width = desc.width, .height = desc.height } };
        rendering.layer_count = 1;
        rendering.color_attachment_count = (uint)attachments.len;
        rendering.p_color_attachments = attachments.ptr;
        if (desc.depth != null) {
            gpu::TextureViewDesc depth_view_desc = { .mip_count = 1, .layer_count = 1 };
            CachedViewCreation depth_creation;
            vk::ImageView depth_view = resolve_texture_view_tracked(
                state:    state,
                slot:     depth_slot,
                view:     &depth_view_desc,
                creation: &depth_creation,
            )!;
            if (depth_creation.cell != null) creations[creation_count++] = depth_creation;
            depth_attach = vk::rendering_attachment_info();
            depth_attach.image_view = depth_view;
            depth_attach.image_layout = vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attach.load_op = load_op_to_vk(desc.depth.load_op);
            depth_attach.store_op = store_op_to_vk(desc.depth.store_op);
            depth_attach.clear_value.depth_stencil = {
                .depth   = desc.depth.clear.depth,
                .stencil = desc.depth.clear.stencil,
            };
            rendering.p_depth_attachment = &depth_attach;
        }
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

    record.active_render_width = desc.width;
    record.active_render_height = desc.height;
    record.state = CommandState.RECORDING_RENDER_PASS;
}

fn void? vk_cmd_set_viewport(gpu::CommandList* commands, gpu::Viewport* viewport) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    if (viewport == null) return gpu::INVALID_ARGUMENT~;
    if (!math::is_finite(viewport.x)
        || !math::is_finite(viewport.y)
        || !math::is_finite(viewport.width)
        || !math::is_finite(viewport.height)
        || !math::is_finite(viewport.min_depth)
        || !math::is_finite(viewport.max_depth)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (viewport.x < 0.0f || viewport.y < 0.0f
        || viewport.width <= 0.0f || viewport.height <= 0.0f) {
        return gpu::INVALID_ARGUMENT~;
    }
    float right = viewport.x + viewport.width;
    float bottom = viewport.y + viewport.height;
    if (!math::is_finite(right) || !math::is_finite(bottom)
        || right > (float)record.active_render_width
        || bottom > (float)record.active_render_height) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (viewport.min_depth < 0.0f || viewport.min_depth > 1.0f
        || viewport.max_depth < 0.0f || viewport.max_depth > 1.0f) {
        return gpu::INVALID_ARGUMENT~;
    }

    vk::Viewport vk_viewport = {
        .x         = viewport.x,
        .y         = viewport.y,
        .width     = viewport.width,
        .height    = viewport.height,
        .min_depth = viewport.min_depth,
        .max_depth = viewport.max_depth,
    };
    vk::cmd_set_viewport(record.command_buffer, 0, 1, &vk_viewport);
}

fn void? vk_cmd_set_scissor(gpu::CommandList* commands, gpu::ScissorRect* scissor) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    if (scissor == null) return gpu::INVALID_ARGUMENT~;
    if (scissor.x < 0 || scissor.y < 0 || scissor.width < 0 || scissor.height < 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    long right = (long)scissor.x + (long)scissor.width;
    long bottom = (long)scissor.y + (long)scissor.height;
    if (right > int::max || bottom > int::max
        || (ulong)right > (ulong)record.active_render_width
        || (ulong)bottom > (ulong)record.active_render_height) {
        return gpu::INVALID_ARGUMENT~;
    }

    vk::Rect2D vk_scissor = {
        .offset = { .x = scissor.x, .y = scissor.y },
        .extent = { .width = (uint)scissor.width, .height = (uint)scissor.height },
    };
    vk::cmd_set_scissor(record.command_buffer, 0, 1, &vk_scissor);
}

fn void? vk_cmd_end_render_pass(gpu::CommandList* commands) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    record.active_render_width = 0;
    record.active_render_height = 0;
    record.state = CommandState.RECORDING;
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

<* Aliases of one cache entry can carry different dynamic-state snapshots, so
   a handle change replays state even when the pipeline bind is skipped. *>
fn void? bind_graphics_and_push(VkDeviceState* state, CommandRecord* record, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root) @private {
    vk::CommandBuffer cb = record.command_buffer;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if ((ulong)handle != (ulong)record.last_pipeline) {
        if (slot.entry != record.last_entry) {
            vk::cmd_bind_pipeline(cb, vk::PipelineBindPoint.GRAPHICS, slot.pipeline);
            bind_descriptor_heap(state, cb, slot.layout, vk::PipelineBindPoint.GRAPHICS);
            record.last_entry = slot.entry;
        }
        replay_dynamic_state(cb, slot);
        record.last_pipeline = handle;
    }
    gpu::GraphicsRootPush push = { .vertex_root_gpu = vertex_root, .fragment_root_gpu = fragment_root };
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(cb, slot.layout, (vk::ShaderStageFlags)push_stages, 0, GRAPHICS_ROOT_PUSH_SIZE, &push);
}

fn void? vk_cmd_draw(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, uint vertex_count, uint instance_count) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_graphics_and_push(
        state:         state,
        record:        record,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    )!;
    vk::cmd_draw(cb, vertex_count, instance_count, 0, 0);
}

fn void? vk_cmd_draw_indexed(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan index_span, uint index_count, uint instance_count, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    usz index_size = index_element_size(index_type);
    BufferSlot* buffer = resolve_index_span(
        state:      state,
        span:       &index_span,
        index_type: index_type,
        required:   (usz)index_count * index_size,
    )!;

    vk::CommandBuffer cb = record.command_buffer;
    bind_graphics_and_push(
        state:         state,
        record:        record,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    )!;
    vk::cmd_bind_index_buffer(cb, buffer.buffer, (vk::DeviceSize)index_span.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
    vk::cmd_draw_indexed(cb, index_count, instance_count, 0, 0, 0);
}

fn void? vk_cmd_draw_indirect(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DrawIndirectCommand::size,
        count:    draw_count,
    )!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_graphics_and_push(
        state:         state,
        record:        record,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    )!;
    vk::cmd_draw_indirect(cb, args_buffer.buffer, (vk::DeviceSize)args.offset, draw_count, gpu::DrawIndirectCommand::size);
}

<* Element count comes from the GPU-written args and is not checked here. *>
fn void? bind_index_span(VkDeviceState* state, vk::CommandBuffer cb, gpu::GpuSpan* index_span, gpu::IndexType index_type) @private {
    BufferSlot* buffer = resolve_index_span(
        state:      state,
        span:       index_span,
        index_type: index_type,
        required:   0,
    )!;
    vk::cmd_bind_index_buffer(cb, buffer.buffer, (vk::DeviceSize)index_span.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
}

<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn void? vk_cmd_draw_indexed_indirect(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DrawIndexedIndirectCommand::size,
        count:    draw_count,
    )!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_graphics_and_push(
        state:         state,
        record:        record,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    )!;
    bind_index_span(
        state:      state,
        cb:         cb,
        index_span: &index_span,
        index_type: index_type,
    )!;
    vk::cmd_draw_indexed_indirect(cb, args_buffer.buffer, (vk::DeviceSize)args.offset, draw_count, gpu::DrawIndexedIndirectCommand::size);
}

fn void? vk_cmd_draw_indexed_indirect_count(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, gpu::GpuSpan count_span, uint max_draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DrawIndexedIndirectCommand::size,
        count:    max_draw_count,
    )!;
    BufferSlot* count_buffer = resolve_indirect_span(
        state:    state,
        span:     &count_span,
        arg_size: 4,
        count:    1,
    )!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_graphics_and_push(
        state:         state,
        record:        record,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    )!;
    bind_index_span(
        state:      state,
        cb:         cb,
        index_span: &index_span,
        index_type: index_type,
    )!;
    vk::cmd_draw_indexed_indirect_count(
        cb,
        args_buffer.buffer,
        (vk::DeviceSize)args.offset,
        count_buffer.buffer,
        (vk::DeviceSize)count_span.offset,
        max_draw_count,
        gpu::DrawIndexedIndirectCommand::size,
    );
}
