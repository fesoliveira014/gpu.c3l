module gpu::vk;

import gpu;
import vk;
import std::core::mem;

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

fn void rollback_render_pass_views(VkDeviceState* state, TextureSlot*[] slots, vk::ImageView[] views, uint count) @private {
    for (uint i = 0; i < count; i++) {
        foreach (&cached : slots[i].views) {
            if (!cached.used || cached.view != views[i]) continue;
            vk::destroy_image_view(state.device, cached.view, null);
            cached.used = false;
            break;
        }
    }
}

<* View resolution may allocate cache entries. Keep the pass begin atomic if
   a later view resolution faults. *>
fn void track_created_view(TextureSlot* slot, vk::ImageView view, bool created, TextureSlot*[] slots, vk::ImageView[] views, uint* count) @private {
    if (!created) return;
    slots[*count] = slot;
    views[(*count)++] = view;
}

<* Begin dynamic rendering. Attachments must sit in their attachment layouts,
   have matching usage, and cover the pass dimensions. Color views honor the
   target's mip/layer; the depth view uses mip zero. *>
fn void? vk_cmd_begin_render_pass(gpu::CommandList* commands, gpu::RenderPassDesc* desc) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (desc.colors.len == 0 && desc.depth == null) return gpu::INVALID_ARGUMENT~;
    if (desc.width == 0 || desc.height == 0) return gpu::INVALID_ARGUMENT~;
    if (desc.colors.len > state.caps.max_color_attachments) return gpu::INVALID_ARGUMENT~;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;

    @pool() {
        TextureSlot*[] color_slots = {};
        TextureSlot*[] created_view_slots = {};
        vk::ImageView[] created_views = {};
        uint created_view_count = 0;
        vk::RenderingAttachmentInfo[] attachments = {};
        sz max_created_views = desc.colors.len + (desc.depth != null ? 1 : 0);
        created_view_slots = mem::talloc_array(TextureSlot*, max_created_views);
        created_views = mem::talloc_array(vk::ImageView, max_created_views);
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
                if (effective_layout(commands, slot, target.texture) != gpu::TextureLayout.COLOR_ATTACHMENT) return gpu::INVALID_RESOURCE_STATE~;
            }
        }

        TextureSlot* depth_slot = null;
        if (desc.depth != null) {
            depth_slot = state.textures.get(desc.depth.texture)!;
            if (!depth_slot.usage.depth_attach) return gpu::INVALID_ARGUMENT~;
            if (desc.width > depth_slot.width || desc.height > depth_slot.height) return gpu::INVALID_ARGUMENT~;
            if (effective_layout(commands, depth_slot, desc.depth.texture) != gpu::TextureLayout.DEPTH_STENCIL) return gpu::INVALID_RESOURCE_STATE~;
        }

        defer catch rollback_render_pass_views(
            state: state,
            slots: created_view_slots,
            views: created_views,
            count: created_view_count,
        );

        foreach (i, &target : desc.colors) {
            gpu::TextureViewDesc view_desc = {
                .base_mip    = target.mip_level,
                .mip_count   = 1,
                .base_layer  = target.array_layer,
                .layer_count = 1,
            };
            bool created;
            vk::ImageView view = resolve_texture_view(
                state:   state,
                slot:    color_slots[i],
                view:    &view_desc,
                created: &created,
            )!;
            track_created_view(
                slot:   color_slots[i],
                view:   view,
                created: created,
                slots:  created_view_slots,
                views:  created_views,
                count:  &created_view_count,
            );

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
            bool depth_view_created;
            vk::ImageView depth_view = resolve_texture_view(
                state:   state,
                slot:    depth_slot,
                view:    &depth_view_desc,
                created: &depth_view_created,
            )!;
            track_created_view(
                slot:    depth_slot,
                view:    depth_view,
                created: depth_view_created,
                slots:   created_view_slots,
                views:   created_views,
                count:   &created_view_count,
            );
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

<* Aliases of one cache entry can carry different dynamic-state snapshots, so
   a handle change replays state even when the pipeline bind is skipped. *>
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
    gpu::GraphicsRootPush push = { .vertex_root_gpu = vertex_root, .fragment_root_gpu = fragment_root };
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(cb, slot.layout, (vk::ShaderStageFlags)push_stages, 0, GRAPHICS_ROOT_PUSH_SIZE, &push);
}

fn void? vk_cmd_draw(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, uint vertex_count, uint instance_count) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(commands, handle, vertex_root, fragment_root, cb)!;
    vk::cmd_draw(cb, vertex_count, instance_count, 0, 0);
}

fn void? vk_cmd_draw_indexed(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan index_span, uint index_count, uint instance_count, gpu::IndexType index_type) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    usz index_size = index_element_size(index_type);
    BufferSlot* buffer = resolve_index_span(
        state:      state,
        span:       &index_span,
        index_type: index_type,
        required:   (usz)index_count * index_size,
    )!;

    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(commands, handle, vertex_root, fragment_root, cb)!;
    vk::cmd_bind_index_buffer(cb, buffer.buffer, (vk::DeviceSize)index_span.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
    vk::cmd_draw_indexed(cb, index_count, instance_count, 0, 0, 0);
}

fn void? vk_cmd_draw_indirect(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count) @private {
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DrawIndirectCommand::size,
        count:    draw_count,
    )!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(
        commands:      commands,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        cb:            cb,
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
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    validate_pipeline_kind(state, handle, PipelineKind.GRAPHICS)!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:    state,
        span:     &args,
        arg_size: gpu::DrawIndexedIndirectCommand::size,
        count:    draw_count,
    )!;
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(
        commands:      commands,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        cb:            cb,
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
    if (commands.state != gpu::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
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
    vk::CommandBuffer cb = (vk::CommandBuffer)commands.backend;
    bind_graphics_and_push(
        commands:      commands,
        handle:        handle,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        cb:            cb,
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
