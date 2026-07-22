module gpu::vk @private;

import gpu @public;
import vk;
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

fn vk::ResolveModeFlagBits color_resolve_mode(gpu::Format format) @private {
    switch (format) {
        case gpu::Format.R8_UINT:
        case gpu::Format.R16_UINT:
        case gpu::Format.R32_UINT:
            return vk::ResolveModeFlagBits.SAMPLE_ZERO_BIT;
        default:
            return vk::ResolveModeFlagBits.AVERAGE_BIT;
    }
}

fn vk::ClearColorValue clear_color_to_vk(
    gpu::Format format,
    gpu::ClearColor clear,
) @private {
    vk::ClearColorValue value;
    switch (format) {
        case gpu::Format.R8_UINT:
        case gpu::Format.R16_UINT:
        case gpu::Format.R32_UINT:
            value.uint32 = clear.uint_rgba;
        default:
            value.float32 = clear.rgba;
    }
    return value;
}

fn void emit_raster_state(
    CommandRecord* record,
    gpu::DynamicRasterState* raster,
) @private {
    vk::cmd_set_primitive_topology(
        record.command_buffer,
        topology_to_vk(raster.topology),
    );
    gpu::note_command_native_call();
    vk::cmd_set_cull_mode(
        record.command_buffer,
        (vk::CullModeFlags)cull_mode_to_vk(raster.cull_mode),
    );
    gpu::note_command_native_call();
    vk::cmd_set_front_face(
        record.command_buffer,
        front_face_to_vk(raster.front_face),
    );
    gpu::note_command_native_call();
    vk::cmd_set_depth_bias_enable(
        record.command_buffer,
        raster.depth_bias_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::note_command_native_call();
    float depth_bias_constant = 0.0f;
    float depth_bias_clamp = 0.0f;
    float depth_bias_slope = 0.0f;
    if (raster.depth_bias_enable) {
        depth_bias_constant = raster.depth_bias_constant;
        depth_bias_clamp = raster.depth_bias_clamp;
        depth_bias_slope = raster.depth_bias_slope;
    }
    vk::cmd_set_depth_bias(
        command_buffer:             record.command_buffer,
        depth_bias_constant_factor: depth_bias_constant,
        depth_bias_clamp:           depth_bias_clamp,
        depth_bias_slope_factor:    depth_bias_slope,
    );
    gpu::note_command_native_call();
}

<* Begin dynamic rendering with precreated attachment views. Attachments must
   sit in their attachment layouts, have matching usage, and cover the pass
   dimensions. *>
fn void? vk_cmd_begin_render_pass(
    gpu::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(
        state,
        record,
        reference_checkpoint,
    );
    if (desc.colors.len == 0 && desc.depth == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.width == 0 || desc.height == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.colors.len > state.caps.max_color_attachments) {
        return gpu::INVALID_ARGUMENT~;
    }
    vk::CommandBuffer cb = record.command_buffer;
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] color_slots;
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] resolve_slots;
    vk::RenderingAttachmentInfo[gpu::MAX_COLOR_ATTACHMENTS] attachments;
    RenderCompatibility compatibility;
    compatibility.color_count = (uint)desc.colors.len;
    bool samples_initialized;

    foreach (i, &target : desc.colors) {
        AttachmentViewSlot* view = state.attachment_views.get(target.view)!;
        TextureSlot* texture = state.textures.get(view.texture)!;
        validate_texture_queue_access(
            state:       state,
            queue_roles: record.queue.roles,
            slot:        texture,
            handle:      view.texture,
            operation:   "cmd_begin_render_pass",
        )!;
        track_command_reference(
            state:      state,
            record:     record,
            owner:      target.view.owner,
            index:      target.view.index,
            generation: target.view.generation,
        )!;
        if (!texture.usage.color_attach) {
            return report_contract_failure(
                state:               state,
                operation:           "cmd_begin_render_pass",
                public_fault:        gpu::INVALID_ARGUMENT,
                resource_kind:       gpu::DebugResourceKind.TEXTURE,
                has_public_identity: true,
                index:               view.texture.index,
                generation:          view.texture.generation,
                debug_name:          slot_name(&texture.debug_name[0]),
                rejected_field:      "TextureUsage.color_attach",
                invariant:           "color attachment must include color_attach usage",
                backend_text:        null,
            )~;
        }
        if (desc.width > view.width || desc.height > view.height) {
            ZString rejected_field = desc.width > view.width
                ? "RenderPassDesc.width"
                : "RenderPassDesc.height";
            return report_contract_failure(
                state:               state,
                operation:           "cmd_begin_render_pass",
                public_fault:        gpu::INVALID_ARGUMENT,
                resource_kind:       gpu::DebugResourceKind.TEXTURE,
                has_public_identity: true,
                index:               view.texture.index,
                generation:          view.texture.generation,
                debug_name:          slot_name(&texture.debug_name[0]),
                rejected_field:      rejected_field,
                invariant:           "render area must fit every color attachment mip",
                backend_text:        null,
            )~;
        }
        if (!samples_initialized) {
            compatibility.sample_count = view.sample_count;
            samples_initialized = true;
        } else if (view.sample_count != compatibility.sample_count) {
            return gpu::INVALID_ARGUMENT~;
        }
        compatibility.color_formats[i] = view.format;
        color_slots[i] = view;
        resolve_slots[i] = null;

        if (target.resolve_view.is_valid()) {
            if (view.sample_count == gpu::SampleCount.ONE) {
                return gpu::INVALID_ARGUMENT~;
            }
            AttachmentViewSlot* resolve = state.attachment_views.get(
                target.resolve_view,
            )!;
            TextureSlot* resolve_texture = state.textures.get(
                resolve.texture,
            )!;
            validate_texture_queue_access(
                state:       state,
                queue_roles: record.queue.roles,
                slot:        resolve_texture,
                handle:      resolve.texture,
                operation:   "cmd_begin_render_pass",
            )!;
            track_command_reference(
                state:      state,
                record:     record,
                owner:      target.resolve_view.owner,
                index:      target.resolve_view.index,
                generation: target.resolve_view.generation,
            )!;
            if (!resolve_texture.usage.color_attach
                || resolve.sample_count != gpu::SampleCount.ONE
                || resolve.format != view.format
                || desc.width > resolve.width
                || desc.height > resolve.height) {
                return gpu::INVALID_ARGUMENT~;
            }
            resolve_slots[i] = resolve;
        }
    }

    AttachmentViewSlot* depth_view = null;
    if (desc.depth != null) {
        depth_view = state.attachment_views.get(desc.depth.view)!;
        TextureSlot* depth_texture = state.textures.get(depth_view.texture)!;
        validate_texture_queue_access(
            state:       state,
            queue_roles: record.queue.roles,
            slot:        depth_texture,
            handle:      depth_view.texture,
            operation:   "cmd_begin_render_pass",
        )!;
        track_command_reference(
            state:      state,
            record:     record,
            owner:      desc.depth.view.owner,
            index:      desc.depth.view.index,
            generation: desc.depth.view.generation,
        )!;
        if (!depth_texture.usage.depth_attach
            || desc.width > depth_view.width
            || desc.height > depth_view.height) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (samples_initialized
            && depth_view.sample_count != compatibility.sample_count) {
            return gpu::INVALID_ARGUMENT~;
        }
        compatibility.depth_format = depth_view.format;
        if (!samples_initialized) {
            compatibility.sample_count = depth_view.sample_count;
            samples_initialized = true;
        }
    }

    validate_active_pipeline_render_compatibility(
        state,
        record,
        &compatibility,
    )!;

    foreach (i, &target : desc.colors) {
        AttachmentViewSlot* view = color_slots[i];
        attachments[i] = vk::rendering_attachment_info();
        attachments[i].image_view = view.native;
        attachments[i].image_layout =
            vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        attachments[i].load_op = load_op_to_vk(target.load_op);
        attachments[i].store_op = store_op_to_vk(target.store_op);
        attachments[i].clear_value.color = clear_color_to_vk(
            view.format,
            target.clear,
        );
        if (resolve_slots[i] != null) {
            attachments[i].resolve_mode = color_resolve_mode(view.format);
            attachments[i].resolve_image_view = resolve_slots[i].native;
            attachments[i].resolve_image_layout =
                vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    vk::RenderingAttachmentInfo depth_attachment;
    vk::RenderingInfo rendering = vk::rendering_info();
    rendering.render_area = {
        .offset = { .x = 0, .y = 0 },
        .extent = {
            .width  = desc.width,
            .height = desc.height,
        },
    };
    rendering.layer_count = 1;
    rendering.color_attachment_count = (uint)desc.colors.len;
    rendering.p_color_attachments = &attachments[0];
    if (desc.depth != null) {
        depth_attachment = vk::rendering_attachment_info();
        depth_attachment.image_view = depth_view.native;
        depth_attachment.image_layout =
            vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.load_op = load_op_to_vk(desc.depth.load_op);
        depth_attachment.store_op = store_op_to_vk(desc.depth.store_op);
        depth_attachment.clear_value.depth_stencil = {
            .depth   = desc.depth.clear.depth,
            .stencil = 0,
        };
        rendering.p_depth_attachment = &depth_attachment;
    }
    vk::cmd_begin_rendering(cb, &rendering);
    gpu::note_command_native_call();
    record.active_render = compatibility;

    vk::Viewport viewport = {
        .x         = 0.0,
        .y         = 0.0,
        .width     = (float)desc.width,
        .height    = (float)desc.height,
        .min_depth = 0.0,
        .max_depth = 1.0,
    };
    vk::cmd_set_viewport(cb, 0, 1, &viewport);
    gpu::note_command_native_call();
    vk::Rect2D scissor = {
        .offset = { .x = 0, .y = 0 },
        .extent = {
            .width  = desc.width,
            .height = desc.height,
        },
    };
    vk::cmd_set_scissor(cb, 0, 1, &scissor);
    gpu::note_command_native_call();

    gpu::DynamicRasterState raster = {};
    emit_raster_state(record, &raster);

    record.active_render_width = desc.width;
    record.active_render_height = desc.height;
    record.depth_state_set = false;
    record.state = CommandState.RECORDING_RENDER_PASS;
}

fn void? vk_cmd_set_viewport(gpu::CommandEncoderState* commands, gpu::Viewport* viewport) @private {
    CommandRecord* record = encoder_command(commands);
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
    gpu::note_command_native_call();
}

fn void? vk_cmd_set_scissor(gpu::CommandEncoderState* commands, gpu::ScissorRect* scissor) @private {
    CommandRecord* record = encoder_command(commands);
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
    gpu::note_command_native_call();
}

fn void? vk_cmd_end_render_pass(gpu::CommandEncoderState* commands) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    gpu::note_command_native_call();
    if (record.bound_pipeline.handle.is_valid()) {
        validate_bound_pipeline_identity(&record.bound_pipeline)!;
        if (record.bound_pipeline.kind == PipelineKind.GRAPHICS) {
            record.bound_pipeline = {};
        }
    }
    record.active_render_width = 0;
    record.active_render_height = 0;
    record.active_render = {};
    record.depth_state_set = false;
    record.state = CommandState.RECORDING;
}

fn void? vk_cmd_set_depth_state(
    gpu::CommandEncoderState* commands,
    gpu::DepthState* depth,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (depth == null || !compare_op_valid(depth.compare)) {
        return gpu::INVALID_ARGUMENT~;
    }
    vk::cmd_set_depth_test_enable(
        record.command_buffer,
        depth.test_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::note_command_native_call();
    vk::cmd_set_depth_write_enable(
        record.command_buffer,
        depth.write_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::note_command_native_call();
    vk::cmd_set_depth_compare_op(
        record.command_buffer,
        compare_op_to_vk(depth.compare),
    );
    gpu::note_command_native_call();
    record.depth_state_set = true;
}

fn void? vk_cmd_set_raster_state(
    gpu::CommandEncoderState* commands,
    gpu::DynamicRasterState* raster,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (raster == null
        || !primitive_topology_valid(raster.topology)
        || !cull_mode_valid(raster.cull_mode)
        || !front_face_valid(raster.front_face)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (raster.depth_bias_enable
        && (!math::is_finite(raster.depth_bias_constant)
            || !math::is_finite(raster.depth_bias_slope)
            || !math::is_finite(raster.depth_bias_clamp))) {
        return gpu::INVALID_ARGUMENT~;
    }
    emit_raster_state(record, raster);
}

fn BoundPipeline*? validate_graphics_execution_state(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    ZString operation,
) @private {
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     operation,
    )!;
    if (!record.depth_state_set) return gpu::COMMAND_RECORDING_ERROR~;
    return pipeline;
}

fn void push_graphics_roots(
    CommandRecord* record,
    BoundPipeline* pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
) @private {
    gpu::GraphicsRootPush push = {
        .vertex_root_gpu   = vertex_root,
        .fragment_root_gpu = fragment_root,
    };
    $if env::TESTING:
        last_vertex_root.store((ulong)push.vertex_root_gpu, AtomicOrdering.RELAXED);
        last_fragment_root.store((ulong)push.fragment_root_gpu, AtomicOrdering.RELAXED);
    $endif
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(
        command_buffer: record.command_buffer,
        layout:         pipeline.layout,
        stage_flags:    (vk::ShaderStageFlags)push_stages,
        offset:         0,
        size:           GRAPHICS_ROOT_PUSH_SIZE,
        p_values:       &push,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_draw(gpu::CommandEncoderState* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, uint vertex_count, uint instance_count) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw",
    )!;
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_draw(cb, vertex_count, instance_count, 0, 0);
    gpu::note_command_native_call();
}

fn void? vk_cmd_draw_indexed(gpu::CommandEncoderState* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan index_span, uint index_count, uint instance_count, gpu::IndexType index_type) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw_indexed",
    )!;
    usz index_size = index_element_size(index_type)!;
    ResolvedSpan buffer = resolve_index_span(
        state:      state,
        record:     record,
        span:       &index_span,
        index_type: index_type,
        required:   (usz)index_count * index_size,
        operation:  "cmd_draw_indexed",
    )!;

    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_bind_index_buffer(cb, buffer.slot.buffer, (vk::DeviceSize)buffer.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
    gpu::note_command_native_call();
    vk::cmd_draw_indexed(cb, index_count, instance_count, 0, 0, 0);
    gpu::note_command_native_call();
}

fn void? vk_cmd_draw_indirect(gpu::CommandEncoderState* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw_indirect",
    )!;
    ResolvedSpan args_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &args,
        arg_size:       gpu::DrawIndirectCommand::size,
        count:          draw_count,
        operation:      "cmd_draw_indirect",
        rejected_field: "draw_count",
        max_count:      state.caps.max_draw_indirect_count,
    )!;
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_draw_indirect(
        command_buffer: cb,
        buffer:         args_buffer.slot.buffer,
        offset:         (vk::DeviceSize)args_buffer.offset,
        draw_count:     draw_count,
        stride:         gpu::DrawIndirectCommand::size,
    );
    gpu::note_command_native_call();
}

fn void bind_index_buffer(
    vk::CommandBuffer cb,
    ResolvedSpan* index_span,
    gpu::IndexType index_type,
) @private {
    vk::cmd_bind_index_buffer(
        command_buffer: cb,
        buffer:         index_span.slot.buffer,
        offset:         (vk::DeviceSize)index_span.offset,
        index_type:     index_type == gpu::IndexType.U16
            ? vk::IndexType.UINT16
            : vk::IndexType.UINT32,
    );
    gpu::note_command_native_call();
}
<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn void? vk_cmd_draw_indexed_indirect(gpu::CommandEncoderState* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw_indexed_indirect",
    )!;
    ResolvedSpan args_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &args,
        arg_size:       gpu::DrawIndexedIndirectCommand::size,
        count:          draw_count,
        operation:      "cmd_draw_indexed_indirect",
        rejected_field: "draw_count",
        max_count:      state.caps.max_draw_indirect_count,
    )!;
    ResolvedSpan index_buffer = resolve_index_span(
        state:      state,
        record:     record,
        span:       &index_span,
        index_type: index_type,
        required:   0,
        operation:  "cmd_draw_indexed_indirect",
    )!;
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    bind_index_buffer(
        cb:         cb,
        index_span: &index_buffer,
        index_type: index_type,
    );
    vk::cmd_draw_indexed_indirect(
        command_buffer: cb,
        buffer:         args_buffer.slot.buffer,
        offset:         (vk::DeviceSize)args_buffer.offset,
        draw_count:     draw_count,
        stride:         gpu::DrawIndexedIndirectCommand::size,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_draw_indexed_indirect_count(gpu::CommandEncoderState* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, gpu::GpuSpan count_span, uint max_draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw_indexed_indirect_count",
    )!;
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    ResolvedSpan args_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &args,
        arg_size:       gpu::DrawIndexedIndirectCommand::size,
        count:          max_draw_count,
        operation:      "cmd_draw_indexed_indirect_count",
        rejected_field: "args",
    )!;
    ResolvedSpan count_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       4,
        count:          1,
        operation:      "cmd_draw_indexed_indirect_count",
        rejected_field: "count_span",
    )!;
    ResolvedSpan index_buffer = resolve_index_span(
        state:      state,
        record:     record,
        span:       &index_span,
        index_type: index_type,
        required:   0,
        operation:  "cmd_draw_indexed_indirect_count",
    )!;
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    bind_index_buffer(
        cb:         cb,
        index_span: &index_buffer,
        index_type: index_type,
    );
    vk::cmd_draw_indexed_indirect_count(
        cb,
        args_buffer.slot.buffer,
        (vk::DeviceSize)args_buffer.offset,
        count_buffer.slot.buffer,
        (vk::DeviceSize)count_buffer.offset,
        max_draw_count,
        gpu::DrawIndexedIndirectCommand::size,
    );
    gpu::note_command_native_call();
}

fn BoundPipeline*? validate_generated_graphics_execution_state(
    VkDeviceState* state,
    CommandRecord* record,
    ZString operation,
) @private {
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     operation,
    )!;
    if (!record.depth_state_set) return gpu::COMMAND_RECORDING_ERROR~;
    return pipeline;
}

fn void? vk_cmd_draw_generated(
    gpu::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    BoundPipeline* pipeline = validate_generated_graphics_execution_state(
        state,
        record,
        "cmd_draw_generated",
    )!;
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    validate_generated_work_count(
        state,
        max_draw_count,
        "cmd_draw_generated",
        "max_draw_count",
    )!;
    if (max_draw_count == 0) return;

    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan resolved_records = resolve_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDrawRecord::size,
        max_count:   max_draw_count,
        operation:   "cmd_draw_generated",
    )!;
    ResolvedSpan resolved_count = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       4,
        count:          1,
        operation:      "cmd_draw_generated",
        rejected_field: "count_span",
    )!;
    uint stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    execute_generated_work(
        state:         state,
        record:        record,
        pipeline:      pipeline,
        kind:          gpu::GeneratedWorkKind.DRAW,
        layout:        state.generated_draw_layout,
        shader_stages: (vk::ShaderStageFlags)stages,
        records:       &resolved_records,
        count:         &resolved_count,
        max_count:     max_draw_count,
    )!;
}

fn void? vk_cmd_draw_indexed_generated(
    gpu::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    BoundPipeline* pipeline = validate_generated_graphics_execution_state(
        state,
        record,
        "cmd_draw_indexed_generated",
    )!;
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    validate_generated_work_count(
        state,
        max_draw_count,
        "cmd_draw_indexed_generated",
        "max_draw_count",
    )!;
    (void)index_element_size(index_type)!;
    if (max_draw_count == 0) return;

    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan resolved_records = resolve_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDrawIndexedRecord::size,
        max_count:   max_draw_count,
        operation:   "cmd_draw_indexed_generated",
    )!;
    ResolvedSpan resolved_count = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       4,
        count:          1,
        operation:      "cmd_draw_indexed_generated",
        rejected_field: "count_span",
    )!;
    ResolvedSpan resolved_index = resolve_index_span(
        state:      state,
        record:     record,
        span:       &index_span,
        index_type: index_type,
        required:   0,
        operation:  "cmd_draw_indexed_generated",
    )!;
    bind_index_buffer(
        cb:         record.command_buffer,
        index_span: &resolved_index,
        index_type: index_type,
    );
    uint stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    execute_generated_work(
        state:         state,
        record:        record,
        pipeline:      pipeline,
        kind:          gpu::GeneratedWorkKind.DRAW_INDEXED,
        layout:        state.generated_indexed_draw_layout,
        shader_stages: (vk::ShaderStageFlags)stages,
        records:       &resolved_records,
        count:         &resolved_count,
        max_count:     max_draw_count,
    )!;
}
