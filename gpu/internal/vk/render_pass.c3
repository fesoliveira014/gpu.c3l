module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::math;

struct PreparedRenderPass {
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] color_slots;
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] resolve_slots;
    gpu::AttachmentViewHandle[gpu::MAX_COLOR_ATTACHMENTS] color_handles;
    gpu::AttachmentViewHandle[gpu::MAX_COLOR_ATTACHMENTS] resolve_handles;
    vk::RenderingAttachmentInfo[gpu::MAX_COLOR_ATTACHMENTS] attachments;
    AttachmentViewSlot*       depth_view;
    gpu::AttachmentViewHandle depth_handle;
    vk::RenderingAttachmentInfo depth_attachment;
    RenderCompatibility       compatibility;
    uint                      width;
    uint                      height;
    uint                      color_count;
    bool                      has_depth;
}
struct PreparedIndexedDraw {
    BoundPipeline   pipeline;
    ResolvedSpan    index;
    gpu::GpuSpan    index_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            index_count;
    uint            instance_count;
    gpu::IndexType  index_type;
}

struct PreparedIndirectDraw {
    BoundPipeline   pipeline;
    ResolvedSpan    args;
    gpu::GpuSpan    args_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
}

struct PreparedIndexedIndirectDraw {
    BoundPipeline   pipeline;
    ResolvedSpan    args;
    ResolvedSpan    index;
    gpu::GpuSpan    args_span;
    gpu::GpuSpan    index_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
    gpu::IndexType  index_type;
}

struct PreparedIndexedIndirectCountDraw {
    BoundPipeline   pipeline;
    ResolvedSpan    args;
    ResolvedSpan    count;
    ResolvedSpan    index;
    gpu::GpuSpan    args_span;
    gpu::GpuSpan    count_span;
    gpu::GpuSpan    index_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            max_draw_count;
    gpu::IndexType  index_type;
}

struct PreparedGeneratedDraw {
    BoundPipeline                  pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    gpu::GpuSpan                   records_span;
    gpu::GpuSpan                   count_span;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
}

struct PreparedGeneratedIndexedDraw {
    BoundPipeline                  pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    ResolvedSpan                   index;
    gpu::GpuSpan                   records_span;
    gpu::GpuSpan                   count_span;
    gpu::GpuSpan                   index_span;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
    gpu::IndexType                 index_type;
}

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
    gpu::internal::note_command_native_call();
    vk::cmd_set_cull_mode(
        record.command_buffer,
        (vk::CullModeFlags)cull_mode_to_vk(raster.cull_mode),
    );
    gpu::internal::note_command_native_call();
    vk::cmd_set_front_face(
        record.command_buffer,
        front_face_to_vk(raster.front_face),
    );
    gpu::internal::note_command_native_call();
    vk::cmd_set_depth_bias_enable(
        record.command_buffer,
        raster.depth_bias_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::internal::note_command_native_call();
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
    gpu::internal::note_command_native_call();
}

fn PreparedRenderPass? lower_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    if (desc == null
        || desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS
        || (desc.colors.len > 0 && desc.colors.ptr == null)) {
        return gpu::INVALID_ARGUMENT~;
    }
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    PreparedRenderPass pass = {
        .width       = desc.width,
        .height      = desc.height,
        .color_count = (uint)desc.colors.len,
        .has_depth   = desc.depth != null,
    };
    pass.compatibility.color_count = pass.color_count;
    bool samples_initialized;
    foreach (i, &target : desc.colors) {
        AttachmentViewSlot* view = state.attachment_views.get(target.view)!;
        if (view.native == null) return gpu::INVALID_RESOURCE_STATE~;
        pass.color_slots[i] = view;
        pass.color_handles[i] = target.view;
        pass.compatibility.color_formats[i] = view.format;
        if (!samples_initialized) {
            pass.compatibility.sample_count = view.sample_count;
            samples_initialized = true;
        }
        vk::RenderingAttachmentInfo* attachment = &pass.attachments[i];
        *attachment = vk::rendering_attachment_info();
        attachment.image_view = view.native;
        attachment.image_layout = vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        attachment.load_op = load_op_to_vk(target.load_op);
        attachment.store_op = store_op_to_vk(target.store_op);
        attachment.clear_value.color = clear_color_to_vk(
            view.format,
            target.clear,
        );

        if (target.resolve_view.is_valid()) {
            AttachmentViewSlot* resolve = state.attachment_views.get(
                target.resolve_view,
            )!;
            if (resolve.native == null) return gpu::INVALID_RESOURCE_STATE~;
            pass.resolve_slots[i] = resolve;
            pass.resolve_handles[i] = target.resolve_view;
            attachment.resolve_mode = color_resolve_mode(view.format);
            attachment.resolve_image_view = resolve.native;
            attachment.resolve_image_layout =
                vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    if (desc.depth != null) {
        AttachmentViewSlot* depth = state.attachment_views.get(
            desc.depth.view,
        )!;
        if (depth.native == null) return gpu::INVALID_RESOURCE_STATE~;
        pass.depth_view = depth;
        pass.depth_handle = desc.depth.view;
        pass.compatibility.depth_format = depth.format;
        if (!samples_initialized) {
            pass.compatibility.sample_count = depth.sample_count;
        }
        pass.depth_attachment = vk::rendering_attachment_info();
        pass.depth_attachment.image_view = depth.native;
        pass.depth_attachment.image_layout =
            vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        pass.depth_attachment.load_op = load_op_to_vk(desc.depth.load_op);
        pass.depth_attachment.store_op = store_op_to_vk(desc.depth.store_op);
        pass.depth_attachment.clear_value.depth_stencil = {
            .depth   = desc.depth.clear.depth,
            .stencil = 0,
        };
    }
    return pass;
}

fn void? validate_render_pass_desc_fields(
    VkDeviceState* state,
    gpu::RenderPassDesc* desc,
) {
    if (!gpu::internal::render_pass_desc_valid(desc)
        || desc.colors.len > state.caps.max_color_attachments) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn PreparedRenderPass? validate_and_lower_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    if (commands == null || desc == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    validate_render_pass_desc_fields(state, desc)!;
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;

    foreach (i, &target : desc.colors) {
        AttachmentViewSlot* view = pass.color_slots[i];
        TextureSlot* texture = state.textures.get(view.texture)!;
        validate_texture_queue_access(
            state:       state,
            queue_roles: record.queue.roles,
            slot:        texture,
            handle:      view.texture,
            operation:   "cmd_begin_render_pass",
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
        if (i > 0
            && view.sample_count != pass.compatibility.sample_count) {
            return gpu::INVALID_ARGUMENT~;
        }

        AttachmentViewSlot* resolve = pass.resolve_slots[i];
        if (resolve != null) {
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
            if (view.sample_count == gpu::SampleCount.ONE
                || !resolve_texture.usage.color_attach
                || resolve.sample_count != gpu::SampleCount.ONE
                || resolve.format != view.format
                || desc.width > resolve.width
                || desc.height > resolve.height) {
                return gpu::INVALID_ARGUMENT~;
            }
        }
    }

    if (pass.has_depth) {
        AttachmentViewSlot* depth = pass.depth_view;
        TextureSlot* depth_texture = state.textures.get(depth.texture)!;
        validate_texture_queue_access(
            state:       state,
            queue_roles: record.queue.roles,
            slot:        depth_texture,
            handle:      depth.texture,
            operation:   "cmd_begin_render_pass",
        )!;
        if (!depth_texture.usage.depth_attach
            || desc.width > depth.width
            || desc.height > depth.height) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (pass.color_count > 0
            && depth.sample_count != pass.compatibility.sample_count) {
            return gpu::INVALID_ARGUMENT~;
        }
    }
    validate_active_pipeline_render_compatibility(
        state,
        record,
        &pass.compatibility,
    )!;
    return pass;
}

fn void? track_render_pass(
    gpu::internal::CommandEncoderState* commands,
    PreparedRenderPass* pass,
) {
    VkDeviceState* state = encoder_device_state(commands);
    CommandRecord* record = encoder_command(commands);
    CommandResourceReference[
        gpu::MAX_COLOR_ATTACHMENTS * 2 + 1
    ] references;
    uint reference_count;
    for (uint i = 0; i < pass.color_count; i++) {
        gpu::AttachmentViewHandle color = pass.color_handles[i];
        references[reference_count++] = {
            .owner      = color.owner,
            .index      = color.index,
            .generation = color.generation,
        };
        gpu::AttachmentViewHandle resolve = pass.resolve_handles[i];
        if (resolve.is_valid()) {
            references[reference_count++] = {
                .owner      = resolve.owner,
                .index      = resolve.index,
                .generation = resolve.generation,
            };
        }
    }
    if (pass.has_depth) {
        references[reference_count++] = {
            .owner      = pass.depth_handle.owner,
            .index      = pass.depth_handle.index,
            .generation = pass.depth_handle.generation,
        };
    }
    preflight_command_references(
        state,
        record,
        references[:reference_count],
    )!;
    for (uint i = 0; i < pass.color_count; i++) {
        gpu::AttachmentViewHandle color = pass.color_handles[i];
        track_command_reference(
            state,
            record,
            color.owner,
            color.index,
            color.generation,
        )!;
        gpu::AttachmentViewHandle resolve = pass.resolve_handles[i];
        if (resolve.is_valid()) {
            track_command_reference(
                state,
                record,
                resolve.owner,
                resolve.index,
                resolve.generation,
            )!;
        }
    }
    if (pass.has_depth) {
        track_command_reference(
            state,
            record,
            pass.depth_handle.owner,
            pass.depth_handle.index,
            pass.depth_handle.generation,
        )!;
    }
}

fn void record_render_pass(
    gpu::internal::CommandEncoderState* commands,
    PreparedRenderPass* pass,
) {
    CommandRecord* record = encoder_command(commands);
    vk::CommandBuffer cb = record.command_buffer;
    vk::RenderingInfo rendering = vk::rendering_info();
    rendering.render_area = {
        .offset = { .x = 0, .y = 0 },
        .extent = { .width = pass.width, .height = pass.height },
    };
    rendering.layer_count = 1;
    rendering.color_attachment_count = pass.color_count;
    rendering.p_color_attachments = pass.color_count == 0
        ? null
        : &pass.attachments[0];
    if (pass.has_depth) {
        rendering.p_depth_attachment = &pass.depth_attachment;
    }
    vk::cmd_begin_rendering(cb, &rendering);
    gpu::internal::note_command_native_call();
    record.active_render = pass.compatibility;

    vk::Viewport viewport = {
        .x         = 0.0,
        .y         = 0.0,
        .width     = (float)pass.width,
        .height    = (float)pass.height,
        .min_depth = 0.0,
        .max_depth = 1.0,
    };
    vk::cmd_set_viewport(cb, 0, 1, &viewport);
    gpu::internal::note_command_native_call();
    vk::Rect2D scissor = {
        .offset = { .x = 0, .y = 0 },
        .extent = { .width = pass.width, .height = pass.height },
    };
    vk::cmd_set_scissor(cb, 0, 1, &scissor);
    gpu::internal::note_command_native_call();
    gpu::DynamicRasterState raster;
    emit_raster_state(record, &raster);

    record.active_render_width = pass.width;
    record.active_render_height = pass.height;
    record.depth_state_set = false;
    record.state = CommandState.RECORDING_RENDER_PASS;
}

fn void? trusted_begin_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    record_render_pass(commands, &pass);
}

fn void? trusted_tracking_begin_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass(commands, &pass);
}

fn void? checked_begin_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = validate_and_lower_render_pass(
        commands,
        desc,
    )!;
    record_render_pass(commands, &pass);
}

fn void? checked_tracking_begin_render_pass(
    gpu::internal::CommandEncoderState* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = validate_and_lower_render_pass(
        commands,
        desc,
    )!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass(commands, &pass);
}

fn void? vk_cmd_set_viewport(gpu::internal::CommandEncoderState* commands, gpu::Viewport* viewport) @private {
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
    gpu::internal::note_command_native_call();
}

fn void? trusted_set_viewport(
    gpu::internal::CommandEncoderState* commands,
    gpu::Viewport* viewport,
) {
    if (viewport == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
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
    gpu::internal::note_command_native_call();
}

fn void? checked_set_viewport(
    gpu::internal::CommandEncoderState* commands,
    gpu::Viewport* viewport,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(encoder_device_state(commands));
    return vk_cmd_set_viewport(commands, viewport);
}

fn void? vk_cmd_set_scissor(gpu::internal::CommandEncoderState* commands, gpu::ScissorRect* scissor) @private {
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
    gpu::internal::note_command_native_call();
}

fn void? trusted_set_scissor(
    gpu::internal::CommandEncoderState* commands,
    gpu::ScissorRect* scissor,
) {
    if (scissor == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::Rect2D vk_scissor = {
        .offset = { .x = scissor.x, .y = scissor.y },
        .extent = {
            .width  = (uint)scissor.width,
            .height = (uint)scissor.height,
        },
    };
    vk::cmd_set_scissor(record.command_buffer, 0, 1, &vk_scissor);
    gpu::internal::note_command_native_call();
}

fn void? checked_set_scissor(
    gpu::internal::CommandEncoderState* commands,
    gpu::ScissorRect* scissor,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(encoder_device_state(commands));
    return vk_cmd_set_scissor(commands, scissor);
}

fn void? vk_cmd_end_render_pass(gpu::internal::CommandEncoderState* commands) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    gpu::internal::note_command_native_call();
    if (record.bound_pipeline.handle.is_valid()
        && record.bound_pipeline.kind == PipelineKind.GRAPHICS) {
        record.bound_pipeline = {};
    }
    record.active_render_width = 0;
    record.active_render_height = 0;
    record.active_render = {};
    record.depth_state_set = false;
    record.state = CommandState.RECORDING;
}

fn void? vk_cmd_set_depth_state(
    gpu::internal::CommandEncoderState* commands,
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
    gpu::internal::note_command_native_call();
    vk::cmd_set_depth_write_enable(
        record.command_buffer,
        depth.write_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::internal::note_command_native_call();
    vk::cmd_set_depth_compare_op(
        record.command_buffer,
        compare_op_to_vk(depth.compare),
    );
    gpu::internal::note_command_native_call();
    record.depth_state_set = true;
}

fn void? trusted_set_depth_state(
    gpu::internal::CommandEncoderState* commands,
    gpu::DepthState* depth,
) {
    if (depth == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::cmd_set_depth_test_enable(
        record.command_buffer,
        depth.test_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::internal::note_command_native_call();
    vk::cmd_set_depth_write_enable(
        record.command_buffer,
        depth.write_enable ? vk::TRUE : vk::FALSE,
    );
    gpu::internal::note_command_native_call();
    vk::cmd_set_depth_compare_op(
        record.command_buffer,
        compare_op_to_vk(depth.compare),
    );
    gpu::internal::note_command_native_call();
    record.depth_state_set = true;
}

fn void? checked_set_depth_state(
    gpu::internal::CommandEncoderState* commands,
    gpu::DepthState* depth,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(encoder_device_state(commands));
    return vk_cmd_set_depth_state(commands, depth);
}

fn void? vk_cmd_set_raster_state(
    gpu::internal::CommandEncoderState* commands,
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

fn void? trusted_set_raster_state(
    gpu::internal::CommandEncoderState* commands,
    gpu::DynamicRasterState* raster,
) {
    if (raster == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    emit_raster_state(record, raster);
}

fn void? checked_set_raster_state(
    gpu::internal::CommandEncoderState* commands,
    gpu::DynamicRasterState* raster,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(encoder_device_state(commands));
    return vk_cmd_set_raster_state(commands, raster);
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
    gpu::internal::note_command_native_call();
}

fn BoundPipeline? graphics_pipeline_snapshot(CommandRecord* record) {
    if (!record.depth_state_set) return gpu::COMMAND_RECORDING_ERROR~;
    return bound_pipeline_snapshot(record);
}

fn void record_draw(
    gpu::internal::CommandEncoderState* commands,
    BoundPipeline* pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    CommandRecord* record = encoder_command(commands);
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::cmd_draw(
        command_buffer: record.command_buffer,
        vertex_count:   vertex_count,
        instance_count: instance_count,
        first_vertex:   0,
        first_instance: 0,
    );
    gpu::internal::note_command_native_call();
}

fn void? trusted_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    BoundPipeline pipeline = graphics_pipeline_snapshot(record)!;
    record_draw(
        commands:      commands,
        pipeline:      &pipeline,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        vertex_count:  vertex_count,
        instance_count: instance_count,
    );
}

fn void? checked_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    BoundPipeline pipeline = *validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw",
    )!;
    record_draw(
        commands:       commands,
        pipeline:       &pipeline,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   vertex_count,
        instance_count: instance_count,
    );
}

fn PreparedIndexedDraw? lower_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    usz required = gpu::internal::indirect_argument_bytes(
        argument_size: trusted_index_element_size(index_type),
        count:         index_count,
        max_count:     uint::max,
    )!;
    return {
        .pipeline      = graphics_pipeline_snapshot(record)!,
        .index         = lower_index_span(
            state:      state,
            span:       &index_span,
            index_type: index_type,
            required:   required,
        )!,
        .index_span    = index_span,
        .vertex_root   = vertex_root,
        .fragment_root = fragment_root,
        .index_count   = index_count,
        .instance_count = instance_count,
        .index_type    = index_type,
    };
}

fn PreparedIndexedDraw? validate_and_lower_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    usz index_size = index_element_size(index_type)!;
    return {
        .pipeline = *validate_graphics_execution_state(
            state:         state,
            record:        record,
            vertex_root:   vertex_root,
            fragment_root: fragment_root,
            operation:     "cmd_draw_indexed",
        )!,
        .index = validate_and_lower_index_span(
            state:      state,
            record:     record,
            span:       &index_span,
            index_type: index_type,
            required:   (usz)index_count * index_size,
            operation:  "cmd_draw_indexed",
        )!,
        .index_span     = index_span,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .index_count    = index_count,
        .instance_count = instance_count,
        .index_type     = index_type,
    };
}

fn void? track_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedDraw* draw,
) {
    return track_span_command_reference(
        encoder_device_state(commands),
        encoder_command(commands),
        &draw.index_span,
    );
}

fn void record_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedDraw* draw,
) {
    CommandRecord* record = encoder_command(commands);
    push_graphics_roots(
        record,
        &draw.pipeline,
        draw.vertex_root,
        draw.fragment_root,
    );
    bind_index_buffer(
        cb:         record.command_buffer,
        index_span: &draw.index,
        index_type: draw.index_type,
    );
    vk::cmd_draw_indexed(
        command_buffer: record.command_buffer,
        index_count:    draw.index_count,
        instance_count: draw.instance_count,
        first_index:    0,
        vertex_offset:  0,
        first_instance: 0,
    );
    gpu::internal::note_command_native_call();
}

fn void? trusted_draw_indexed(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = lower_indexed_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    )!;
    record_indexed_draw(commands, &draw);
}

fn void? trusted_tracking_draw_indexed(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = lower_indexed_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    )!;
    track_indexed_draw(commands, &draw)!;
    record_indexed_draw(commands, &draw);
}

fn void? checked_draw_indexed(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = validate_and_lower_indexed_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    )!;
    record_indexed_draw(commands, &draw);
}

fn void? checked_tracking_draw_indexed(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = validate_and_lower_indexed_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    )!;
    track_indexed_draw(commands, &draw)!;
    record_indexed_draw(commands, &draw);
}

fn PreparedIndirectDraw? lower_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    return {
        .pipeline = graphics_pipeline_snapshot(record)!,
        .args = lower_indirect_span(
            state:    state,
            span:     &args,
            arg_size: gpu::DrawIndirectCommand::size,
            count:    draw_count,
        )!,
        .args_span     = args,
        .vertex_root   = vertex_root,
        .fragment_root = fragment_root,
        .draw_count    = draw_count,
    };
}

fn PreparedIndirectDraw? validate_and_lower_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    return {
        .pipeline = *validate_graphics_execution_state(
            state:         state,
            record:        record,
            vertex_root:   vertex_root,
            fragment_root: fragment_root,
            operation:     "cmd_draw_indirect",
        )!,
        .args = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &args,
            arg_size:       gpu::DrawIndirectCommand::size,
            count:          draw_count,
            operation:      "cmd_draw_indirect",
            rejected_field: "draw_count",
            max_count:      state.caps.max_draw_indirect_count,
        )!,
        .args_span     = args,
        .vertex_root   = vertex_root,
        .fragment_root = fragment_root,
        .draw_count    = draw_count,
    };
}

fn void? track_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndirectDraw* draw,
) {
    return track_span_command_reference(
        encoder_device_state(commands),
        encoder_command(commands),
        &draw.args_span,
    );
}

fn void record_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndirectDraw* draw,
) {
    CommandRecord* record = encoder_command(commands);
    push_graphics_roots(
        record,
        &draw.pipeline,
        draw.vertex_root,
        draw.fragment_root,
    );
    vk::cmd_draw_indirect(
        command_buffer: record.command_buffer,
        buffer:         draw.args.slot.buffer,
        offset:         (vk::DeviceSize)draw.args.offset,
        draw_count:     draw.draw_count,
        stride:         gpu::DrawIndirectCommand::size,
    );
    gpu::internal::note_command_native_call();
}

fn void? trusted_draw_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = lower_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
    )!;
    record_indirect_draw(commands, &draw);
}

fn void? trusted_tracking_draw_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = lower_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
    )!;
    track_indirect_draw(commands, &draw)!;
    record_indirect_draw(commands, &draw);
}

fn void? checked_draw_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = validate_and_lower_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
    )!;
    record_indirect_draw(commands, &draw);
}

fn void? checked_tracking_draw_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = validate_and_lower_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
    )!;
    track_indirect_draw(commands, &draw)!;
    record_indirect_draw(commands, &draw);
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
    gpu::internal::note_command_native_call();
}

fn PreparedIndexedIndirectDraw? lower_indexed_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    return {
        .pipeline = graphics_pipeline_snapshot(record)!,
        .args = lower_indirect_span(
            state:    state,
            span:     &args,
            arg_size: gpu::DrawIndexedIndirectCommand::size,
            count:    draw_count,
        )!,
        .index = lower_index_span(
            state:      state,
            span:       &index_span,
            index_type: index_type,
            required:   0,
        )!,
        .args_span      = args,
        .index_span     = index_span,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .draw_count     = draw_count,
        .index_type     = index_type,
    };
}

fn PreparedIndexedIndirectDraw? validate_and_lower_indexed_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    return {
        .pipeline = *validate_graphics_execution_state(
            state:         state,
            record:        record,
            vertex_root:   vertex_root,
            fragment_root: fragment_root,
            operation:     "cmd_draw_indexed_indirect",
        )!,
        .args = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &args,
            arg_size:       gpu::DrawIndexedIndirectCommand::size,
            count:          draw_count,
            operation:      "cmd_draw_indexed_indirect",
            rejected_field: "draw_count",
            max_count:      state.caps.max_draw_indirect_count,
        )!,
        .index = validate_and_lower_index_span(
            state:      state,
            record:     record,
            span:       &index_span,
            index_type: index_type,
            required:   0,
            operation:  "cmd_draw_indexed_indirect",
        )!,
        .args_span      = args,
        .index_span     = index_span,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .draw_count     = draw_count,
        .index_type     = index_type,
    };
}

fn void? track_indexed_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    VkDeviceState* state = encoder_device_state(commands);
    CommandRecord* record = encoder_command(commands);
    CommandResourceReference[2] references = {
        {
            .owner      = draw.args_span.owner,
            .index      = draw.args_span.index,
            .generation = draw.args_span.generation,
        },
        {
            .owner      = draw.index_span.owner,
            .index      = draw.index_span.index,
            .generation = draw.index_span.generation,
        },
    };
    preflight_command_references(state, record, references[..])!;
    track_span_command_reference(state, record, &draw.args_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void record_indexed_indirect_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    CommandRecord* record = encoder_command(commands);
    push_graphics_roots(
        record,
        &draw.pipeline,
        draw.vertex_root,
        draw.fragment_root,
    );
    bind_index_buffer(
        cb:         record.command_buffer,
        index_span: &draw.index,
        index_type: draw.index_type,
    );
    vk::cmd_draw_indexed_indirect(
        command_buffer: record.command_buffer,
        buffer:         draw.args.slot.buffer,
        offset:         (vk::DeviceSize)draw.args.offset,
        draw_count:     draw.draw_count,
        stride:         gpu::DrawIndexedIndirectCommand::size,
    );
    gpu::internal::note_command_native_call();
}

fn void? trusted_draw_indexed_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw = lower_indexed_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
        index_span:     index_span,
        index_type:     index_type,
    )!;
    record_indexed_indirect_draw(commands, &draw);
}

fn void? trusted_tracking_draw_indexed_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw = lower_indexed_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
        index_span:     index_span,
        index_type:     index_type,
    )!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_draw(commands, &draw)!;
    record_indexed_indirect_draw(commands, &draw);
}

fn void? checked_draw_indexed_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw =
        validate_and_lower_indexed_indirect_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            draw_count:     draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    record_indexed_indirect_draw(commands, &draw);
}

fn void? checked_tracking_draw_indexed_indirect(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw =
        validate_and_lower_indexed_indirect_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            draw_count:     draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_draw(commands, &draw)!;
    record_indexed_indirect_draw(commands, &draw);
}

<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn PreparedIndexedIndirectCountDraw? lower_indexed_indirect_count_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    return {
        .pipeline = graphics_pipeline_snapshot(record)!,
        .args = lower_indirect_span(
            state:    state,
            span:     &args,
            arg_size: gpu::DrawIndexedIndirectCommand::size,
            count:    max_draw_count,
        )!,
        .count = lower_indirect_span(
            state:    state,
            span:     &count_span,
            arg_size: uint::size,
            count:    1,
        )!,
        .index = lower_index_span(
            state:      state,
            span:       &index_span,
            index_type: index_type,
            required:   0,
        )!,
        .args_span      = args,
        .count_span     = count_span,
        .index_span     = index_span,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .max_draw_count = max_draw_count,
        .index_type     = index_type,
    };
}

fn PreparedIndexedIndirectCountDraw? validate_and_lower_indexed_indirect_count_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    return {
        .pipeline = *validate_graphics_execution_state(
            state:         state,
            record:        record,
            vertex_root:   vertex_root,
            fragment_root: fragment_root,
            operation:     "cmd_draw_indexed_indirect_count",
        )!,
        .args = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &args,
            arg_size:       gpu::DrawIndexedIndirectCommand::size,
            count:          max_draw_count,
            operation:      "cmd_draw_indexed_indirect_count",
            rejected_field: "args",
        )!,
        .count = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &count_span,
            arg_size:       uint::size,
            count:          1,
            operation:      "cmd_draw_indexed_indirect_count",
            rejected_field: "count_span",
        )!,
        .index = validate_and_lower_index_span(
            state:      state,
            record:     record,
            span:       &index_span,
            index_type: index_type,
            required:   0,
            operation:  "cmd_draw_indexed_indirect_count",
        )!,
        .args_span      = args,
        .count_span     = count_span,
        .index_span     = index_span,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .max_draw_count = max_draw_count,
        .index_type     = index_type,
    };
}

fn void? track_indexed_indirect_count_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    VkDeviceState* state = encoder_device_state(commands);
    CommandRecord* record = encoder_command(commands);
    CommandResourceReference[3] references = {
        {
            .owner      = draw.args_span.owner,
            .index      = draw.args_span.index,
            .generation = draw.args_span.generation,
        },
        {
            .owner      = draw.count_span.owner,
            .index      = draw.count_span.index,
            .generation = draw.count_span.generation,
        },
        {
            .owner      = draw.index_span.owner,
            .index      = draw.index_span.index,
            .generation = draw.index_span.generation,
        },
    };
    preflight_command_references(state, record, references[..])!;
    track_span_command_reference(state, record, &draw.args_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void record_indexed_indirect_count_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    CommandRecord* record = encoder_command(commands);
    push_graphics_roots(
        record,
        &draw.pipeline,
        draw.vertex_root,
        draw.fragment_root,
    );
    bind_index_buffer(
        cb:         record.command_buffer,
        index_span: &draw.index,
        index_type: draw.index_type,
    );
    vk::cmd_draw_indexed_indirect_count(
        command_buffer:      record.command_buffer,
        buffer:              draw.args.slot.buffer,
        offset:              (vk::DeviceSize)draw.args.offset,
        count_buffer:        draw.count.slot.buffer,
        count_buffer_offset: (vk::DeviceSize)draw.count.offset,
        max_draw_count:      draw.max_draw_count,
        stride:              gpu::DrawIndexedIndirectCommand::size,
    );
    gpu::internal::note_command_native_call();
}

fn void? trusted_draw_indexed_indirect_count(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        lower_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    record_indexed_indirect_count_draw(commands, &draw);
}

fn void? trusted_tracking_draw_indexed_indirect_count(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        lower_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_count_draw(commands, &draw)!;
    record_indexed_indirect_count_draw(commands, &draw);
}

fn void? checked_draw_indexed_indirect_count(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        validate_and_lower_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    record_indexed_indirect_count_draw(commands, &draw);
}

fn void? checked_tracking_draw_indexed_indirect_count(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        validate_and_lower_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_count_draw(commands, &draw)!;
    record_indexed_indirect_count_draw(commands, &draw);
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

fn vk::ShaderStageFlags generated_graphics_shader_stages() =>
    (vk::ShaderStageFlags)((uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT);

fn PreparedGeneratedDraw? lower_generated_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    PreparedGeneratedDraw draw = {
        .pipeline     = graphics_pipeline_snapshot(record)!,
        .records_span = records,
        .count_span   = count_span,
        .layout       = state.generated_draw_layout,
        .max_count    = max_draw_count,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    if (draw.layout == null) return gpu::BACKEND_ERROR~;
    draw.records = lower_generated_work_records(
        state:       state,
        records:     &records,
        record_size: gpu::GeneratedDrawRecord::size,
        max_count:   max_draw_count,
    )!;
    draw.count = lower_indirect_span(
        state:    state,
        span:     &count_span,
        arg_size: uint::size,
        count:    1,
    )!;
    return draw;
}

fn PreparedGeneratedDraw? validate_and_lower_generated_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    PreparedGeneratedDraw draw = {
        .pipeline = *validate_generated_graphics_execution_state(
            state,
            record,
            "cmd_draw_generated",
        )!,
        .records_span = records,
        .count_span   = count_span,
        .layout       = state.generated_draw_layout,
        .max_count    = max_draw_count,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    if (draw.layout == null) return gpu::BACKEND_ERROR~;
    draw.records = validate_and_lower_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDrawRecord::size,
        max_count:   max_draw_count,
        operation:   "cmd_draw_generated",
    )!;
    draw.count = validate_and_lower_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       uint::size,
        count:          1,
        operation:      "cmd_draw_generated",
        rejected_field: "count_span",
    )!;
    return draw;
}

fn void? track_generated_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedGeneratedDraw* draw,
) {
    VkDeviceState* state = encoder_device_state(commands);
    CommandRecord* record = encoder_command(commands);
    CommandResourceReference[2] references = {
        {
            .owner      = draw.records_span.owner,
            .index      = draw.records_span.index,
            .generation = draw.records_span.generation,
        },
        {
            .owner      = draw.count_span.owner,
            .index      = draw.count_span.index,
            .generation = draw.count_span.generation,
        },
    };
    preflight_command_references(state, record, references[..])!;
    track_span_command_reference(state, record, &draw.records_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
}

fn void? record_generated_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedGeneratedDraw* draw,
) {
    if (draw.max_count == 0) return;
    execute_generated_work(
        state:         encoder_device_state(commands),
        record:        encoder_command(commands),
        pipeline:      &draw.pipeline,
        kind:          gpu::GeneratedWorkKind.DRAW,
        layout:        draw.layout,
        shader_stages: generated_graphics_shader_stages(),
        records:       &draw.records,
        count:         &draw.count,
        max_count:     draw.max_count,
    )!;
}

fn void? trusted_draw_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = lower_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    record_generated_draw(commands, &draw)!;
}

fn void? trusted_tracking_draw_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = lower_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_draw(commands, &draw)!;
    record_generated_draw(commands, &draw)!;
}

fn void? checked_draw_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = validate_and_lower_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    record_generated_draw(commands, &draw)!;
}

fn void? checked_tracking_draw_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = validate_and_lower_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_draw(commands, &draw)!;
    record_generated_draw(commands, &draw)!;
}

fn PreparedGeneratedIndexedDraw? lower_generated_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    PreparedGeneratedIndexedDraw draw = {
        .pipeline     = graphics_pipeline_snapshot(record)!,
        .records_span = records,
        .count_span   = count_span,
        .index_span   = index_span,
        .layout       = state.generated_indexed_draw_layout,
        .max_count    = max_draw_count,
        .index_type   = index_type,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    if (draw.layout == null) return gpu::BACKEND_ERROR~;
    draw.records = lower_generated_work_records(
        state:       state,
        records:     &records,
        record_size: gpu::GeneratedDrawIndexedRecord::size,
        max_count:   max_draw_count,
    )!;
    draw.count = lower_indirect_span(
        state:    state,
        span:     &count_span,
        arg_size: uint::size,
        count:    1,
    )!;
    draw.index = lower_index_span(
        state:      state,
        span:       &index_span,
        index_type: index_type,
        required:   0,
    )!;
    return draw;
}

fn PreparedGeneratedIndexedDraw? validate_and_lower_generated_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = encoder_command(commands);
    if (record == null || record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    note_command_semantic_check(state);
    (void)index_element_size(index_type)!;
    PreparedGeneratedIndexedDraw draw = {
        .pipeline = *validate_generated_graphics_execution_state(
            state,
            record,
            "cmd_draw_indexed_generated",
        )!,
        .records_span = records,
        .count_span   = count_span,
        .index_span   = index_span,
        .layout       = state.generated_indexed_draw_layout,
        .max_count    = max_draw_count,
        .index_type   = index_type,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    if (draw.layout == null) return gpu::BACKEND_ERROR~;
    draw.records = validate_and_lower_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDrawIndexedRecord::size,
        max_count:   max_draw_count,
        operation:   "cmd_draw_indexed_generated",
    )!;
    draw.count = validate_and_lower_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       uint::size,
        count:          1,
        operation:      "cmd_draw_indexed_generated",
        rejected_field: "count_span",
    )!;
    draw.index = validate_and_lower_index_span(
        state:      state,
        record:     record,
        span:       &index_span,
        index_type: index_type,
        required:   0,
        operation:  "cmd_draw_indexed_generated",
    )!;
    return draw;
}

fn void? track_generated_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    VkDeviceState* state = encoder_device_state(commands);
    CommandRecord* record = encoder_command(commands);
    CommandResourceReference[3] references = {
        {
            .owner      = draw.records_span.owner,
            .index      = draw.records_span.index,
            .generation = draw.records_span.generation,
        },
        {
            .owner      = draw.count_span.owner,
            .index      = draw.count_span.index,
            .generation = draw.count_span.generation,
        },
        {
            .owner      = draw.index_span.owner,
            .index      = draw.index_span.index,
            .generation = draw.index_span.generation,
        },
    };
    preflight_command_references(state, record, references[..])!;
    track_span_command_reference(state, record, &draw.records_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void? record_generated_indexed_draw(
    gpu::internal::CommandEncoderState* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    if (draw.max_count == 0) return;
    CommandRecord* record = encoder_command(commands);
    execute_generated_work(
        state:         encoder_device_state(commands),
        record:        record,
        pipeline:      &draw.pipeline,
        kind:          gpu::GeneratedWorkKind.DRAW_INDEXED,
        layout:        draw.layout,
        shader_stages: generated_graphics_shader_stages(),
        records:       &draw.records,
        count:         &draw.count,
        max_count:     draw.max_count,
        index_span:    &draw.index,
        index_type:    draw.index_type,
    )!;
}

fn void? trusted_draw_indexed_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw = lower_generated_indexed_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    )!;
    record_generated_indexed_draw(commands, &draw)!;
}

fn void? trusted_tracking_draw_indexed_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw = lower_generated_indexed_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_indexed_draw(commands, &draw)!;
    record_generated_indexed_draw(commands, &draw)!;
}

fn void? checked_draw_indexed_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw =
        validate_and_lower_generated_indexed_draw(
            commands:       commands,
            records:        records,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    record_generated_indexed_draw(commands, &draw)!;
}

fn void? checked_tracking_draw_indexed_generated(
    gpu::internal::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw =
        validate_and_lower_generated_indexed_draw(
            commands:       commands,
            records:        records,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = encoder_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        encoder_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_indexed_draw(commands, &draw)!;
    record_generated_indexed_draw(commands, &draw)!;
}
