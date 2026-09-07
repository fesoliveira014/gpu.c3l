module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::math;
import std::atomic;

struct PreparedRenderPass {
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] color_slots;
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] resolve_slots;
    gpu::AttachmentViewHandle[gpu::MAX_COLOR_ATTACHMENTS] color_handles;
    gpu::AttachmentViewHandle[gpu::MAX_COLOR_ATTACHMENTS] resolve_handles;
    vk::RenderingAttachmentInfo[gpu::MAX_COLOR_ATTACHMENTS] attachments;
    AttachmentViewSlot*       depth_view;
    gpu::AttachmentViewHandle depth_handle;
    vk::RenderingAttachmentInfo depth_attachment;
    vk::RenderingAttachmentInfo stencil_attachment;
    RenderCompatibility       compatibility;
    uint                      width;
    uint                      height;
    uint                      color_count;
    bool                      has_depth;
    bool                      has_depth_aspect;
    bool                      has_stencil_aspect;
}

struct LoweredColorState {
    uint count;
    uint[gpu::MAX_COLOR_ATTACHMENTS] blend_enables;
    vk::ColorBlendEquationEXT[gpu::MAX_COLOR_ATTACHMENTS] blend_equations;
    vk::ColorComponentFlags[gpu::MAX_COLOR_ATTACHMENTS] write_masks;
}

struct LoweredGraphicsState {
    vk::Viewport            viewport;
    vk::Rect2D              scissor;
    gpu::DynamicRasterState raster;
    gpu::DepthState         depth;
    gpu::StencilState       stencil;
    LoweredColorState       color;
}

struct PreparedIndexedDraw {
    BoundPipeline*  pipeline;
    ResolvedSpan    index;
    gpu::GpuSpan    index_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            index_count;
    uint            instance_count;
    gpu::IndexType  index_type;
}

struct PreparedIndirectDraw {
    BoundPipeline*  pipeline;
    ResolvedSpan    args;
    gpu::GpuSpan    args_span;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
}

struct PreparedIndexedIndirectDraw {
    BoundPipeline*  pipeline;
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
    BoundPipeline*  pipeline;
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
    BoundPipeline*                 pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    gpu::GpuSpan                   records_span;
    gpu::GpuSpan                   count_span;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
}

struct PreparedGeneratedIndexedDraw {
    BoundPipeline*                 pipeline;
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

struct DynamicStateRejection {
    ZString field;
    ZString invariant;
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
    CommandRecord* commands,
    gpu::DynamicRasterState* raster,
) @private {
    CommandRecord* record = commands;
    if (record.logical_pipeline.kind != PipelineKind.MESH) {
        vk::cmd_set_primitive_topology(
            record.command_buffer,
            topology_to_vk(raster.topology),
        );
    }
    vk::cmd_set_cull_mode(
        record.command_buffer,
        (vk::CullModeFlags)cull_mode_to_vk(raster.cull_mode),
    );
    vk::cmd_set_front_face(
        record.command_buffer,
        front_face_to_vk(raster.front_face),
    );
    vk::cmd_set_depth_bias_enable(
        record.command_buffer,
        raster.depth_bias_enable ? vk::TRUE : vk::FALSE,
    );
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
}

fn bool stencil_faces_equal(gpu::StencilFaceState* a, gpu::StencilFaceState* b) @private {
    return a.compare == b.compare
        && a.fail == b.fail
        && a.pass == b.pass
        && a.depth_fail == b.depth_fail
        && a.compare_mask == b.compare_mask
        && a.write_mask == b.write_mask
        && a.reference == b.reference;
}

fn void emit_stencil_face(
    vk::CommandBuffer command_buffer,
    vk::StencilFaceFlags faces,
    gpu::StencilFaceState* face,
) @private {
    vk::cmd_set_stencil_op(
        command_buffer: command_buffer,
        face_mask:      faces,
        fail_op:        stencil_op_to_vk(face.fail),
        pass_op:        stencil_op_to_vk(face.pass),
        depth_fail_op:  stencil_op_to_vk(face.depth_fail),
        compare_op:     compare_op_to_vk(face.compare),
    );
    vk::cmd_set_stencil_compare_mask(command_buffer, faces, face.compare_mask);
    vk::cmd_set_stencil_write_mask(command_buffer, faces, face.write_mask);
    vk::cmd_set_stencil_reference(command_buffer, faces, face.reference);
}

fn void emit_stencil_state(
    CommandRecord* commands,
    gpu::StencilState* stencil,
) @private {
    CommandRecord* record = commands;
    vk::cmd_set_stencil_test_enable(
        record.command_buffer,
        stencil.test_enable ? vk::TRUE : vk::FALSE,
    );
    if (!stencil.test_enable || stencil_faces_equal(&stencil.front, &stencil.back)) {
        emit_stencil_face(
            record.command_buffer,
            (vk::StencilFaceFlags)vk::StencilFaceFlagBits.FRONT_AND_BACK,
            &stencil.front,
        );
        return;
    }
    emit_stencil_face(
        record.command_buffer,
        (vk::StencilFaceFlags)vk::StencilFaceFlagBits.FRONT_BIT,
        &stencil.front,
    );
    emit_stencil_face(
        record.command_buffer,
        (vk::StencilFaceFlags)vk::StencilFaceFlagBits.BACK_BIT,
        &stencil.back,
    );
}

fn void emit_depth_state(
    CommandRecord* commands,
    gpu::DepthState* depth,
) @private {
    CommandRecord* record = commands;
    vk::cmd_set_depth_test_enable(
        record.command_buffer,
        depth.test_enable ? vk::TRUE : vk::FALSE,
    );
    vk::cmd_set_depth_write_enable(
        record.command_buffer,
        depth.write_enable ? vk::TRUE : vk::FALSE,
    );
    vk::cmd_set_depth_compare_op(
        record.command_buffer,
        compare_op_to_vk(depth.compare),
    );
}

fn PreparedRenderPass? lower_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    if (desc == null
        || desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS
        || (desc.colors.len > 0 && desc.colors.ptr == null)
        || (desc.width == 0) != (desc.height == 0)) {
        return gpu::INVALID_ARGUMENT~;
    }
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
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
        attachment.image_layout = texture_layout_to_vk(
            gpu::TextureLayout.COLOR_ATTACHMENT,
            view.format,
            state.unified_layouts,
        );
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
            attachment.resolve_image_layout = texture_layout_to_vk(
                gpu::TextureLayout.COLOR_ATTACHMENT,
                resolve.format,
                state.unified_layouts,
            );
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
        vk::ImageLayout layout = texture_layout_to_vk(
            gpu::TextureLayout.DEPTH_ATTACHMENT,
            depth.format,
            state.unified_layouts,
        );
        vk::ClearDepthStencilValue clear = {
            .depth   = desc.depth.clear.depth,
            .stencil = desc.depth.clear.stencil,
        };
        pass.has_depth_aspect = gpu::internal::format_has_depth(depth.format);
        pass.has_stencil_aspect = gpu::internal::format_has_stencil(depth.format);
        if (pass.has_depth_aspect) {
            pass.depth_attachment = vk::rendering_attachment_info();
            pass.depth_attachment.image_view = depth.native;
            pass.depth_attachment.image_layout = layout;
            pass.depth_attachment.load_op = load_op_to_vk(desc.depth.load_op);
            pass.depth_attachment.store_op = store_op_to_vk(desc.depth.store_op);
            pass.depth_attachment.clear_value.depth_stencil = clear;
        }
        if (pass.has_stencil_aspect) {
            pass.stencil_attachment = vk::rendering_attachment_info();
            pass.stencil_attachment.image_view = depth.native;
            pass.stencil_attachment.image_layout = layout;
            pass.stencil_attachment.load_op = load_op_to_vk(desc.depth.stencil_load_op);
            pass.stencil_attachment.store_op = store_op_to_vk(desc.depth.stencil_store_op);
            pass.stencil_attachment.clear_value.depth_stencil = clear;
        }
    }
    if (desc.width == 0 && desc.height == 0) {
        AttachmentViewSlot* first = pass.color_count > 0
            ? pass.color_slots[0]
            : pass.depth_view;
        if (first == null) return gpu::INVALID_ARGUMENT~;
        pass.width = first.width;
        pass.height = first.height;
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

<* FULL only: within one command list, a DONT_CARE stencil store followed
   by a LOAD of the same texture mip and layer through any attachment view.
   Runs after the pass is recorded, so a rejected begin leaves no history.
   The set holds up to the list's retained-reference capacity. *>
fn void note_stencil_load_after_discard(
    VkDeviceState* state,
    CommandRecord* record,
    AttachmentViewSlot* view,
    ZString operation,
    gpu::DepthTargetDesc* depth,
) @private {
    CommandScratch* scratch = record.scratch;
    StencilDiscardKey key = {
        .texture     = view.texture,
        .mip_level   = view.mip_level,
        .array_layer = view.array_layer,
    };
    uint found = scratch.stencil_discard_count;
    for (uint i = 0; i < scratch.stencil_discard_count; i++) {
        if (stencil_discard_key_equals(scratch.stencil_discards[i], key)) found = i;
    }
    bool discarded = found < scratch.stencil_discard_count;
    if (depth.stencil_load_op == gpu::LoadOp.LOAD && discarded) {
        report_state_diagnostic(
            state:          state,
            category:       { .public_contract },
            operation:      operation,
            rejected_field: "DepthTargetDesc.stencil_load_op",
            invariant:      "a LOAD after a DONT_CARE stencil store reads undefined contents",
        );
    }
    bool discards = depth.stencil_store_op == gpu::StoreOp.DONT_CARE;
    if (discards && !discarded
        && scratch.stencil_discard_count < scratch.stencil_discards.len) {
        scratch.stencil_discards[scratch.stencil_discard_count++] = key;
    } else if (!discards && discarded) {
        scratch.stencil_discards[found] =
            scratch.stencil_discards[--scratch.stencil_discard_count];
    }
}

fn bool stencil_discard_key_equals(StencilDiscardKey a, StencilDiscardKey b) @private {
    return a.texture.owner == b.texture.owner
        && a.texture.index == b.texture.index
        && a.texture.generation == b.texture.generation
        && a.mip_level == b.mip_level
        && a.array_layer == b.array_layer;
}

fn PreparedRenderPass? full_prepare_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    ZString operation,
) {
    if (commands == null || desc == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
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
            operation:   operation,
        )!;
        if (!texture.usage.color_attach) {
            return report_contract_failure(
                state:               state,
                operation:           operation,
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
        if (pass.width > view.width || pass.height > view.height) {
            ZString rejected_field = pass.width > view.width
                ? "RenderPassDesc.width"
                : "RenderPassDesc.height";
            return report_contract_failure(
                state:               state,
                operation:           operation,
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
                operation:   operation,
            )!;
            if (view.sample_count == gpu::SampleCount.ONE
                || !resolve_texture.usage.color_attach
                || resolve.sample_count != gpu::SampleCount.ONE
                || resolve.format != view.format
                || pass.width > resolve.width
                || pass.height > resolve.height) {
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
            operation:   operation,
        )!;
        if (!depth_texture.usage.depth_attach
            || pass.width > depth.width
            || pass.height > depth.height) {
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
    CommandRecord* commands,
    PreparedRenderPass* pass,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
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
    CommandRecord* commands,
    PreparedRenderPass* pass,
) {
    CommandRecord* record = commands;
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
    if (pass.has_depth_aspect) {
        rendering.p_depth_attachment = &pass.depth_attachment;
    }
    if (pass.has_stencil_aspect) {
        rendering.p_stencil_attachment = &pass.stencil_attachment;
    }
    vk::cmd_begin_rendering(cb, &rendering);
    record.active_render = pass.compatibility;
    record.state = gpu::internal::CommandState.RECORDING_RENDER_PASS;
}

fn void? trusted_begin_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    record_render_pass(commands, &pass);
}

fn void? full_begin_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = full_prepare_render_pass(
        commands,
        desc,
        "cmd_begin_render_pass",
    )!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass(commands, &pass);
    if (pass.has_stencil_aspect) {
        note_stencil_load_after_discard(
            state:     commands.device_state,
            record:    record,
            view:      pass.depth_view,
            operation: "cmd_begin_render_pass",
            depth:     desc.depth,
        );
    }
}

fn fault report_dynamic_state_rejection(
    VkDeviceState* state,
    ZString operation,
    DynamicStateRejection rejection,
) @private {
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejection.field,
        invariant:           rejection.invariant,
        backend_text:        null,
    );
}

fn void? reject_dynamic_state(
    VkDeviceState* state,
    ZString operation,
    ZString field,
    ZString invariant,
) @private {
    DynamicStateRejection rejection = {
        .field     = field,
        .invariant = invariant,
    };
    return report_dynamic_state_rejection(state, operation, rejection)~;
}

fn void? validate_viewport(
    VkDeviceState* state,
    ZString operation,
    gpu::Viewport* viewport,
) @private {
    if (!math::is_finite(viewport.x)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.x",
            invariant: "coordinate must be finite",
        );
    }
    if (!math::is_finite(viewport.y)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.y",
            invariant: "coordinate must be finite",
        );
    }
    if (!math::is_finite(viewport.width)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.width",
            invariant: "dimension must be finite",
        );
    }
    if (!math::is_finite(viewport.height)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.height",
            invariant: "dimension must be finite",
        );
    }
    if (!math::is_finite(viewport.min_depth)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.min_depth",
            invariant: "depth endpoint must be finite and in [0, 1]",
        );
    }
    if (!math::is_finite(viewport.max_depth)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.max_depth",
            invariant: "depth endpoint must be finite and in [0, 1]",
        );
    }
    if (viewport.width <= 0.0f) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.width",
            invariant: "width must be positive",
        );
    }
    if (viewport.height == 0.0f) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.height",
            invariant: "height must be nonzero by library policy",
        );
    }
    if (viewport.width > (float)state.viewport_limits.max_width) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.width",
            invariant: "width must not exceed the selected device maximum",
        );
    }
    if (viewport.height > (float)state.viewport_limits.max_height
        || viewport.height < -(float)state.viewport_limits.max_height) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.height",
            invariant: "absolute height must not exceed the selected device maximum",
        );
    }

    float right = viewport.x + viewport.width;
    float end_y = viewport.y + viewport.height;
    if (!math::is_finite(right)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.x + Viewport.width",
            invariant: "computed endpoint must be finite",
        );
    }
    if (!math::is_finite(end_y)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.y + Viewport.height",
            invariant: "computed endpoint must be finite",
        );
    }
    if (viewport.x < state.viewport_limits.min_bound
        || viewport.x > state.viewport_limits.max_bound) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.x",
            invariant: "coordinate must fit the selected device viewport bounds",
        );
    }
    if (right < state.viewport_limits.min_bound
        || right > state.viewport_limits.max_bound) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.x + Viewport.width",
            invariant: "endpoint must fit the selected device viewport bounds",
        );
    }
    if (viewport.y < state.viewport_limits.min_bound
        || viewport.y > state.viewport_limits.max_bound) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.y",
            invariant: "coordinate must fit the selected device viewport bounds",
        );
    }
    if (end_y < state.viewport_limits.min_bound
        || end_y > state.viewport_limits.max_bound) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.y + Viewport.height",
            invariant: "endpoint must fit the selected device viewport bounds",
        );
    }
    if (viewport.min_depth < 0.0f || viewport.min_depth > 1.0f) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.min_depth",
            invariant: "depth endpoint must be in [0, 1]",
        );
    }
    if (viewport.max_depth < 0.0f || viewport.max_depth > 1.0f) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "Viewport.max_depth",
            invariant: "depth endpoint must be in [0, 1]",
        );
    }
}

fn vk::Viewport lower_viewport(gpu::Viewport* viewport) {
    return {
        .x         = viewport.x,
        .y         = viewport.y,
        .width     = viewport.width,
        .height    = viewport.height,
        .min_depth = viewport.min_depth,
        .max_depth = viewport.max_depth,
    };
}

fn void emit_viewport(
    CommandRecord* commands,
    vk::Viewport* viewport,
) @private {
    CommandRecord* record = commands;
    vk::cmd_set_viewport(record.command_buffer, 0, 1, viewport);
}

fn bool graphics_state_settable(gpu::internal::CommandState state) =>
    state == gpu::internal::CommandState.RECORDING
        || state == gpu::internal::CommandState.RECORDING_RENDER_PASS;

fn void? validate_graphics_state_queue(CommandRecord* record) @private {
    if (!record.queue.roles.graphics) return gpu::INVALID_ARGUMENT~;
}

fn void? trusted_set_viewport(
    CommandRecord* commands,
    gpu::Viewport* viewport,
) {
    if (viewport == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::Viewport native = lower_viewport(viewport);
    emit_viewport(commands, &native);
}

fn void? full_set_viewport(
    CommandRecord* commands,
    gpu::Viewport* viewport,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    validate_viewport(state, "cmd_set_viewport", viewport)!;
    vk::Viewport native = lower_viewport(viewport);
    emit_viewport(commands, &native);
}

fn void? validate_scissor(
    VkDeviceState* state,
    ZString operation,
    gpu::ScissorRect* scissor,
) @private {
    if (scissor.x < 0 || scissor.y < 0
        || scissor.width < 0 || scissor.height < 0) {
        ZString field = scissor.x < 0
            ? "ScissorRect.x"
            : scissor.y < 0
                ? "ScissorRect.y"
                : scissor.width < 0
                    ? "ScissorRect.width"
                    : "ScissorRect.height";
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     field,
            invariant: "scissor fields must be nonnegative",
        );
    }
    long right = (long)scissor.x + (long)scissor.width;
    long bottom = (long)scissor.y + (long)scissor.height;
    if (right > int::max) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "ScissorRect.x + ScissorRect.width",
            invariant: "endpoint must fit signed native coordinate range",
        );
    }
    if (bottom > int::max) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "ScissorRect.y + ScissorRect.height",
            invariant: "endpoint must fit signed native coordinate range",
        );
    }
}

fn vk::Rect2D lower_scissor(gpu::ScissorRect* scissor) {
    return {
        .offset = { .x = scissor.x, .y = scissor.y },
        .extent = {
            .width  = (uint)scissor.width,
            .height = (uint)scissor.height,
        },
    };
}

fn void? validate_raster_state(
    VkDeviceState* state,
    ZString operation,
    gpu::DynamicRasterState* raster,
) @private {
    if (raster == null) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster",
            invariant: "raster state must not be null",
        );
    }
    if (!primitive_topology_valid(raster.topology)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.topology",
            invariant: "topology must be defined",
        );
    }
    if (!cull_mode_valid(raster.cull_mode)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.cull_mode",
            invariant: "cull mode must be defined",
        );
    }
    if (!front_face_valid(raster.front_face)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.front_face",
            invariant: "front face must be defined",
        );
    }
    if (raster.depth_bias_enable
        && !math::is_finite(raster.depth_bias_constant)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.depth_bias_constant",
            invariant: "enabled depth bias value must be finite",
        );
    }
    if (raster.depth_bias_enable
        && !math::is_finite(raster.depth_bias_slope)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.depth_bias_slope",
            invariant: "enabled depth bias value must be finite",
        );
    }
    if (raster.depth_bias_enable
        && !math::is_finite(raster.depth_bias_clamp)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.raster.depth_bias_clamp",
            invariant: "enabled depth bias value must be finite",
        );
    }
}

fn void? validate_depth_state(
    VkDeviceState* state,
    ZString operation,
    gpu::DepthState* depth,
) @private {
    if (depth == null) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.depth",
            invariant: "depth state must not be null",
        );
    }
    if (!compare_op_valid(depth.compare)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     "GraphicsState.depth.compare",
            invariant: "compare operation must be defined",
        );
    }
}

fn bool stencil_face_valid(gpu::StencilFaceState* face) @private {
    return compare_op_valid(face.compare)
        && stencil_op_valid(face.fail)
        && stencil_op_valid(face.pass)
        && stencil_op_valid(face.depth_fail);
}

fn bool stencil_state_valid(gpu::StencilState* stencil) @private {
    return !stencil.test_enable
        || (stencil_face_valid(&stencil.front) && stencil_face_valid(&stencil.back));
}

fn void? validate_stencil_face(
    VkDeviceState* state,
    ZString operation,
    gpu::StencilFaceState* face,
    ZString field,
) @private {
    if (!stencil_face_valid(face)) {
        return reject_dynamic_state(
            state:     state,
            operation: operation,
            field:     field,
            invariant: "stencil compare and operations must be defined",
        );
    }
}

fn void? validate_stencil_state(
    VkDeviceState* state,
    ZString operation,
    gpu::StencilState* stencil,
) @private {
    if (!stencil.test_enable) return;
    validate_stencil_face(
        state:     state,
        operation: operation,
        face:      &stencil.front,
        field:     "GraphicsState.stencil.front",
    )!;
    validate_stencil_face(
        state:     state,
        operation: operation,
        face:      &stencil.back,
        field:     "GraphicsState.stencil.back",
    )!;
}

<* FULL only: an enabled stencil test on a pipeline without a stencil aspect
   is legal and does nothing. *>
fn void note_stencil_without_aspect(
    VkDeviceState* state,
    CommandRecord* record,
    ZString operation,
    gpu::StencilState* stencil,
) @private {
    if (!stencil.test_enable) return;
    BoundPipeline*? bound = borrow_bound_pipeline(record);
    if (catch err = bound) return;
    BoundPipeline* pipeline = bound;
    if (gpu::internal::format_has_stencil(pipeline.render.depth_format)) return;
    report_state_diagnostic(
        state:          state,
        category:       { .performance },
        operation:      operation,
        rejected_field: "GraphicsState.stencil.test_enable",
        invariant:      "stencil test has no effect without a stencil aspect in the bound pipeline",
    );
}

fn LoweredGraphicsState? lower_graphics_state(
    gpu::GraphicsState* state,
) {
    if (state == null || !stencil_state_valid(&state.stencil)) return gpu::INVALID_ARGUMENT~;
    return {
        .viewport = lower_viewport(&state.viewport),
        .scissor  = lower_scissor(&state.scissor),
        .raster   = state.raster,
        .depth    = state.depth,
        .stencil  = state.stencil,
        .color    = lower_color_state(&state.color)!,
    };
}

fn LoweredGraphicsState? full_prepare_graphics_state(
    VkDeviceState* device,
    CommandRecord* record,
    ZString operation,
    gpu::GraphicsState* state,
) @private {
    if (state == null) {
        DynamicStateRejection rejection = {
            .field     = "GraphicsState",
            .invariant = "graphics state must not be null",
        };
        return report_dynamic_state_rejection(
            device,
            operation,
            rejection,
        )~;
    }
    validate_viewport(device, operation, &state.viewport)!;
    validate_scissor(device, operation, &state.scissor)!;
    validate_raster_state(device, operation, &state.raster)!;
    validate_depth_state(device, operation, &state.depth)!;
    validate_stencil_state(device, operation, &state.stencil)!;
    note_stencil_without_aspect(
        state:     device,
        record:    record,
        operation: operation,
        stencil:   &state.stencil,
    );
    LoweredColorState color = full_prepare_color_state(
        state:     device,
        record:    record,
        color:     &state.color,
        operation: operation,
    )!;
    return {
        .viewport = lower_viewport(&state.viewport),
        .scissor  = lower_scissor(&state.scissor),
        .raster   = state.raster,
        .depth    = state.depth,
        .stencil  = state.stencil,
        .color    = color,
    };
}

fn void emit_scissor(
    CommandRecord* commands,
    vk::Rect2D* scissor,
) @private {
    CommandRecord* record = commands;
    vk::cmd_set_scissor(record.command_buffer, 0, 1, scissor);
}

fn void emit_graphics_state(
    CommandRecord* commands,
    LoweredGraphicsState* state,
) @private {
    emit_viewport(commands, &state.viewport);
    emit_scissor(commands, &state.scissor);
    emit_raster_state(commands, &state.raster);
    emit_depth_state(commands, &state.depth);
    emit_stencil_state(commands, &state.stencil);
    emit_color_state(commands, &state.color);
}

fn ColorCompatibility color_compatibility(
    RenderCompatibility* render,
) @private {
    ColorCompatibility compatibility = {
        .count = render.color_count,
    };
    for (uint i = 0; i < render.color_count; i++) {
        compatibility.formats[i] = render.color_formats[i];
    }
    return compatibility;
}

fn bool color_compatibility_matches(
    ColorCompatibility* color,
    RenderCompatibility* render,
) @private {
    if (color.count != render.color_count) return false;
    for (uint i = 0; i < color.count; i++) {
        if (color.formats[i] != render.color_formats[i]) return false;
    }
    return true;
}

fn LoweredColorState? lower_color_state(
    gpu::ColorState* state,
) {
    if (state == null
        || state.targets.len > gpu::MAX_COLOR_ATTACHMENTS
        || (state.targets.len > 0 && state.targets.ptr == null)) {
        return gpu::INVALID_ARGUMENT~;
    }
    LoweredColorState lowered = {
        .count = (uint)state.targets.len,
    };
    foreach (i, &target : state.targets) {
        lowered.blend_enables[i] =
            target.blend.enable ? vk::TRUE : vk::FALSE;
        vk::ColorBlendEquationEXT* equation =
            &lowered.blend_equations[i];
        if (target.blend.enable) {
            *equation = {
                .src_color_blend_factor =
                    blend_factor_to_vk(target.blend.src_color),
                .dst_color_blend_factor =
                    blend_factor_to_vk(target.blend.dst_color),
                .color_blend_op =
                    blend_op_to_vk(target.blend.color_op),
                .src_alpha_blend_factor =
                    blend_factor_to_vk(target.blend.src_alpha),
                .dst_alpha_blend_factor =
                    blend_factor_to_vk(target.blend.dst_alpha),
                .alpha_blend_op =
                    blend_op_to_vk(target.blend.alpha_op),
            };
        } else {
            *equation = {
                .src_color_blend_factor = vk::BlendFactor.ONE,
                .dst_color_blend_factor = vk::BlendFactor.ZERO,
                .color_blend_op         = vk::BlendOp.ADD,
                .src_alpha_blend_factor = vk::BlendFactor.ONE,
                .dst_alpha_blend_factor = vk::BlendFactor.ZERO,
                .alpha_blend_op         = vk::BlendOp.ADD,
            };
        }
        lowered.write_masks[i] =
            color_write_mask_to_vk(target.write_mask);
    }
    return lowered;
}

fn void emit_color_state(
    CommandRecord* commands,
    LoweredColorState* state,
) @private {
    if (state.count == 0) return;
    CommandRecord* record = commands;
    VkDynamicColorDispatch* dispatch =
        &commands.device_state.device_dispatch.dynamic_color;
    dispatch.set_blend_enable(
        record.command_buffer,
        0,
        state.count,
        &state.blend_enables[0],
    );
    dispatch.set_blend_equation(
        record.command_buffer,
        0,
        state.count,
        &state.blend_equations[0],
    );
    dispatch.set_write_mask(
        record.command_buffer,
        0,
        state.count,
        &state.write_masks[0],
    );
}

fn fault color_state_rejection(
    VkDeviceState* state,
    ZString operation,
    ZString field,
    ZString invariant,
) @private {
    DynamicStateRejection rejection = {
        .field     = field,
        .invariant = invariant,
    };
    return report_dynamic_state_rejection(
        state,
        operation,
        rejection,
    );
}

fn LoweredColorState? full_prepare_color_state(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::ColorState* color,
    ZString operation,
) @private {
    if (!record.logical_pipeline.handle.is_valid()) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PIPELINE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "pipeline",
            invariant:           "the required pipeline bind point must be active",
            backend_text:        null,
        )~;
    }
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     operation,
        any_raster:    true,
    )!;
    if (color.targets.len != pipeline.render.color_count) {
        return color_state_rejection(
            state:     state,
            operation: operation,
            field:     "GraphicsState.color.targets",
            invariant: "target count must match the selected graphics pipeline",
        )~;
    }
    if (record.state
            == gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        if (catch compatibility_fault =
            validate_pipeline_render_compatibility(
                pipeline,
                &record.active_render,
            )) {
            return color_state_rejection(
                state:     state,
                operation: operation,
                field:     "pipeline",
                invariant: "selected pipeline formats must match the active render pass",
            )~;
        }
    }
    foreach (i, &target : color.targets) {
        if (!blend_state_valid(&target.blend)) {
            return color_state_rejection(
                state:     state,
                operation: operation,
                field:     "GraphicsState.color.targets.blend",
                invariant: "enabled blend factors and operations must be defined",
            )~;
        }
        if (!color_write_mask_valid(target.write_mask)) {
            return color_state_rejection(
                state:     state,
                operation: operation,
                field:     "GraphicsState.color.targets.write_mask",
                invariant: "write mask may contain only RGBA channel bits",
            )~;
        }
        if (target.blend.enable
            && color_format_is_integer(
                pipeline.render.color_formats[i],
            )) {
            return color_state_rejection(
                state:     state,
                operation: operation,
                field:     "GraphicsState.color.targets.blend.enable",
                invariant: "integer color targets do not support blending",
            )~;
        }
    }
    return lower_color_state(color);
}

fn void? trusted_set_graphics_state(
    CommandRecord* commands,
    gpu::GraphicsState* state,
) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    LoweredGraphicsState graphics = lower_graphics_state(state)!;
    emit_graphics_state(commands, &graphics);
}

fn void? full_set_graphics_state(
    CommandRecord* commands,
    gpu::GraphicsState* state,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* device = commands.device_state;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    LoweredGraphicsState graphics = full_prepare_graphics_state(
        device:    device,
        record:    record,
        operation: "cmd_set_graphics_state",
        state:     state,
    )!;
    emit_graphics_state(commands, &graphics);
    record.graphics_state_initialized = true;
    record.color_state =
        color_compatibility(&record.logical_pipeline.render);
    record.color_state_initialized = true;
}

fn void? trusted_set_scissor(
    CommandRecord* commands,
    gpu::ScissorRect* scissor,
) {
    if (scissor == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::Rect2D native = lower_scissor(scissor);
    emit_scissor(commands, &native);
}

fn void? full_set_scissor(
    CommandRecord* commands,
    gpu::ScissorRect* scissor,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    if (!graphics_state_settable(record.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    validate_scissor(state, "cmd_set_scissor", scissor)!;
    vk::Rect2D native = lower_scissor(scissor);
    emit_scissor(commands, &native);
}

fn void? vk_cmd_end_render_pass(CommandRecord* commands) @private {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    record.active_render = {};
    record.state = gpu::internal::CommandState.RECORDING;
}

fn void? validate_color_execution_state(
    CommandRecord* record,
    BoundPipeline* pipeline,
) @private {
    if (pipeline.render.color_count == 0) return;
    if (!record.color_state_initialized
        || !color_compatibility_matches(
            &record.color_state,
            &pipeline.render,
        )) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
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
    if (!record.graphics_state_initialized) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_color_execution_state(record, pipeline)!;
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
    vk::cmd_push_constants(
        command_buffer: record.command_buffer,
        layout:         pipeline.layout,
        stage_flags:    pipeline.kind == PipelineKind.MESH
            ? record.device_state.mesh_push_stages
            : generated_graphics_shader_stages(),
        offset:         0,
        size:           GRAPHICS_ROOT_PUSH_SIZE,
        p_values:       &push,
    );
}

fn void record_draw(
    CommandRecord* commands,
    BoundPipeline* pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    CommandRecord* record = commands;
    push_graphics_roots(record, pipeline, vertex_root, fragment_root);
    vk::cmd_draw(
        command_buffer: record.command_buffer,
        vertex_count:   vertex_count,
        instance_count: instance_count,
        first_vertex:   0,
        first_instance: 0,
    );
}

fn void? trusted_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    BoundPipeline* pipeline = borrow_bound_pipeline(record)!;
    record_draw(
        commands:      commands,
        pipeline:      pipeline,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        vertex_count:  vertex_count,
        instance_count: instance_count,
    );
}

fn void? full_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    BoundPipeline* pipeline = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw",
    )!;
    record_draw(
        commands:       commands,
        pipeline:       pipeline,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   vertex_count,
        instance_count: instance_count,
    );
}

fn PreparedIndexedDraw? lower_indexed_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    usz required = gpu::internal::indirect_argument_bytes(
        argument_size: trusted_index_element_size(index_type),
        count:         index_count,
        max_count:     uint::max,
    )!;
    return {
        .pipeline      = borrow_bound_pipeline(record)!,
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

fn PreparedIndexedDraw? full_prepare_indexed_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    usz index_size = index_element_size(index_type)!;
    return {
        .pipeline = validate_graphics_execution_state(
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
    CommandRecord* commands,
    PreparedIndexedDraw* draw,
) {
    return track_span_command_reference(
        commands.device_state,
        commands,
        &draw.index_span,
    );
}

fn void record_indexed_draw(
    CommandRecord* commands,
    PreparedIndexedDraw* draw,
) {
    CommandRecord* record = commands;
    push_graphics_roots(
        record,
        draw.pipeline,
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
}

fn void? trusted_draw_indexed(
    CommandRecord* commands,
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

fn void? full_draw_indexed(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = full_prepare_indexed_draw(
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
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .pipeline = borrow_bound_pipeline(record)!,
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

fn PreparedIndirectDraw? full_prepare_indirect_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .pipeline = validate_graphics_execution_state(
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
    CommandRecord* commands,
    PreparedIndirectDraw* draw,
) {
    return track_span_command_reference(
        commands.device_state,
        commands,
        &draw.args_span,
    );
}

fn void record_indirect_draw(
    CommandRecord* commands,
    PreparedIndirectDraw* draw,
) {
    CommandRecord* record = commands;
    push_graphics_roots(
        record,
        draw.pipeline,
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
}

fn void? trusted_draw_indirect(
    CommandRecord* commands,
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

fn void? full_draw_indirect(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = full_prepare_indirect_draw(
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
}

fn PreparedIndexedIndirectDraw? lower_indexed_indirect_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .pipeline = borrow_bound_pipeline(record)!,
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

fn PreparedIndexedIndirectDraw? full_prepare_indexed_indirect_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .pipeline = validate_graphics_execution_state(
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
    CommandRecord* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
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
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &draw.args_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void record_indexed_indirect_draw(
    CommandRecord* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    CommandRecord* record = commands;
    push_graphics_roots(
        record,
        draw.pipeline,
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
}

fn void? trusted_draw_indexed_indirect(
    CommandRecord* commands,
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

fn void? full_draw_indexed_indirect(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw =
        full_prepare_indexed_indirect_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            draw_count:     draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_indexed_indirect_draw(commands, &draw)!;
    record_indexed_indirect_draw(commands, &draw);
}

<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn PreparedIndexedIndirectCountDraw? lower_indexed_indirect_count_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    return {
        .pipeline = borrow_bound_pipeline(record)!,
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

fn PreparedIndexedIndirectCountDraw? full_prepare_indexed_indirect_count_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    if (!state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    return {
        .pipeline = validate_graphics_execution_state(
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
    CommandRecord* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
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
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &draw.args_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void record_indexed_indirect_count_draw(
    CommandRecord* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    CommandRecord* record = commands;
    push_graphics_roots(
        record,
        draw.pipeline,
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
}

fn void? trusted_draw_indexed_indirect_count(
    CommandRecord* commands,
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

fn void? full_draw_indexed_indirect_count(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        full_prepare_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
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
    if (!record.graphics_state_initialized) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_color_execution_state(record, pipeline)!;
    return pipeline;
}

fn void? validate_color_execution_state_for_test(
    VkDeviceState* state,
    CommandRecord* record,
    bool generated,
) @feat(TESTING) {
    if (generated) {
        (void)validate_generated_graphics_execution_state(
            state,
            record,
            "cmd_draw_generated",
        )!;
        return;
    }
    (void)validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   {},
        fragment_root: {},
        operation:     "cmd_draw",
    )!;
}

fn vk::ShaderStageFlags generated_graphics_shader_stages() =>
    (vk::ShaderStageFlags)((uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT);

fn PreparedGeneratedDraw? lower_generated_draw(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    PreparedGeneratedDraw draw = {
        .pipeline     = borrow_bound_pipeline(record)!,
        .records_span = records,
        .count_span   = count_span,
        .layout       = state.generated_draw_layout,
        .max_count    = max_draw_count,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    draw.layout = required_generated_work_layout(draw.layout)!;
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

fn PreparedGeneratedDraw? full_prepare_generated_draw(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    PreparedGeneratedDraw draw = {
        .pipeline = validate_generated_graphics_execution_state(
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
    draw.layout = required_generated_work_layout(draw.layout)!;
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
    CommandRecord* commands,
    PreparedGeneratedDraw* draw,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
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
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &draw.records_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
}

fn void? record_generated_draw(
    CommandRecord* commands,
    PreparedGeneratedDraw* draw,
) {
    if (draw.max_count == 0) return;
    execute_generated_work(
        state:         commands.device_state,
        record:        commands,
        pipeline:      draw.pipeline,
        kind:          gpu::GeneratedWorkKind.DRAW,
        layout:        draw.layout,
        shader_stages: generated_graphics_shader_stages(),
        records:       &draw.records,
        count:         &draw.count,
        max_count:     draw.max_count,
    )!;
}

fn void? trusted_draw_generated(
    CommandRecord* commands,
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

fn void? full_draw_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = full_prepare_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_generated_draw(commands, &draw)!;
    record_generated_draw(commands, &draw)!;
}

fn PreparedGeneratedIndexedDraw? lower_generated_indexed_draw(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    PreparedGeneratedIndexedDraw draw = {
        .pipeline     = borrow_bound_pipeline(record)!,
        .records_span = records,
        .count_span   = count_span,
        .index_span   = index_span,
        .layout       = state.generated_indexed_draw_layout,
        .max_count    = max_draw_count,
        .index_type   = index_type,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_draw_count == 0) return draw;
    draw.layout = required_generated_work_layout(draw.layout)!;
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

fn PreparedGeneratedIndexedDraw? full_prepare_generated_indexed_draw(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    (void)index_element_size(index_type)!;
    PreparedGeneratedIndexedDraw draw = {
        .pipeline = validate_generated_graphics_execution_state(
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
    draw.layout = required_generated_work_layout(draw.layout)!;
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
    CommandRecord* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
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
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &draw.records_span)!;
    track_span_command_reference(state, record, &draw.count_span)!;
    track_span_command_reference(state, record, &draw.index_span)!;
}

fn void? record_generated_indexed_draw(
    CommandRecord* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    if (draw.max_count == 0) return;
    CommandRecord* record = commands;
    execute_generated_work(
        state:         commands.device_state,
        record:        record,
        pipeline:      draw.pipeline,
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
    CommandRecord* commands,
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

fn void? full_draw_indexed_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw =
        full_prepare_generated_indexed_draw(
            commands:       commands,
            records:        records,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    if (draw.max_count == 0) return;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_generated_indexed_draw(commands, &draw)!;
    record_generated_indexed_draw(commands, &draw)!;
}

struct PreparedMeshDraw {
    BoundPipeline*  pipeline;
    gpu::GpuAddress mesh_root;
    gpu::GpuAddress fragment_root;
    gpu::Vec3u      groups;
}

struct PreparedMeshIndirectDraw {
    BoundPipeline*  pipeline;
    ResolvedSpan    args;
    ResolvedSpan    count;
    gpu::GpuSpan    args_span;
    gpu::GpuSpan    count_span;
    gpu::GpuAddress mesh_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
    bool            counted;
}

fn BoundPipeline*? validate_mesh_execution_state(
    VkDeviceState* state,
    CommandRecord* record,
    ZString operation,
) @private {
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.MESH,
        operation:     operation,
    )!;
    if (!record.graphics_state_initialized) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_color_execution_state(record, pipeline)!;
    return pipeline;
}

fn void? validate_mesh_groups(
    VkDeviceState* state,
    BoundPipeline* pipeline,
    gpu::Vec3u groups,
    ZString operation,
) @private {
    gpu::Vec3u limit = pipeline.mesh_task_stage
        ? state.mesh_shader_caps.max_task_work_group_count
        : state.mesh_shader_caps.max_mesh_work_group_count;
    if (groups.x <= limit.x && groups.y <= limit.y && groups.z <= limit.z) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.PIPELINE,
        has_public_identity: true,
        index:               pipeline.handle.index,
        generation:          pipeline.handle.generation,
        debug_name:          null,
        rejected_field:      "groups",
        invariant:           "work-group counts must fit DeviceCaps.mesh_shaders",
        backend_text:        null,
    )~;
}

fn void record_mesh_draw(CommandRecord* record, PreparedMeshDraw* draw) @private {
    push_graphics_roots(record, draw.pipeline, draw.mesh_root, draw.fragment_root);
    record.device_state.device_dispatch.mesh_shaders.cmd_draw_mesh_tasks(
        record.command_buffer,
        draw.groups.x,
        draw.groups.y,
        draw.groups.z,
    );
}

fn void record_mesh_indirect_draw(
    CommandRecord* record,
    PreparedMeshIndirectDraw* draw,
) @private {
    push_graphics_roots(record, draw.pipeline, draw.mesh_root, draw.fragment_root);
    VkMeshShaderDispatch* dispatch = &record.device_state.device_dispatch.mesh_shaders;
    if (draw.counted) {
        dispatch.cmd_draw_mesh_tasks_indirect_count(
            record.command_buffer,
            draw.args.slot.buffer,
            (vk::DeviceSize)draw.args.offset,
            draw.count.slot.buffer,
            (vk::DeviceSize)draw.count.offset,
            draw.draw_count,
            gpu::DrawMeshTasksIndirectCommand::size,
        );
        return;
    }
    dispatch.cmd_draw_mesh_tasks_indirect(
        record.command_buffer,
        draw.args.slot.buffer,
        (vk::DeviceSize)draw.args.offset,
        draw.draw_count,
        gpu::DrawMeshTasksIndirectCommand::size,
    );
}

fn void? trusted_draw_mesh_tasks(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::Vec3u groups,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (!record.device_state.mesh_shaders_enabled) return gpu::UNSUPPORTED_FEATURE~;
    PreparedMeshDraw draw = {
        .pipeline      = borrow_bound_pipeline(record)!,
        .mesh_root     = mesh_root,
        .fragment_root = fragment_root,
        .groups        = groups,
    };
    record_mesh_draw(record, &draw);
}

fn void? full_draw_mesh_tasks(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::Vec3u groups,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    if (!state.mesh_shaders_enabled) return gpu::UNSUPPORTED_FEATURE~;
    PreparedMeshDraw draw = {
        .pipeline      = validate_mesh_execution_state(state, record, "cmd_draw_mesh_tasks")!,
        .mesh_root     = mesh_root,
        .fragment_root = fragment_root,
        .groups        = groups,
    };
    validate_mesh_groups(
        state:     state,
        pipeline:  draw.pipeline,
        groups:    groups,
        operation: "cmd_draw_mesh_tasks",
    )!;
    record_mesh_draw(record, &draw);
}

fn PreparedMeshIndirectDraw? lower_mesh_indirect_draw(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint draw_count,
    bool counted,
) @private {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    if (!state.mesh_shaders_enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (counted && !state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    PreparedMeshIndirectDraw draw = {
        .pipeline = borrow_bound_pipeline(record)!,
        .args = lower_indirect_span(
            state:    state,
            span:     &args,
            arg_size: gpu::DrawMeshTasksIndirectCommand::size,
            count:    draw_count,
        )!,
        .args_span     = args,
        .count_span    = count_span,
        .mesh_root     = mesh_root,
        .fragment_root = fragment_root,
        .draw_count    = draw_count,
        .counted       = counted,
    };
    if (counted) {
        draw.count = lower_indirect_span(
            state:    state,
            span:     &count_span,
            arg_size: uint::size,
            count:    1,
        )!;
    }
    return draw;
}

fn PreparedMeshIndirectDraw? full_prepare_mesh_indirect_draw(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint draw_count,
    bool counted,
    ZString operation,
) @private {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    if (!state.mesh_shaders_enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (counted && !state.draw_indirect_count) return gpu::UNSUPPORTED_FEATURE~;
    PreparedMeshIndirectDraw draw = {
        .pipeline = validate_mesh_execution_state(state, record, operation)!,
        .args = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &args,
            arg_size:       gpu::DrawMeshTasksIndirectCommand::size,
            count:          draw_count,
            operation:      operation,
            rejected_field: counted ? "args" : "draw_count",
            max_count:      counted ? uint::max : state.caps.max_draw_indirect_count,
        )!,
        .args_span     = args,
        .count_span    = count_span,
        .mesh_root     = mesh_root,
        .fragment_root = fragment_root,
        .draw_count    = draw_count,
        .counted       = counted,
    };
    if (counted) {
        draw.count = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &count_span,
            arg_size:       uint::size,
            count:          1,
            operation:      operation,
            rejected_field: "count_span",
        )!;
    }
    return draw;
}

fn void? track_mesh_indirect_draw(
    CommandRecord* commands,
    PreparedMeshIndirectDraw* draw,
) @private {
    track_span_command_reference(commands.device_state, commands, &draw.args_span)!;
    if (draw.counted) {
        track_span_command_reference(commands.device_state, commands, &draw.count_span)!;
    }
}

fn void? trusted_draw_mesh_tasks_indirect(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedMeshIndirectDraw draw = lower_mesh_indirect_draw(
        commands:      commands,
        mesh_root:     mesh_root,
        fragment_root: fragment_root,
        args:          args,
        count_span:    {},
        draw_count:    draw_count,
        counted:       false,
    )!;
    record_mesh_indirect_draw(commands, &draw);
}

fn void? full_draw_mesh_tasks_indirect(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedMeshIndirectDraw draw = full_prepare_mesh_indirect_draw(
        commands:      commands,
        mesh_root:     mesh_root,
        fragment_root: fragment_root,
        args:          args,
        count_span:    {},
        draw_count:    draw_count,
        counted:       false,
        operation:     "cmd_draw_mesh_tasks_indirect",
    )!;
    track_mesh_indirect_draw(commands, &draw)!;
    record_mesh_indirect_draw(commands, &draw);
}

fn void? trusted_draw_mesh_tasks_indirect_count(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedMeshIndirectDraw draw = lower_mesh_indirect_draw(
        commands:      commands,
        mesh_root:     mesh_root,
        fragment_root: fragment_root,
        args:          args,
        count_span:    count_span,
        draw_count:    max_draw_count,
        counted:       true,
    )!;
    record_mesh_indirect_draw(commands, &draw);
}

fn void? full_draw_mesh_tasks_indirect_count(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedMeshIndirectDraw draw = full_prepare_mesh_indirect_draw(
        commands:      commands,
        mesh_root:     mesh_root,
        fragment_root: fragment_root,
        args:          args,
        count_span:    count_span,
        draw_count:    max_draw_count,
        counted:       true,
        operation:     "cmd_draw_mesh_tasks_indirect_count",
    )!;
    track_mesh_indirect_draw(commands, &draw)!;
    record_mesh_indirect_draw(commands, &draw);
}
