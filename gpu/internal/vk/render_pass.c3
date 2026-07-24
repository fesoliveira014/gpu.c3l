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

struct LoweredGraphicsState {
    vk::Viewport            viewport;
    vk::Rect2D              scissor;
    gpu::DynamicRasterState raster;
    gpu::DepthState         depth;
}

struct LoweredColorState {
    uint count;
    uint[gpu::MAX_COLOR_ATTACHMENTS] blend_enables;
    vk::ColorBlendEquationEXT[gpu::MAX_COLOR_ATTACHMENTS] blend_equations;
    vk::ColorComponentFlags[gpu::MAX_COLOR_ATTACHMENTS] write_masks;
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

struct DynamicStateRejection {
    ZString field;
    ZString invariant;
}

struct DynamicStateObservation @if(env::TESTING) {
    ulong                   begin_rendering_emissions;
    ulong                   viewport_emissions;
    ulong                   scissor_emissions;
    ulong                   raster_state_emissions;
    ulong                   depth_state_emissions;
    vk::Viewport            viewport;
    vk::Rect2D              scissor;
    gpu::DynamicRasterState raster;
    gpu::DepthState         depth;
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
    gpu::internal::CommandRecord* commands,
    gpu::DynamicRasterState* raster,
) @private {
    VkCommandRecord* record = backend_command(commands);
    $if env::TESTING:
        VkDeviceState* state = command_device_state(commands);
        if (state.dynamic_state_observation_enabled) {
            state.raster_state_emissions++;
            state.last_raster_state = *raster;
        }
    $endif
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

fn void emit_depth_state(
    gpu::internal::CommandRecord* commands,
    gpu::DepthState* depth,
) @private {
    VkCommandRecord* record = backend_command(commands);
    $if env::TESTING:
        VkDeviceState* state = command_device_state(commands);
        if (state.dynamic_state_observation_enabled) {
            state.depth_state_emissions++;
            state.last_depth_state = *depth;
        }
    $endif
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
}

fn PreparedRenderPass? lower_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    if (desc == null
        || desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS
        || (desc.colors.len > 0 && desc.colors.ptr == null)) {
        return gpu::INVALID_ARGUMENT~;
    }
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
    bool resource_agnostic =
        state.caps.resource_agnostic_texture_sync;
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
            layout:            gpu::TextureLayout.COLOR_ATTACHMENT,
            format:            view.format,
            resource_agnostic: resource_agnostic,
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
                layout:            gpu::TextureLayout.COLOR_ATTACHMENT,
                format:            resolve.format,
                resource_agnostic: resource_agnostic,
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
        pass.depth_attachment = vk::rendering_attachment_info();
        pass.depth_attachment.image_view = depth.native;
        pass.depth_attachment.image_layout = texture_layout_to_vk(
            layout:            gpu::TextureLayout.DEPTH_ATTACHMENT,
            format:            depth.format,
            resource_agnostic: resource_agnostic,
        );
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

fn PreparedRenderPass? checked_prepare_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    ZString operation,
) {
    if (commands == null || desc == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
        if (desc.width > view.width || desc.height > view.height) {
            ZString rejected_field = desc.width > view.width
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
            operation:   operation,
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
    gpu::internal::CommandRecord* commands,
    PreparedRenderPass* pass,
) {
    VkDeviceState* state = command_device_state(commands);
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedRenderPass* pass,
) {
    VkCommandRecord* record = backend_command(commands);
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
    $if env::TESTING:
        VkDeviceState* state = command_device_state(commands);
        if (state.dynamic_state_observation_enabled) {
            state.begin_rendering_emissions++;
        }
    $endif
    record.active_render = pass.compatibility;
    record.root.state = gpu::internal::CommandState.RECORDING_RENDER_PASS;
}

fn void record_render_pass_with_state(
    gpu::internal::CommandRecord* commands,
    PreparedRenderPass* pass,
    LoweredGraphicsState* graphics,
) {
    record_render_pass(commands, pass);
    emit_graphics_state(commands, graphics);
}

fn void? trusted_begin_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    record_render_pass(commands, &pass);
}

fn void? trusted_tracking_begin_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass(commands, &pass);
}

fn void? checked_begin_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = checked_prepare_render_pass(
        commands,
        desc,
        "cmd_begin_render_pass",
    )!;
    record_render_pass(commands, &pass);
}

fn void? checked_tracking_begin_render_pass(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = checked_prepare_render_pass(
        commands,
        desc,
        "cmd_begin_render_pass",
    )!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass(commands, &pass);
}

fn void? trusted_begin_render_pass_with_state(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    gpu::GraphicsState* state,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    LoweredGraphicsState graphics = lower_graphics_state(state)!;
    record_render_pass_with_state(commands, &pass, &graphics);
}

fn void? trusted_tracking_begin_render_pass_with_state(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    gpu::GraphicsState* state,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    LoweredGraphicsState graphics = lower_graphics_state(state)!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass_with_state(commands, &pass, &graphics);
}

fn void? checked_begin_render_pass_with_state(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    gpu::GraphicsState* state,
) {
    PreparedRenderPass pass = checked_prepare_render_pass(
        commands,
        desc,
        "cmd_begin_render_pass_with_state",
    )!;
    LoweredGraphicsState graphics = checked_prepare_graphics_state(
        device:    command_device_state(commands),
        operation: "cmd_begin_render_pass_with_state",
        state:     state,
    )!;
    record_render_pass_with_state(commands, &pass, &graphics);
    backend_command(commands).graphics_state_initialized = true;
}

fn void? checked_tracking_begin_render_pass_with_state(
    gpu::internal::CommandRecord* commands,
    gpu::RenderPassDesc* desc,
    gpu::GraphicsState* state,
) {
    PreparedRenderPass pass = checked_prepare_render_pass(
        commands,
        desc,
        "cmd_begin_render_pass_with_state",
    )!;
    LoweredGraphicsState graphics = checked_prepare_graphics_state(
        device:    command_device_state(commands),
        operation: "cmd_begin_render_pass_with_state",
        state:     state,
    )!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_render_pass(commands, &pass)!;
    record_render_pass_with_state(commands, &pass, &graphics);
    record.graphics_state_initialized = true;
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

fn vk::Viewport lower_viewport(gpu::Viewport* viewport) @private {
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
    gpu::internal::CommandRecord* commands,
    vk::Viewport* viewport,
) @private {
    VkCommandRecord* record = backend_command(commands);
    $if env::TESTING:
        VkDeviceState* state = command_device_state(commands);
        if (state.dynamic_state_observation_enabled) {
            state.viewport_emissions++;
            state.last_viewport = *viewport;
        }
    $endif
    vk::cmd_set_viewport(record.command_buffer, 0, 1, viewport);
    gpu::internal::note_command_native_call();
}

fn bool graphics_state_settable(gpu::internal::CommandState state) =>
    state == gpu::internal::CommandState.RECORDING
        || state == gpu::internal::CommandState.RECORDING_RENDER_PASS;

fn void? validate_graphics_state_queue(VkCommandRecord* record) @private {
    if (!record.queue.roles.graphics) return gpu::INVALID_ARGUMENT~;
}

fn void? trusted_set_viewport(
    gpu::internal::CommandRecord* commands,
    gpu::Viewport* viewport,
) {
    if (viewport == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::Viewport native = lower_viewport(viewport);
    emit_viewport(commands, &native);
}

fn void? checked_set_viewport(
    gpu::internal::CommandRecord* commands,
    gpu::Viewport* viewport,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = command_device_state(commands);
    note_command_semantic_check(state);
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
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

fn vk::Rect2D lower_scissor(gpu::ScissorRect* scissor) @private {
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

fn LoweredGraphicsState? lower_graphics_state(
    gpu::GraphicsState* state,
) @private {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    return {
        .viewport = lower_viewport(&state.viewport),
        .scissor  = lower_scissor(&state.scissor),
        .raster   = state.raster,
        .depth    = state.depth,
    };
}

fn LoweredGraphicsState? checked_prepare_graphics_state(
    VkDeviceState* device,
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
    return lower_graphics_state(state);
}

fn void emit_scissor(
    gpu::internal::CommandRecord* commands,
    vk::Rect2D* scissor,
) @private {
    VkCommandRecord* record = backend_command(commands);
    $if env::TESTING:
        VkDeviceState* state = command_device_state(commands);
        if (state.dynamic_state_observation_enabled) {
            state.scissor_emissions++;
            state.last_scissor = *scissor;
        }
    $endif
    vk::cmd_set_scissor(record.command_buffer, 0, 1, scissor);
    gpu::internal::note_command_native_call();
}

fn void emit_graphics_state(
    gpu::internal::CommandRecord* commands,
    LoweredGraphicsState* state,
) @private {
    emit_viewport(commands, &state.viewport);
    emit_scissor(commands, &state.scissor);
    emit_raster_state(commands, &state.raster);
    emit_depth_state(commands, &state.depth);
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
) @private {
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
    gpu::internal::CommandRecord* commands,
    LoweredColorState* state,
) @private {
    if (state.count == 0) return;
    VkCommandRecord* record = backend_command(commands);
    VkDynamicColorDispatch* dispatch =
        &command_device_state(commands).device_dispatch.dynamic_color;
    dispatch.set_blend_enable(
        record.command_buffer,
        0,
        state.count,
        &state.blend_enables[0],
    );
    gpu::internal::note_command_native_call();
    dispatch.set_blend_equation(
        record.command_buffer,
        0,
        state.count,
        &state.blend_equations[0],
    );
    gpu::internal::note_command_native_call();
    dispatch.set_write_mask(
        record.command_buffer,
        0,
        state.count,
        &state.write_masks[0],
    );
    gpu::internal::note_command_native_call();
}

fn fault color_state_rejection(
    VkDeviceState* state,
    ZString field,
    ZString invariant,
) @private {
    DynamicStateRejection rejection = {
        .field     = field,
        .invariant = invariant,
    };
    return report_dynamic_state_rejection(
        state,
        "cmd_set_color_state",
        rejection,
    );
}

fn LoweredColorState? checked_prepare_color_state(
    VkDeviceState* state,
    VkCommandRecord* record,
    gpu::ColorState* color,
) @private {
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     "cmd_set_color_state",
    )!;
    if (pipeline.color_profile != GraphicsColorProfile.DYNAMIC) {
        return color_state_rejection(
            state,
            "pipeline",
            "selected graphics pipeline must use dynamic color state",
        )~;
    }
    if (color.targets.len != pipeline.render.color_count) {
        return color_state_rejection(
            state,
            "ColorState.targets",
            "target count must match the selected dynamic pipeline",
        )~;
    }
    if (record.root.state
            == gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        if (catch compatibility_fault =
            validate_pipeline_render_compatibility(
                pipeline,
                &record.active_render,
            )) {
            return color_state_rejection(
                state,
                "pipeline",
                "selected pipeline formats must match the active render pass",
            )~;
        }
    }
    foreach (i, &target : color.targets) {
        if (!blend_state_valid(&target.blend)) {
            return color_state_rejection(
                state,
                "ColorState.targets.blend",
                "enabled blend factors and operations must be defined",
            )~;
        }
        if (!color_write_mask_valid(target.write_mask)) {
            return color_state_rejection(
                state,
                "ColorState.targets.write_mask",
                "write mask may contain only RGBA channel bits",
            )~;
        }
        if (target.blend.enable
            && color_format_is_integer(
                pipeline.render.color_formats[i],
            )) {
            return color_state_rejection(
                state,
                "ColorState.targets.blend.enable",
                "integer color targets do not support blending",
            )~;
        }
    }
    return lower_color_state(color);
}

fn void? trusted_set_color_state(
    gpu::internal::CommandRecord* commands,
    gpu::ColorState* state,
) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    LoweredColorState color = lower_color_state(state)!;
    emit_color_state(commands, &color);
}

fn void? checked_set_color_state(
    gpu::internal::CommandRecord* commands,
    gpu::ColorState* state,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* device = command_device_state(commands);
    note_command_semantic_check(device);
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    LoweredColorState color =
        checked_prepare_color_state(device, record, state)!;
    emit_color_state(commands, &color);
    record.color_state =
        color_compatibility(&record.logical_pipeline.render);
    record.color_state_initialized = true;
}

fn void? trusted_set_graphics_state(
    gpu::internal::CommandRecord* commands,
    gpu::GraphicsState* state,
) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    LoweredGraphicsState graphics = lower_graphics_state(state)!;
    emit_graphics_state(commands, &graphics);
}

fn void? checked_set_graphics_state(
    gpu::internal::CommandRecord* commands,
    gpu::GraphicsState* state,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* device = command_device_state(commands);
    note_command_semantic_check(device);
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    LoweredGraphicsState graphics = checked_prepare_graphics_state(
        device,
        "cmd_set_graphics_state",
        state,
    )!;
    emit_graphics_state(commands, &graphics);
    record.graphics_state_initialized = true;
}

fn void reset_dynamic_state_observation(
    VkDeviceState* state,
) @if(env::TESTING) {
    state.dynamic_state_observation_enabled = true;
    state.begin_rendering_emissions = 0;
    state.viewport_emissions = 0;
    state.scissor_emissions = 0;
    state.raster_state_emissions = 0;
    state.depth_state_emissions = 0;
    state.last_viewport = {};
    state.last_scissor = {};
    state.last_raster_state = {};
    state.last_depth_state = {};
}

fn DynamicStateObservation dynamic_state_observation(
    VkDeviceState* state,
) @if(env::TESTING) {
    return {
        .begin_rendering_emissions = state.begin_rendering_emissions,
        .viewport_emissions        = state.viewport_emissions,
        .scissor_emissions         = state.scissor_emissions,
        .raster_state_emissions    = state.raster_state_emissions,
        .depth_state_emissions     = state.depth_state_emissions,
        .viewport                  = state.last_viewport,
        .scissor                   = state.last_scissor,
        .raster                    = state.last_raster_state,
        .depth                     = state.last_depth_state,
    };
}

fn void? trusted_set_scissor(
    gpu::internal::CommandRecord* commands,
    gpu::ScissorRect* scissor,
) {
    if (scissor == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    vk::Rect2D native = lower_scissor(scissor);
    emit_scissor(commands, &native);
}

fn void? checked_set_scissor(
    gpu::internal::CommandRecord* commands,
    gpu::ScissorRect* scissor,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = command_device_state(commands);
    note_command_semantic_check(state);
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    validate_scissor(state, "cmd_set_scissor", scissor)!;
    vk::Rect2D native = lower_scissor(scissor);
    emit_scissor(commands, &native);
}

fn void? vk_cmd_end_render_pass(gpu::internal::CommandRecord* commands) @private {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    gpu::internal::note_command_native_call();
    record.active_render = {};
    record.root.state = gpu::internal::CommandState.RECORDING;
}

fn void? vk_cmd_set_depth_state(
    gpu::internal::CommandRecord* commands,
    gpu::DepthState* depth,
) @private {
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (depth == null || !compare_op_valid(depth.compare)) {
        return gpu::INVALID_ARGUMENT~;
    }
    emit_depth_state(commands, depth);
}

fn void? trusted_set_depth_state(
    gpu::internal::CommandRecord* commands,
    gpu::DepthState* depth,
) {
    if (depth == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    emit_depth_state(commands, depth);
}

fn void? checked_set_depth_state(
    gpu::internal::CommandRecord* commands,
    gpu::DepthState* depth,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(command_device_state(commands));
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    return vk_cmd_set_depth_state(commands, depth);
}

fn void? vk_cmd_set_raster_state(
    gpu::internal::CommandRecord* commands,
    gpu::DynamicRasterState* raster,
) @private {
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
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
    emit_raster_state(commands, raster);
}

fn void? trusted_set_raster_state(
    gpu::internal::CommandRecord* commands,
    gpu::DynamicRasterState* raster,
) {
    if (raster == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    emit_raster_state(commands, raster);
}

fn void? checked_set_raster_state(
    gpu::internal::CommandRecord* commands,
    gpu::DynamicRasterState* raster,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    note_command_semantic_check(command_device_state(commands));
    VkCommandRecord* record = backend_command(commands);
    if (!graphics_state_settable(record.root.state)) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_graphics_state_queue(record)!;
    return vk_cmd_set_raster_state(commands, raster);
}

fn void? validate_dynamic_color_execution_state(
    VkCommandRecord* record,
    BoundPipeline* pipeline,
) @private {
    if (pipeline.color_profile != GraphicsColorProfile.DYNAMIC
        || pipeline.render.color_count == 0) return;
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
    VkCommandRecord* record,
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
    validate_dynamic_color_execution_state(record, pipeline)!;
    return pipeline;
}

fn void push_graphics_roots(
    VkCommandRecord* record,
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

fn BoundPipeline? graphics_pipeline_snapshot(VkCommandRecord* record) {
    return bound_pipeline_snapshot(record);
}

fn void record_draw(
    gpu::internal::CommandRecord* commands,
    BoundPipeline* pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedIndexedDraw? checked_prepare_indexed_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndexedDraw* draw,
) {
    return track_span_command_reference(
        command_device_state(commands),
        backend_command(commands),
        &draw.index_span,
    );
}

fn void record_indexed_draw(
    gpu::internal::CommandRecord* commands,
    PreparedIndexedDraw* draw,
) {
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = checked_prepare_indexed_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    PreparedIndexedDraw draw = checked_prepare_indexed_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedIndirectDraw? checked_prepare_indirect_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndirectDraw* draw,
) {
    return track_span_command_reference(
        command_device_state(commands),
        backend_command(commands),
        &draw.args_span,
    );
}

fn void record_indirect_draw(
    gpu::internal::CommandRecord* commands,
    PreparedIndirectDraw* draw,
) {
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = checked_prepare_indirect_draw(
        commands:       commands,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        draw_count:     draw_count,
    )!;
    record_indirect_draw(commands, &draw);
}

fn void? checked_tracking_draw_indirect(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    PreparedIndirectDraw draw = checked_prepare_indirect_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedIndexedIndirectDraw? checked_prepare_indexed_indirect_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    VkDeviceState* state = command_device_state(commands);
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndexedIndirectDraw* draw,
) {
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_draw(commands, &draw)!;
    record_indexed_indirect_draw(commands, &draw);
}

fn void? checked_draw_indexed_indirect(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw =
        checked_prepare_indexed_indirect_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectDraw draw =
        checked_prepare_indexed_indirect_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            draw_count:     draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_draw(commands, &draw)!;
    record_indexed_indirect_draw(commands, &draw);
}

<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn PreparedIndexedIndirectCountDraw? lower_indexed_indirect_count_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedIndexedIndirectCountDraw? checked_prepare_indexed_indirect_count_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    VkDeviceState* state = command_device_state(commands);
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedIndexedIndirectCountDraw* draw,
) {
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_count_draw(commands, &draw)!;
    record_indexed_indirect_count_draw(commands, &draw);
}

fn void? checked_draw_indexed_indirect_count(
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        checked_prepare_indexed_indirect_count_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedIndexedIndirectCountDraw draw =
        checked_prepare_indexed_indirect_count_draw(
            commands:       commands,
            vertex_root:    vertex_root,
            fragment_root:  fragment_root,
            args:           args,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_indexed_indirect_count_draw(commands, &draw)!;
    record_indexed_indirect_count_draw(commands, &draw);
}

fn BoundPipeline*? validate_generated_graphics_execution_state(
    VkDeviceState* state,
    VkCommandRecord* record,
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
    validate_dynamic_color_execution_state(record, pipeline)!;
    return pipeline;
}

fn void? validate_color_execution_state_for_test(
    VkDeviceState* state,
    VkCommandRecord* record,
    bool generated,
) @if(env::TESTING) {
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedGeneratedDraw? checked_prepare_generated_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedGeneratedDraw* draw,
) {
    VkDeviceState* state = command_device_state(commands);
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedGeneratedDraw* draw,
) {
    if (draw.max_count == 0) return;
    execute_generated_work(
        state:         command_device_state(commands),
        record:        backend_command(commands),
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_draw(commands, &draw)!;
    record_generated_draw(commands, &draw)!;
}

fn void? checked_draw_generated(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = checked_prepare_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    record_generated_draw(commands, &draw)!;
}

fn void? checked_tracking_draw_generated(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    PreparedGeneratedDraw draw = checked_prepare_generated_draw(
        commands:       commands,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    )!;
    if (draw.max_count == 0) return;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_draw(commands, &draw)!;
    record_generated_draw(commands, &draw)!;
}

fn PreparedGeneratedIndexedDraw? lower_generated_indexed_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    VkCommandRecord* record = backend_command(commands);
    if (record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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

fn PreparedGeneratedIndexedDraw? checked_prepare_generated_indexed_draw(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkCommandRecord* record = backend_command(commands);
    if (record == null || record.root.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = command_device_state(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    VkDeviceState* state = command_device_state(commands);
    VkCommandRecord* record = backend_command(commands);
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
    gpu::internal::CommandRecord* commands,
    PreparedGeneratedIndexedDraw* draw,
) {
    if (draw.max_count == 0) return;
    VkCommandRecord* record = backend_command(commands);
    execute_generated_work(
        state:         command_device_state(commands),
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
    gpu::internal::CommandRecord* commands,
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
    gpu::internal::CommandRecord* commands,
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
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_indexed_draw(commands, &draw)!;
    record_generated_indexed_draw(commands, &draw)!;
}

fn void? checked_draw_indexed_generated(
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw =
        checked_prepare_generated_indexed_draw(
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
    gpu::internal::CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    PreparedGeneratedIndexedDraw draw =
        checked_prepare_generated_indexed_draw(
            commands:       commands,
            records:        records,
            count_span:     count_span,
            max_draw_count: max_draw_count,
            index_span:     index_span,
            index_type:     index_type,
        )!;
    if (draw.max_count == 0) return;
    VkCommandRecord* record = backend_command(commands);
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        command_device_state(commands),
        record,
        checkpoint,
    );
    track_generated_indexed_draw(commands, &draw)!;
    record_generated_indexed_draw(commands, &draw)!;
}
