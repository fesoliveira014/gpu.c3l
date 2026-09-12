module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::math;
import std::atomic;

struct PreparedRenderPass {
    AttachmentViewSlot*[gpu::MAX_COLOR_ATTACHMENTS] color_slots;
    vk::RenderingAttachmentInfo[gpu::MAX_COLOR_ATTACHMENTS] attachments;
    AttachmentViewSlot*       depth_view;
    vk::RenderingAttachmentInfo depth_attachment;
    vk::RenderingAttachmentInfo stencil_attachment;
    uint                      width;
    uint                      height;
    uint                      color_count;
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
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            index_count;
    uint            instance_count;
    gpu::IndexType  index_type;
}

struct PreparedIndirectDraw {
    BoundPipeline*  pipeline;
    ResolvedSpan    args;
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
}

struct PreparedIndexedIndirectDraw {
    BoundPipeline*  pipeline;
    ResolvedSpan    args;
    ResolvedSpan    index;
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
    gpu::GpuAddress vertex_root;
    gpu::GpuAddress fragment_root;
    uint            max_draw_count;
    gpu::IndexType  index_type;
}

struct PreparedGeneratedDraw {
    BoundPipeline*                 pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
}

struct PreparedGeneratedIndexedDraw {
    BoundPipeline*                 pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    ResolvedSpan                   index;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
    gpu::IndexType                 index_type;
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
    gpu::GpuAddress mesh_root;
    gpu::GpuAddress fragment_root;
    uint            draw_count;
    bool            counted;
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
    record.raster_topology = raster.topology;
    record.raster_topology_emitted = record.logical_pipeline.kind != PipelineKind.MESH;
    if (record.raster_topology_emitted) {
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
    };
    foreach (i, &target : desc.colors) {
        AttachmentViewSlot* view = state.attachment_views.get(target.view)!;
        if (view.native == null) return gpu::INVALID_RESOURCE_STATE~;
        pass.color_slots[i] = view;
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
    record.state = gpu::internal::CommandState.RECORDING_RENDER_PASS;
}

fn void? vk_cmd_begin_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    PreparedRenderPass pass = lower_render_pass(commands, desc)!;
    record_render_pass(commands, &pass);
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

fn void? vk_cmd_set_viewport(
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

fn vk::Rect2D lower_scissor(gpu::ScissorRect* scissor) {
    return {
        .offset = { .x = scissor.x, .y = scissor.y },
        .extent = {
            .width  = (uint)scissor.width,
            .height = (uint)scissor.height,
        },
    };
}

fn LoweredGraphicsState? lower_graphics_state(
    gpu::GraphicsState* state,
) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    return {
        .viewport = lower_viewport(&state.viewport),
        .scissor  = lower_scissor(&state.scissor),
        .raster   = state.raster,
        .depth    = state.depth,
        .stencil  = state.stencil,
        .color    = lower_color_state(&state.color)!,
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

fn void? vk_cmd_set_graphics_state(
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
    record.graphics_state_initialized = true;
}

fn void? vk_cmd_set_scissor(
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

fn void? vk_cmd_end_render_pass(CommandRecord* commands) @private {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    vk::cmd_end_rendering(record.command_buffer);
    record.state = gpu::internal::CommandState.RECORDING;
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
    uint push_stages = pipeline.kind == PipelineKind.MESH
        ? (uint)record.device_state.mesh_push_stages
        : (uint)vk::ShaderStageFlagBits.VERTEX_BIT
            | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    push_root_block(
        command_buffer: record.command_buffer,
        inline_root:    record.inline_root,
        inline_offset:  pipeline.inline_root_offset,
        layout:         pipeline.layout,
        stages:         (vk::ShaderStageFlags)push_stages,
        header:         &push,
        header_size:    GRAPHICS_ROOT_PUSH_SIZE,
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
    push_graphics_roots(
        record:        record,
        pipeline:      pipeline,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
    );
    vk::cmd_draw(
        command_buffer: record.command_buffer,
        vertex_count:   vertex_count,
        instance_count: instance_count,
        first_vertex:   0,
        first_instance: 0,
    );
}

fn void? vk_cmd_draw(
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
        argument_size: index_element_bytes(index_type),
        count:         index_count,
        max_count:     uint::max,
    )!;
    return {
        .pipeline      = borrow_bound_pipeline(record)!,
        .index         = resolve_span_slot(
            state:      state,
            span:       &index_span,
            required:   required,
        )!,
        .vertex_root   = vertex_root,
        .fragment_root = fragment_root,
        .index_count   = index_count,
        .instance_count = instance_count,
        .index_type    = index_type,
    };
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

fn void? vk_cmd_draw_indexed(
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
        .vertex_root   = vertex_root,
        .fragment_root = fragment_root,
        .draw_count    = draw_count,
    };
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

fn void? vk_cmd_draw_indirect(
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
        .index = resolve_span_slot(
            state:      state,
            span:       &index_span,
            required:   0,
        )!,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .draw_count     = draw_count,
        .index_type     = index_type,
    };
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

fn void? vk_cmd_draw_indexed_indirect(
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
        .index = resolve_span_slot(
            state:      state,
            span:       &index_span,
            required:   0,
        )!,
        .vertex_root    = vertex_root,
        .fragment_root  = fragment_root,
        .max_draw_count = max_draw_count,
        .index_type     = index_type,
    };
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

fn void? vk_cmd_draw_indexed_indirect_count(
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

fn void? vk_cmd_draw_generated(
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
    draw.index = resolve_span_slot(
        state:      state,
        span:       &index_span,
        required:   0,
    )!;
    return draw;
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

fn void? vk_cmd_draw_indexed_generated(
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

fn void record_mesh_draw(CommandRecord* record, PreparedMeshDraw* draw) @private {
    push_graphics_roots(
        record:        record,
        pipeline:      draw.pipeline,
        vertex_root:   draw.mesh_root,
        fragment_root: draw.fragment_root,
    );
    record.device_state.device_dispatch.mesh_shaders.cmd_draw_mesh_tasks(
        command_buffer: record.command_buffer,
        group_count_x:  draw.groups.x,
        group_count_y:  draw.groups.y,
        group_count_z:  draw.groups.z,
    );
}

fn void record_mesh_indirect_draw(
    CommandRecord* record,
    PreparedMeshIndirectDraw* draw,
) @private {
    push_graphics_roots(
        record:        record,
        pipeline:      draw.pipeline,
        vertex_root:   draw.mesh_root,
        fragment_root: draw.fragment_root,
    );
    VkMeshShaderDispatch* dispatch = &record.device_state.device_dispatch.mesh_shaders;
    if (draw.counted) {
        dispatch.cmd_draw_mesh_tasks_indirect_count(
            command_buffer:      record.command_buffer,
            buffer:              draw.args.slot.buffer,
            offset:              (vk::DeviceSize)draw.args.offset,
            count_buffer:        draw.count.slot.buffer,
            count_buffer_offset: (vk::DeviceSize)draw.count.offset,
            max_draw_count:      draw.draw_count,
            stride:              gpu::DrawMeshTasksIndirectCommand::size,
        );
        return;
    }
    dispatch.cmd_draw_mesh_tasks_indirect(
        command_buffer: record.command_buffer,
        buffer:         draw.args.slot.buffer,
        offset:         (vk::DeviceSize)draw.args.offset,
        draw_count:     draw.draw_count,
        stride:         gpu::DrawMeshTasksIndirectCommand::size,
    );
}

fn void? vk_cmd_draw_mesh_tasks(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::Vec3u groups,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = record.device_state;
    if (!state.mesh_shaders_enabled) return gpu::UNSUPPORTED_FEATURE~;
    PreparedMeshDraw draw = {
        .pipeline      = borrow_bound_pipeline(record)!,
        .mesh_root     = mesh_root,
        .fragment_root = fragment_root,
        .groups        = groups,
    };
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

fn void? vk_cmd_draw_mesh_tasks_indirect(
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

fn void? vk_cmd_draw_mesh_tasks_indirect_count(
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
