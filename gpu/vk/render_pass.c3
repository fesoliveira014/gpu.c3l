module gpu::vk @private;

import gpu @public;
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

<* Begin dynamic rendering. Attachments must sit in their attachment layouts,
   have matching usage, and cover the pass dimensions. Color views honor the
   target's mip/layer; the depth view uses mip zero. *>
fn void? vk_cmd_begin_render_pass(
    gpu::CommandList* commands,
    gpu::RenderPassDesc* desc,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
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

    @pool() {
        TextureSlot*[] color_slots = {};
        TextureSlot*[] resolve_slots = {};
        vk::RenderingAttachmentInfo[] attachments = {};
        RenderCompatibility compatibility;
        compatibility.color_count = (uint)desc.colors.len;
        bool samples_initialized;
        sz resolve_count;

        if (desc.colors.len > 0) {
            color_slots = mem::talloc_array(TextureSlot*, desc.colors.len);
            resolve_slots = mem::talloc_array(
                TextureSlot*,
                desc.colors.len,
            );
            attachments = mem::talloc_array(
                vk::RenderingAttachmentInfo,
                desc.colors.len,
            );
            foreach (i, &target : desc.colors) {
                resolve_slots[i] = null;
                TextureSlot* slot = state.textures.get(target.texture)!;
                validate_texture_recording_access(
                    state:     state,
                    record:    record,
                    slot:      slot,
                    handle:    target.texture,
                    operation: "cmd_begin_render_pass",
                )!;
                color_slots[i] = slot;
                if (!slot.usage.color_attach) {
                    return report_contract_failure(
                        state:               state,
                        operation:           "cmd_begin_render_pass",
                        public_fault:        gpu::INVALID_ARGUMENT,
                        resource_kind:       gpu::DebugResourceKind.TEXTURE,
                        has_public_identity: true,
                        index:               target.texture.index,
                        generation:          target.texture.generation,
                        debug_name:          slot_name(&slot.debug_name[0]),
                        rejected_field:      "TextureUsage.color_attach",
                        invariant:           "color attachment must include color_attach usage",
                        backend_text:        null,
                    )~;
                }
                if (target.mip_level >= slot.mip_levels
                    || target.array_layer >= slot.array_layers) {
                    return gpu::INVALID_ARGUMENT~;
                }
                MipExtent extent = texture_mip_extent(
                    slot,
                    target.mip_level,
                );
                if (desc.width > extent.width
                    || desc.height > extent.height) {
                    ZString rejected_field = desc.width > extent.width
                        ? "RenderPassDesc.width"
                        : "RenderPassDesc.height";
                    return report_contract_failure(
                        state:               state,
                        operation:           "cmd_begin_render_pass",
                        public_fault:        gpu::INVALID_ARGUMENT,
                        resource_kind:       gpu::DebugResourceKind.TEXTURE,
                        has_public_identity: true,
                        index:               target.texture.index,
                        generation:          target.texture.generation,
                        debug_name:          slot_name(&slot.debug_name[0]),
                        rejected_field:      rejected_field,
                        invariant:           "render area must fit every color attachment mip",
                        backend_text:        null,
                    )~;
                }
                if (!samples_initialized) {
                    compatibility.sample_count = slot.sample_count;
                    samples_initialized = true;
                } else if (slot.sample_count
                    != compatibility.sample_count) {
                    return gpu::INVALID_ARGUMENT~;
                }
                compatibility.color_formats[i] = slot.format;

                if (target.resolve_texture
                    == gpu::TEXTURE_HANDLE_INVALID) {
                    continue;
                }
                if (target.resolve_texture == target.texture
                    || slot.sample_count == gpu::SampleCount.ONE) {
                    return gpu::INVALID_ARGUMENT~;
                }
                TextureSlot* resolve_slot =
                    state.textures.get(target.resolve_texture)!;
                validate_texture_recording_access(
                    state:     state,
                    record:    record,
                    slot:      resolve_slot,
                    handle:    target.resolve_texture,
                    operation: "cmd_begin_render_pass",
                )!;
                if (!resolve_slot.usage.color_attach
                    || resolve_slot.sample_count
                        != gpu::SampleCount.ONE
                    || resolve_slot.format != slot.format
                    || target.resolve_mip_level
                        >= resolve_slot.mip_levels
                    || target.resolve_array_layer
                        >= resolve_slot.array_layers) {
                    return gpu::INVALID_ARGUMENT~;
                }
                MipExtent resolve_extent = texture_mip_extent(
                    resolve_slot,
                    target.resolve_mip_level,
                );
                if (desc.width > resolve_extent.width
                    || desc.height > resolve_extent.height) {
                    return gpu::INVALID_ARGUMENT~;
                }
                resolve_slots[i] = resolve_slot;
                resolve_count++;
            }
        }

        TextureSlot* depth_slot = null;
        if (desc.depth != null) {
            depth_slot = state.textures.get(desc.depth.texture)!;
            validate_texture_recording_access(
                state:     state,
                record:    record,
                slot:      depth_slot,
                handle:    desc.depth.texture,
                operation: "cmd_begin_render_pass",
            )!;
            if (!depth_slot.usage.depth_attach
                || desc.width > depth_slot.width
                || desc.height > depth_slot.height) {
                return gpu::INVALID_ARGUMENT~;
            }
            if (samples_initialized
                && depth_slot.sample_count
                    != compatibility.sample_count) {
                return gpu::INVALID_ARGUMENT~;
            }
            compatibility.depth_format = depth_slot.format;
            if (!samples_initialized) {
                compatibility.sample_count = depth_slot.sample_count;
                samples_initialized = true;
            }
        }

        validate_active_pipeline_render_compatibility(
            state,
            record,
            &compatibility,
        )!;

        sz max_created_views = desc.colors.len
            + resolve_count
            + (desc.depth != null ? 1 : 0);
        CachedViewCreation[] creations = mem::talloc_array(
            CachedViewCreation,
            max_created_views,
        );
        uint creation_count;
        state.texture_view_cache_mutex.lock()!!;
        defer state.texture_view_cache_mutex.unlock();
        defer catch rollback_texture_view_creations(
            state,
            creations[:creation_count],
        );

        foreach (i, &target : desc.colors) {
            gpu::TextureViewDesc view_desc = {
                .base_mip    = target.mip_level,
                .mip_count   = 1,
                .base_layer  = target.array_layer,
                .layer_count = 1,
            };
            CachedViewCreation creation;
            vk::ImageView view = resolve_texture_view_tracked(
                state:     state,
                slot:      color_slots[i],
                texture:   target.texture,
                view:      &view_desc,
                operation: "cmd_begin_render_pass",
                creation:  &creation,
            )!;
            if (creation.cache != null) {
                creations[creation_count++] = creation;
            }

            attachments[i] = vk::rendering_attachment_info();
            attachments[i].image_view = view;
            attachments[i].image_layout =
                vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
            attachments[i].load_op = load_op_to_vk(target.load_op);
            attachments[i].store_op = store_op_to_vk(target.store_op);
            attachments[i].clear_value.color = clear_color_to_vk(
                color_slots[i].format,
                target.clear,
            );

            if (resolve_slots[i] != null) {
                gpu::TextureViewDesc resolve_view_desc = {
                    .base_mip    = target.resolve_mip_level,
                    .mip_count   = 1,
                    .base_layer  = target.resolve_array_layer,
                    .layer_count = 1,
                };
                CachedViewCreation resolve_creation;
                vk::ImageView resolve_view = resolve_texture_view_tracked(
                    state:     state,
                    slot:      resolve_slots[i],
                    texture:   target.resolve_texture,
                    view:      &resolve_view_desc,
                    operation: "cmd_begin_render_pass",
                    creation:  &resolve_creation,
                )!;
                if (resolve_creation.cache != null) {
                    creations[creation_count++] = resolve_creation;
                }
                attachments[i].resolve_mode =
                    color_resolve_mode(color_slots[i].format);
                attachments[i].resolve_image_view = resolve_view;
                attachments[i].resolve_image_layout =
                    vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
            }
        }

        vk::RenderingAttachmentInfo depth_attach;
        vk::RenderingInfo rendering = vk::rendering_info();
        rendering.render_area = {
            .offset = { .x = 0, .y = 0 },
            .extent = {
                .width  = desc.width,
                .height = desc.height,
            },
        };
        rendering.layer_count = 1;
        rendering.color_attachment_count = (uint)attachments.len;
        rendering.p_color_attachments = attachments.ptr;
        if (desc.depth != null) {
            gpu::TextureViewDesc depth_view_desc = {
                .mip_count   = 1,
                .layer_count = 1,
            };
            CachedViewCreation depth_creation;
            vk::ImageView depth_view = resolve_texture_view_tracked(
                state:     state,
                slot:      depth_slot,
                texture:   desc.depth.texture,
                view:      &depth_view_desc,
                operation: "cmd_begin_render_pass",
                creation:  &depth_creation,
            )!;
            if (depth_creation.cache != null) {
                creations[creation_count++] = depth_creation;
            }
            depth_attach = vk::rendering_attachment_info();
            depth_attach.image_view = depth_view;
            depth_attach.image_layout =
                vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attach.load_op = load_op_to_vk(desc.depth.load_op);
            depth_attach.store_op =
                store_op_to_vk(desc.depth.store_op);
            depth_attach.clear_value.depth_stencil = {
                .depth   = desc.depth.clear.depth,
                .stencil = desc.depth.clear.stencil,
            };
            rendering.p_depth_attachment = &depth_attach;
        }
        vk::cmd_begin_rendering(cb, &rendering);
        commit_texture_view_creations(
            state,
            creations[:creation_count],
        );
        record.active_render = compatibility;
    };

    vk::Viewport viewport = {
        .x         = 0.0,
        .y         = 0.0,
        .width     = (float)desc.width,
        .height    = (float)desc.height,
        .min_depth = 0.0,
        .max_depth = 1.0,
    };
    vk::cmd_set_viewport(cb, 0, 1, &viewport);
    vk::Rect2D scissor = {
        .offset = { .x = 0, .y = 0 },
        .extent = {
            .width  = desc.width,
            .height = desc.height,
        },
    };
    vk::cmd_set_scissor(cb, 0, 1, &scissor);

    record.active_render_width = desc.width;
    record.active_render_height = desc.height;
    record.depth_state_set = false;
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
    if (record.active_pipeline.is_valid()) {
        VkDeviceState* state =
            (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
        PipelineSlot* slot = state.pipelines.get(record.active_pipeline)!;
        // Graphics bindings are scoped to their pass; a binding that leaked into
        // the next begin would fail its compatibility preflight.
        if ((PipelineKind)state.pipeline_cache.entries[slot.entry].key.kind
            == PipelineKind.GRAPHICS) {
            record.active_pipeline = gpu::PIPELINE_HANDLE_INVALID;
        }
    }
    record.active_render_width = 0;
    record.active_render_height = 0;
    record.active_render = {};
    record.depth_state_set = false;
    record.state = CommandState.RECORDING;
}

fn void? vk_cmd_set_depth_state(
    gpu::CommandList* commands,
    gpu::DepthState* depth,
) @private {
    CommandRecord* record = resolve_command(commands)!;
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
    vk::cmd_set_depth_write_enable(
        record.command_buffer,
        depth.write_enable ? vk::TRUE : vk::FALSE,
    );
    vk::cmd_set_depth_compare_op(
        record.command_buffer,
        compare_op_to_vk(depth.compare),
    );
    record.depth_state_set = true;
}

fn PipelineSlot*? validate_graphics_execution_state(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    ZString operation,
) @private {
    PipelineSlot* slot = active_pipeline_slot(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     operation,
    )!;
    if (!record.depth_state_set) return gpu::COMMAND_RECORDING_ERROR~;
    validate_root_address(
        state:          state,
        root:           vertex_root,
        operation:      operation,
        rejected_field: "vertex_root",
    )!;
    validate_root_address(
        state:          state,
        root:           fragment_root,
        operation:      operation,
        rejected_field: "fragment_root",
    )!;
    return slot;
}

fn void push_graphics_roots(
    CommandRecord* record,
    PipelineSlot* slot,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
) @private {
    gpu::GraphicsRootPush push = {
        .vertex_root_gpu   = vertex_root,
        .fragment_root_gpu = fragment_root,
    };
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(
        command_buffer: record.command_buffer,
        layout:         slot.layout,
        stage_flags:    (vk::ShaderStageFlags)push_stages,
        offset:         0,
        size:           GRAPHICS_ROOT_PUSH_SIZE,
        p_values:       &push,
    );
}

fn void? vk_cmd_draw(gpu::CommandList* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, uint vertex_count, uint instance_count) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    PipelineSlot* slot = validate_graphics_execution_state(
        state:         state,
        record:        record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        operation:     "cmd_draw",
    )!;
    push_graphics_roots(record, slot, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_draw(cb, vertex_count, instance_count, 0, 0);
}

fn void? vk_cmd_draw_indexed(gpu::CommandList* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan index_span, uint index_count, uint instance_count, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    PipelineSlot* slot = validate_graphics_execution_state(
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

    push_graphics_roots(record, slot, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_bind_index_buffer(cb, buffer.slot.buffer, (vk::DeviceSize)buffer.offset,
        index_type == gpu::IndexType.U16 ? vk::IndexType.UINT16 : vk::IndexType.UINT32);
    vk::cmd_draw_indexed(cb, index_count, instance_count, 0, 0, 0);
}

fn void? vk_cmd_draw_indirect(gpu::CommandList* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    PipelineSlot* slot = validate_graphics_execution_state(
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
    push_graphics_roots(record, slot, vertex_root, fragment_root);
    vk::CommandBuffer cb = record.command_buffer;
    vk::cmd_draw_indirect(
        command_buffer: cb,
        buffer:         args_buffer.slot.buffer,
        offset:         (vk::DeviceSize)args_buffer.offset,
        draw_count:     draw_count,
        stride:         gpu::DrawIndirectCommand::size,
    );
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
<* Index-range bounds live in the GPU-written arguments and are not validated
   here; out-of-bounds indices follow device robustness. *>
fn void? vk_cmd_draw_indexed_indirect(gpu::CommandList* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, uint draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    PipelineSlot* slot = validate_graphics_execution_state(
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
    push_graphics_roots(record, slot, vertex_root, fragment_root);
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
}

fn void? vk_cmd_draw_indexed_indirect_count(gpu::CommandList* commands, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, gpu::GpuSpan args, gpu::GpuSpan count_span, uint max_draw_count, gpu::GpuSpan index_span, gpu::IndexType index_type) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    PipelineSlot* slot = validate_graphics_execution_state(
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
    push_graphics_roots(record, slot, vertex_root, fragment_root);
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
}

fn PipelineSlot*? validate_generated_graphics_execution_state(
    VkDeviceState* state,
    CommandRecord* record,
    ZString operation,
) @private {
    PipelineSlot* pipeline = active_pipeline_slot(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.GRAPHICS,
        operation:     operation,
    )!;
    if (!record.depth_state_set) return gpu::COMMAND_RECORDING_ERROR~;
    return pipeline;
}

fn void? vk_cmd_draw_generated(
    gpu::CommandList* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    PipelineSlot* pipeline = validate_generated_graphics_execution_state(
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
        layout:        state.generated_draw_layout,
        shader_stages: (vk::ShaderStageFlags)stages,
        records:       &resolved_records,
        count:         &resolved_count,
        max_count:     max_draw_count,
    )!;
}

fn void? vk_cmd_draw_indexed_generated(
    gpu::CommandList* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    PipelineSlot* pipeline = validate_generated_graphics_execution_state(
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
        layout:        state.generated_indexed_draw_layout,
        shader_stages: (vk::ShaderStageFlags)stages,
        records:       &resolved_records,
        count:         &resolved_count,
        max_count:     max_draw_count,
    )!;
}
