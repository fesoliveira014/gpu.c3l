module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::thread;
import std::core::mem;

const uint GRAPHICS_ROOT_PUSH_SIZE = gpu::GraphicsRootPush::size;

fn vk::PrimitiveTopology topology_to_vk(gpu::PrimitiveTopology topology) @private {
    switch (topology) {
        case gpu::PrimitiveTopology.TRIANGLES: return vk::PrimitiveTopology.TRIANGLE_LIST;
        case gpu::PrimitiveTopology.LINES:     return vk::PrimitiveTopology.LINE_LIST;
        case gpu::PrimitiveTopology.POINTS:    return vk::PrimitiveTopology.POINT_LIST;
        default:                               return vk::PrimitiveTopology.TRIANGLE_LIST;
    }
}

fn vk::CompareOp compare_op_to_vk(gpu::CompareOp op) @private {
    switch (op) {
        case gpu::CompareOp.NEVER:         return vk::CompareOp.NEVER;
        case gpu::CompareOp.LESS:          return vk::CompareOp.LESS;
        case gpu::CompareOp.EQUAL:         return vk::CompareOp.EQUAL;
        case gpu::CompareOp.LESS_EQUAL:    return vk::CompareOp.LESS_OR_EQUAL;
        case gpu::CompareOp.GREATER:       return vk::CompareOp.GREATER;
        case gpu::CompareOp.NOT_EQUAL:     return vk::CompareOp.NOT_EQUAL;
        case gpu::CompareOp.GREATER_EQUAL: return vk::CompareOp.GREATER_OR_EQUAL;
        case gpu::CompareOp.ALWAYS:        return vk::CompareOp.ALWAYS;
        default:                           return vk::CompareOp.ALWAYS;
    }
}

fn vk::CullModeFlagBits cull_mode_to_vk(gpu::CullMode mode) @private {
    switch (mode) {
        case gpu::CullMode.NONE:  return vk::CullModeFlagBits.NONE;
        case gpu::CullMode.FRONT: return vk::CullModeFlagBits.FRONT_BIT;
        case gpu::CullMode.BACK:  return vk::CullModeFlagBits.BACK_BIT;
        default:                  return vk::CullModeFlagBits.NONE;
    }
}

fn vk::FrontFace front_face_to_vk(gpu::FrontFace face) @private {
    return face == gpu::FrontFace.CLOCKWISE
        ? vk::FrontFace.CLOCKWISE
        : vk::FrontFace.COUNTER_CLOCKWISE;
}

fn vk::BlendFactor blend_factor_to_vk(gpu::BlendFactor factor) @private {
    switch (factor) {
        case gpu::BlendFactor.ZERO:                return vk::BlendFactor.ZERO;
        case gpu::BlendFactor.ONE:                 return vk::BlendFactor.ONE;
        case gpu::BlendFactor.SRC_COLOR:           return vk::BlendFactor.SRC_COLOR;
        case gpu::BlendFactor.ONE_MINUS_SRC_COLOR: return vk::BlendFactor.ONE_MINUS_SRC_COLOR;
        case gpu::BlendFactor.SRC_ALPHA:           return vk::BlendFactor.SRC_ALPHA;
        case gpu::BlendFactor.ONE_MINUS_SRC_ALPHA: return vk::BlendFactor.ONE_MINUS_SRC_ALPHA;
        case gpu::BlendFactor.DST_ALPHA:           return vk::BlendFactor.DST_ALPHA;
        case gpu::BlendFactor.ONE_MINUS_DST_ALPHA: return vk::BlendFactor.ONE_MINUS_DST_ALPHA;
        default:                                   return vk::BlendFactor.ONE;
    }
}

fn vk::BlendOp blend_op_to_vk(gpu::BlendOp op) @private {
    switch (op) {
        case gpu::BlendOp.ADD:              return vk::BlendOp.ADD;
        case gpu::BlendOp.SUBTRACT:         return vk::BlendOp.SUBTRACT;
        case gpu::BlendOp.REVERSE_SUBTRACT: return vk::BlendOp.REVERSE_SUBTRACT;
        case gpu::BlendOp.MIN:              return vk::BlendOp.MIN;
        case gpu::BlendOp.MAX:              return vk::BlendOp.MAX;
        default:                            return vk::BlendOp.ADD;
    }
}

fn bool primitive_topology_valid(gpu::PrimitiveTopology topology) {
    switch (topology) {
        case gpu::PrimitiveTopology.TRIANGLES:
        case gpu::PrimitiveTopology.LINES:
        case gpu::PrimitiveTopology.POINTS:
            return true;
        default:
            return false;
    }
}

fn bool compare_op_valid(gpu::CompareOp op) {
    switch (op) {
        case gpu::CompareOp.NEVER:
        case gpu::CompareOp.LESS:
        case gpu::CompareOp.EQUAL:
        case gpu::CompareOp.LESS_EQUAL:
        case gpu::CompareOp.GREATER:
        case gpu::CompareOp.NOT_EQUAL:
        case gpu::CompareOp.GREATER_EQUAL:
        case gpu::CompareOp.ALWAYS:
            return true;
        default:
            return false;
    }
}

fn bool cull_mode_valid(gpu::CullMode mode) {
    switch (mode) {
        case gpu::CullMode.NONE:
        case gpu::CullMode.FRONT:
        case gpu::CullMode.BACK:
            return true;
        default:
            return false;
    }
}

fn bool front_face_valid(gpu::FrontFace face) {
    switch (face) {
        case gpu::FrontFace.COUNTER_CLOCKWISE:
        case gpu::FrontFace.CLOCKWISE:
            return true;
        default:
            return false;
    }
}

fn bool polygon_mode_valid(gpu::PolygonMode mode) {
    switch (mode) {
        case gpu::PolygonMode.FILL:
        case gpu::PolygonMode.LINE:
            return true;
        default:
            return false;
    }
}

fn bool blend_factor_valid(gpu::BlendFactor factor) {
    switch (factor) {
        case gpu::BlendFactor.ZERO:
        case gpu::BlendFactor.ONE:
        case gpu::BlendFactor.SRC_COLOR:
        case gpu::BlendFactor.ONE_MINUS_SRC_COLOR:
        case gpu::BlendFactor.SRC_ALPHA:
        case gpu::BlendFactor.ONE_MINUS_SRC_ALPHA:
        case gpu::BlendFactor.DST_ALPHA:
        case gpu::BlendFactor.ONE_MINUS_DST_ALPHA:
            return true;
        default:
            return false;
    }
}

fn bool blend_op_valid(gpu::BlendOp op) {
    switch (op) {
        case gpu::BlendOp.ADD:
        case gpu::BlendOp.SUBTRACT:
        case gpu::BlendOp.REVERSE_SUBTRACT:
        case gpu::BlendOp.MIN:
        case gpu::BlendOp.MAX:
            return true;
        default:
            return false;
    }
}

fn bool blend_state_valid(gpu::BlendState* blend) {
    if (!blend.enable) return true;
    return blend_factor_valid(blend.src_color)
        && blend_factor_valid(blend.dst_color)
        && blend_op_valid(blend.color_op)
        && blend_factor_valid(blend.src_alpha)
        && blend_factor_valid(blend.dst_alpha)
        && blend_op_valid(blend.alpha_op);
}

fn bool color_write_mask_valid(gpu::ColorWriteMask mask) @private {
    uint raw = (uint)mask;
    return (raw & (uint)gpu::COLOR_WRITE_ALL) == raw;
}

fn vk::ColorComponentFlags color_write_mask_to_vk(
    gpu::ColorWriteMask mask,
) @private {
    uint flags = 0;
    if (mask.red) flags |= (uint)vk::ColorComponentFlagBits.R_BIT;
    if (mask.green) flags |= (uint)vk::ColorComponentFlagBits.G_BIT;
    if (mask.blue) flags |= (uint)vk::ColorComponentFlagBits.B_BIT;
    if (mask.alpha) flags |= (uint)vk::ColorComponentFlagBits.A_BIT;
    return (vk::ColorComponentFlags)flags;
}

fn bool color_format_is_integer(gpu::Format format) @private {
    return format == gpu::Format.R8_UINT
        || format == gpu::Format.R16_UINT
        || format == gpu::Format.R32_UINT;
}

fn void? check_polygon_mode_support(gpu::PolygonMode mode, bool line_supported) {
    if (mode == gpu::PolygonMode.LINE && !line_supported) return gpu::UNSUPPORTED_FEATURE~;
}

fn void? validate_graphics_pipeline_desc(
    VkDeviceState* state,
    gpu::GraphicsPipelineDesc* desc,
) {
    if (desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS
        || !gpu::internal::sample_count_valid(desc.sample_count)
        || !polygon_mode_valid(desc.polygon_mode)) return gpu::INVALID_ARGUMENT~;
    if (desc.depth_format != gpu::Format.UNDEFINED
        && desc.depth_format != gpu::Format.D32_FLOAT) return gpu::INVALID_ARGUMENT~;
    if (desc.colors.len == 0
        && desc.depth_format == gpu::Format.UNDEFINED) return gpu::INVALID_ARGUMENT~;
    gpu::TextureUsage color_usage = { .color_attach };
    foreach (&target : desc.colors) {
        if (!blend_state_valid(&target.blend)
            || !color_write_mask_valid(target.write_mask)
            || (target.blend.enable && color_format_is_integer(target.format))
            || target.format == gpu::Format.UNDEFINED
            || target.format == gpu::Format.D32_FLOAT
            || format_to_vk(target.format) == vk::Format.UNDEFINED) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (!format_supports_sample_count(
            state,
            target.format,
            color_usage,
            desc.sample_count,
        )!) return gpu::UNSUPPORTED_FEATURE~;
    }
    if (desc.depth_format != gpu::Format.UNDEFINED) {
        gpu::TextureUsage depth_usage = { .depth_attach };
        if (!format_supports_sample_count(
            state,
            desc.depth_format,
            depth_usage,
            desc.sample_count,
        )!) return gpu::UNSUPPORTED_FEATURE~;
    }
    check_polygon_mode_support(desc.polygon_mode, state.caps.line_polygon_mode)!;
}

fn void? validate_dynamic_graphics_pipeline_desc(
    VkDeviceState* state,
    gpu::DynamicGraphicsPipelineDesc* desc,
) {
    if (desc == null
        || desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS
        || (desc.colors.len > 0 && desc.colors.ptr == null)
        || !gpu::internal::sample_count_valid(desc.sample_count)
        || !polygon_mode_valid(desc.polygon_mode)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.depth_format != gpu::Format.UNDEFINED
        && desc.depth_format != gpu::Format.D32_FLOAT) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.colors.len == 0
        && desc.depth_format == gpu::Format.UNDEFINED) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::TextureUsage color_usage = { .color_attach };
    foreach (&target : desc.colors) {
        if (target.format == gpu::Format.UNDEFINED
            || target.format == gpu::Format.D32_FLOAT
            || format_to_vk(target.format) == vk::Format.UNDEFINED) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (!format_supports_sample_count(
            state,
            target.format,
            color_usage,
            desc.sample_count,
        )!) return gpu::UNSUPPORTED_FEATURE~;
    }
    if (desc.depth_format != gpu::Format.UNDEFINED) {
        gpu::TextureUsage depth_usage = { .depth_attach };
        if (!format_supports_sample_count(
            state,
            desc.depth_format,
            depth_usage,
            desc.sample_count,
        )!) return gpu::UNSUPPORTED_FEATURE~;
    }
    check_polygon_mode_support(desc.polygon_mode, state.caps.line_polygon_mode)!;
}

fn bool pipeline_key_is_cached(
    VkDeviceState* state,
    PipelineKey* key,
) {
    ulong hash = hash_key(key);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    return find_entry(state, hash, key)
        != PIPELINE_ENTRY_NONE;
}

fn vk::Result create_graphics_pipeline_native(
    VkDeviceState* state,
    vk::GraphicsPipelineCreateInfo* info,
    vk::Pipeline* out_pipeline,
) {
    $if env::TESTING:
        state.pipeline_native_dynamic_state_count =
            info.p_dynamic_state.dynamic_state_count;
        state.pipeline_native_color_target_count =
            info.p_color_blend_state.attachment_count;
        bool blend_enable = false;
        bool blend_equation = false;
        bool write_mask = false;
        for (uint i = 0;
            i < info.p_dynamic_state.dynamic_state_count;
            i++) {
            switch (info.p_dynamic_state.p_dynamic_states[i]) {
                case vk::DynamicState.COLOR_BLEND_ENABLE_EXT:
                    blend_enable = true;
                case vk::DynamicState.COLOR_BLEND_EQUATION_EXT:
                    blend_equation = true;
                case vk::DynamicState.COLOR_WRITE_MASK_EXT:
                    write_mask = true;
                default:
            }
        }
        state.pipeline_native_dynamic_color_states =
            blend_enable && blend_equation && write_mask;
        state.pipeline_native_canonical_color_attachments =
            info.p_color_blend_state.attachment_count > 0;
        for (uint i = 0;
            i < info.p_color_blend_state.attachment_count;
            i++) {
            vk::PipelineColorBlendAttachmentState* attachment =
                &info.p_color_blend_state.p_attachments[i];
            state.pipeline_native_canonical_color_attachments =
                state.pipeline_native_canonical_color_attachments
                && attachment.blend_enable == vk::FALSE
                && attachment.src_color_blend_factor == vk::BlendFactor.ONE
                && attachment.dst_color_blend_factor == vk::BlendFactor.ZERO
                && attachment.color_blend_op == vk::BlendOp.ADD
                && attachment.src_alpha_blend_factor == vk::BlendFactor.ONE
                && attachment.dst_alpha_blend_factor == vk::BlendFactor.ZERO
                && attachment.alpha_blend_op == vk::BlendOp.ADD
                && attachment.color_write_mask
                    == color_write_mask_to_vk(gpu::COLOR_WRITE_ALL);
        }
        if (state.pipeline_native_create_result != vk::Result.SUCCESS) {
            return state.pipeline_native_create_result;
        }
    $endif
    return vk::create_graphics_pipelines(
        device:            state.device,
        pipeline_cache:    state.driver_cache,
        create_info_count: 1,
        p_create_infos:    info,
        p_allocator:       null,
        p_pipelines:       out_pipeline,
    );
}

<* Create-info structs are assigned directly rather than through by-value
   builder setters (c3c 0.8.0 codegen crash). *>
fn gpu::PipelineHandle? create_graphics_pipeline_from_modules(
    gpu::Device* device,
    PipelineKey* requested_key,
    gpu::ShaderCode* vertex_code,
    gpu::ShaderCode* fragment_code,
    ZString operation,
    ZString vertex_field,
    ZString fragment_field,
    ZString debug_name,
    PreparedShaderIdentity* prepared_vertex_shader,
    PreparedShaderIdentity* prepared_fragment_shader,
    PipelineShaderModule* batch_vertex_shader,
    PipelineShaderModule* batch_fragment_shader,
) {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    PipelineKey key = *requested_key;
    ulong hash = hash_key(&key);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint hit = find_entry(state, hash, &key);
        if (hit != PIPELINE_ENTRY_NONE) {
            PipelineCacheEntry* entry = &state.pipeline_cache.entries[hit];
            entry.refcount++;
            if (try handle = state.pipelines.alloc({
                .pipeline = entry.pipeline,
                .layout   = entry.layout,
                .entry    = hit,
            })) {
                release_prepared_shader_locked(
                    state,
                    prepared_vertex_shader,
                )!!;
                release_prepared_shader_locked(
                    state,
                    prepared_fragment_shader,
                )!!;
                return handle;
            }
            entry.refcount--;
            release_prepared_shader_locked(state, prepared_vertex_shader)!!;
            release_prepared_shader_locked(state, prepared_fragment_shader)!!;
            return gpu::SLOT_TABLE_FULL~;
        }
    }

    PipelineShaderModule owned_vertex_shader;
    defer destroy_pipeline_shader_module(state, &owned_vertex_shader);
    PipelineShaderModule* vertex_shader = batch_vertex_shader;
    if (vertex_shader == null) {
        gpu::ShaderCode code = shader_store_code_for_retained_id(
            state,
            prepared_vertex_shader.id,
            vertex_code.debug_name,
        );
        owned_vertex_shader = create_pipeline_shader_module(
            state:          state,
            code:           &code,
            expected_stage: gpu::ShaderStage.VERTEX,
            operation:      operation,
            rejected_field: vertex_field,
        )!;
        vertex_shader = &owned_vertex_shader;
    }
    PipelineShaderModule owned_fragment_shader;
    defer destroy_pipeline_shader_module(state, &owned_fragment_shader);
    PipelineShaderModule* fragment_shader = batch_fragment_shader;
    if (fragment_shader == null) {
        gpu::ShaderCode code = shader_store_code_for_retained_id(
            state,
            prepared_fragment_shader.id,
            fragment_code.debug_name,
        );
        owned_fragment_shader = create_pipeline_shader_module(
            state:          state,
            code:           &code,
            expected_stage: gpu::ShaderStage.FRAGMENT,
            operation:      operation,
            rejected_field: fragment_field,
        )!;
        fragment_shader = &owned_fragment_shader;
    }
    @pool() {
        vk::Format[] color_formats = {};
        if (key.color_target_count > 0) {
            color_formats = mem::talloc_array(
                vk::Format,
                key.color_target_count,
            );
            foreach (i, &format : color_formats) {
                vk::Format vk_format = format_to_vk(
                    (gpu::Format)key.color_targets[i].format,
                );
                if (vk_format == vk::Format.UNDEFINED) return gpu::INVALID_ARGUMENT~;
                *format = vk_format;
            }
        }

        vk::PipelineShaderStageCreateInfo[2] stages;
        stages[0] = vk::pipeline_shader_stage_create_info()
            .set_stage(vk::ShaderStageFlagBits.VERTEX_BIT)
            .set_mod(vertex_shader.shader_module)
            .set_name(&vertex_shader.entry_point[0]);
        stages[1] = vk::pipeline_shader_stage_create_info()
            .set_stage(vk::ShaderStageFlagBits.FRAGMENT_BIT)
            .set_mod(fragment_shader.shader_module)
            .set_name(&fragment_shader.entry_point[0]);

        vk::PipelineRenderingCreateInfo rendering_info = vk::pipeline_rendering_create_info();
        rendering_info.color_attachment_count = (uint)color_formats.len;
        rendering_info.p_color_attachment_formats = color_formats.ptr;
        if (key.depth_format != (int)gpu::Format.UNDEFINED) {
            rendering_info.depth_attachment_format = format_to_vk(
                (gpu::Format)key.depth_format,
            );
        }

        vk::PipelineVertexInputStateCreateInfo vertex_input = vk::pipeline_vertex_input_state_create_info();

        vk::PipelineInputAssemblyStateCreateInfo input_assembly = vk::pipeline_input_assembly_state_create_info();
        input_assembly.topology = vk::PrimitiveTopology.TRIANGLE_LIST;

        vk::PipelineViewportStateCreateInfo viewport_state = vk::pipeline_viewport_state_create_info();
        viewport_state.viewport_count = 1;
        viewport_state.scissor_count = 1;

        vk::PipelineRasterizationStateCreateInfo raster = vk::pipeline_rasterization_state_create_info();
        raster.polygon_mode = key.polygon_mode == (int)gpu::PolygonMode.LINE
            ? vk::PolygonMode.LINE
            : vk::PolygonMode.FILL;
        raster.cull_mode  = vk::CullModeFlagBits.NONE;
        raster.front_face = vk::FrontFace.COUNTER_CLOCKWISE;
        raster.line_width = 1.0;

        vk::PipelineMultisampleStateCreateInfo multisample = vk::pipeline_multisample_state_create_info();
        multisample.rasterization_samples = sample_count_to_vk(
            (gpu::SampleCount)key.sample_count,
        );

        vk::PipelineDepthStencilStateCreateInfo depth_stencil = vk::pipeline_depth_stencil_state_create_info();

        vk::PipelineColorBlendStateCreateInfo blend_state = vk::pipeline_color_blend_state_create_info();
        if (color_formats.len > 0) {
            vk::PipelineColorBlendAttachmentState[] blend_attachments = mem::talloc_array(vk::PipelineColorBlendAttachmentState, color_formats.len);
            foreach (i, &attachment : blend_attachments) {
                ColorTargetKey* target = &key.color_targets[i];
                if (key.color_profile == (int)GraphicsColorProfile.DYNAMIC) {
                    *attachment = {
                        .blend_enable           = vk::FALSE,
                        .src_color_blend_factor = vk::BlendFactor.ONE,
                        .dst_color_blend_factor = vk::BlendFactor.ZERO,
                        .color_blend_op          = vk::BlendOp.ADD,
                        .src_alpha_blend_factor = vk::BlendFactor.ONE,
                        .dst_alpha_blend_factor = vk::BlendFactor.ZERO,
                        .alpha_blend_op          = vk::BlendOp.ADD,
                        .color_write_mask        = color_write_mask_to_vk(
                            gpu::COLOR_WRITE_ALL,
                        ),
                    };
                } else {
                    *attachment = {
                        .blend_enable           = target.blend_enable != 0 ? vk::TRUE : vk::FALSE,
                        .src_color_blend_factor = blend_factor_to_vk(
                            (gpu::BlendFactor)target.blend.src_color,
                        ),
                        .dst_color_blend_factor = blend_factor_to_vk(
                            (gpu::BlendFactor)target.blend.dst_color,
                        ),
                        .color_blend_op = blend_op_to_vk(
                            (gpu::BlendOp)target.blend.color_op,
                        ),
                        .src_alpha_blend_factor = blend_factor_to_vk(
                            (gpu::BlendFactor)target.blend.src_alpha,
                        ),
                        .dst_alpha_blend_factor = blend_factor_to_vk(
                            (gpu::BlendFactor)target.blend.dst_alpha,
                        ),
                        .alpha_blend_op = blend_op_to_vk(
                            (gpu::BlendOp)target.blend.alpha_op,
                        ),
                        .color_write_mask = color_write_mask_to_vk(
                            target.write_mask,
                        ),
                    };
                }
            }
            blend_state.attachment_count = (uint)blend_attachments.len;
            blend_state.p_attachments = blend_attachments.ptr;
        }

        vk::DynamicState[13] dynamic_states;
        dynamic_states[0] = vk::DynamicState.VIEWPORT;
        dynamic_states[1] = vk::DynamicState.SCISSOR;
        dynamic_states[2] = vk::DynamicState.PRIMITIVE_TOPOLOGY;
        dynamic_states[3] = vk::DynamicState.CULL_MODE;
        dynamic_states[4] = vk::DynamicState.FRONT_FACE;
        dynamic_states[5] = vk::DynamicState.DEPTH_BIAS_ENABLE;
        dynamic_states[6] = vk::DynamicState.DEPTH_BIAS;
        dynamic_states[7] = vk::DynamicState.DEPTH_TEST_ENABLE;
        dynamic_states[8] = vk::DynamicState.DEPTH_WRITE_ENABLE;
        dynamic_states[9] = vk::DynamicState.DEPTH_COMPARE_OP;
        usz dynamic_state_count = 10;
        if (key.color_profile == (int)GraphicsColorProfile.DYNAMIC) {
            dynamic_states[dynamic_state_count++] =
                vk::DynamicState.COLOR_BLEND_ENABLE_EXT;
            dynamic_states[dynamic_state_count++] =
                vk::DynamicState.COLOR_BLEND_EQUATION_EXT;
            dynamic_states[dynamic_state_count++] =
                vk::DynamicState.COLOR_WRITE_MASK_EXT;
        }
        vk::PipelineDynamicStateCreateInfo dynamic_state = vk::pipeline_dynamic_state_create_info();
        dynamic_state.dynamic_state_count = (uint)dynamic_state_count;
        dynamic_state.p_dynamic_states = &dynamic_states[0];

        vk::GraphicsPipelineCreateInfo pipe_info = vk::graphics_pipeline_create_info();
        pipe_info.p_next = &rendering_info;
        pipe_info.stage_count = stages.len;
        pipe_info.p_stages = &stages[0];
        pipe_info.p_vertex_input_state = &vertex_input;
        pipe_info.p_input_assembly_state = &input_assembly;
        pipe_info.p_viewport_state = &viewport_state;
        pipe_info.p_rasterization_state = &raster;
        pipe_info.p_multisample_state = &multisample;
        pipe_info.p_depth_stencil_state = &depth_stencil;
        pipe_info.p_color_blend_state = &blend_state;
        pipe_info.p_dynamic_state = &dynamic_state;
        pipe_info.layout = state.graphics_layout;
        if (strict_heap_uses_descriptor_buffer(&state.strict_heap)) {
            pipe_info.flags = vk::PipelineCreateFlagBits.DESCRIPTOR_BUFFER_BIT_EXT;
        }

        vk::Pipeline pipeline;
        check_pipeline_create_device_result(
            state: state,
            result: create_graphics_pipeline_native(
                state,
                &pipe_info,
                &pipeline,
            ),
        )!;
        state.pipelines_created++;
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.PIPELINE,
            object_handle: (ulong)pipeline,
            name:          debug_name,
        );

        // Compiled outside the lock; a same-key racer may have inserted
        // meanwhile — the loser destroys its pipeline and refs the winner.
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint entry_index;
        uint raced = find_entry(state, hash, &key);
        if (raced != PIPELINE_ENTRY_NONE) {
            vk::destroy_pipeline(state.device, pipeline, null);
            state.pipeline_cache.entries[raced].refcount++;
            release_prepared_shader_locked(
                state,
                prepared_vertex_shader,
            )!!;
            release_prepared_shader_locked(
                state,
                prepared_fragment_shader,
            )!!;
            entry_index = raced;
            pipeline = state.pipeline_cache.entries[raced].pipeline;
        } else if (try idx = insert_entry(
            state: state,
            value: {
                .hash     = hash,
                .key      = key,
                .pipeline = pipeline,
                .layout   = state.graphics_layout,
            },
        )) {
            transfer_prepared_shader(prepared_vertex_shader);
            transfer_prepared_shader(prepared_fragment_shader);
            entry_index = idx;
        } else {
            vk::destroy_pipeline(state.device, pipeline, null);
            return gpu::SLOT_TABLE_FULL~;
        }

        if (try handle = state.pipelines.alloc({
            .pipeline = pipeline,
            .layout   = state.graphics_layout,
            .entry    = entry_index,
        })) {
            return handle;
        }
        release_entry(state, entry_index);
        return gpu::SLOT_TABLE_FULL~;
    };
}

fn gpu::PipelineHandle? vk_create_dynamic_graphics_pipeline(
    gpu::Device* device,
    gpu::DynamicGraphicsPipelineDesc* desc,
) {
    VkDeviceState* state =
        (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    if (!state.caps.dynamic_color_state) return gpu::UNSUPPORTED_FEATURE~;
    validate_dynamic_graphics_pipeline_desc(state, desc)!;
    validate_pipeline_shader_code(
        state:          state,
        code:           &desc.vertex_shader,
        operation:      "create_dynamic_graphics_pipeline",
        rejected_field: "DynamicGraphicsPipelineDesc.vertex_shader",
    )!;
    validate_pipeline_shader_code(
        state:          state,
        code:           &desc.fragment_shader,
        operation:      "create_dynamic_graphics_pipeline",
        rejected_field: "DynamicGraphicsPipelineDesc.fragment_shader",
    )!;
    if (!pipeline_handle_capacity_available(state, 1)) {
        return gpu::SLOT_TABLE_FULL~;
    }
    PreparedShaderIdentity[2] prepared;
    usz prepared_count = 0;
    defer release_prepared_shaders(state, prepared[:prepared_count]);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        prepared[0] = prepare_shader_identity_locked(
            state,
            &desc.vertex_shader,
        )!;
        prepared_count++;
        prepared[1] = prepare_shader_identity_locked(
            state,
            &desc.fragment_shader,
        )!;
        prepared_count++;
    }
    PipelineKey key = build_dynamic_graphics_key(
        desc,
        prepared[0].id,
        prepared[1].id,
    );
    return create_graphics_pipeline_from_modules(
        device:                   device,
        requested_key:            &key,
        vertex_code:              &desc.vertex_shader,
        fragment_code:            &desc.fragment_shader,
        operation:                "create_dynamic_graphics_pipeline",
        vertex_field:             "DynamicGraphicsPipelineDesc.vertex_shader",
        fragment_field:           "DynamicGraphicsPipelineDesc.fragment_shader",
        debug_name:               desc.debug_name,
        prepared_vertex_shader:   &prepared[0],
        prepared_fragment_shader: &prepared[1],
        batch_vertex_shader:      null,
        batch_fragment_shader:    null,
    );
}

fn void? vk_create_dynamic_graphics_pipelines(
    gpu::Device* device,
    gpu::DynamicGraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    if (out_pipelines.len != descs.len) return gpu::INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    VkDeviceState* state =
        (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    if (!state.caps.dynamic_color_state) return gpu::UNSUPPORTED_FEATURE~;

    foreach (&desc : descs) {
        validate_dynamic_graphics_pipeline_desc(state, desc)!;
        validate_pipeline_shader_code(
            state:          state,
            code:           &desc.vertex_shader,
            operation:      "create_dynamic_graphics_pipelines",
            rejected_field: "DynamicGraphicsPipelineDesc.vertex_shader",
        )!;
        validate_pipeline_shader_code(
            state:          state,
            code:           &desc.fragment_shader,
            operation:      "create_dynamic_graphics_pipelines",
            rejected_field: "DynamicGraphicsPipelineDesc.fragment_shader",
        )!;
    }
    if (!pipeline_handle_capacity_available(state, descs.len)) {
        return gpu::SLOT_TABLE_FULL~;
    }

    @pool() {
        PreparedShaderIdentity[] prepared_vertex =
            mem::talloc_array(PreparedShaderIdentity, descs.len);
        PreparedShaderIdentity[] prepared_fragment =
            mem::talloc_array(PreparedShaderIdentity, descs.len);
        $if env::TESTING:
            poison_unprepared_shader_identities(state, prepared_vertex);
            poison_unprepared_shader_identities(state, prepared_fragment);
        $endif
        usz prepared_vertex_count = 0;
        usz prepared_fragment_count = 0;
        defer release_prepared_shaders(
            state,
            prepared_fragment[:prepared_fragment_count],
        );
        defer release_prepared_shaders(
            state,
            prepared_vertex[:prepared_vertex_count],
        );
        {
            state.resource_mutex.lock()!!;
            defer state.resource_mutex.unlock();
            foreach (i, &desc : descs) {
                prepared_vertex[i] = prepare_shader_identity_locked(
                    state,
                    &desc.vertex_shader,
                )!;
                prepared_vertex_count++;
                prepared_fragment[i] = prepare_shader_identity_locked(
                    state,
                    &desc.fragment_shader,
                )!;
                prepared_fragment_count++;
            }
        }

        gpu::PipelineHandle[] created =
            mem::talloc_array(gpu::PipelineHandle, descs.len);
        usz created_count = 0;
        foreach (i, &desc : descs) {
            if (pipeline_creation_should_fail(state, (usz)i)) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return gpu::PIPELINE_CREATE_FAILED~;
            }
            PipelineKey key = build_dynamic_graphics_key(
                desc,
                prepared_vertex[i].id,
                prepared_fragment[i].id,
            );
            gpu::PipelineHandle? created_optional =
                create_graphics_pipeline_from_modules(
                    device:                   device,
                    requested_key:            &key,
                    vertex_code:              &desc.vertex_shader,
                    fragment_code:            &desc.fragment_shader,
                    operation:                "create_dynamic_graphics_pipelines",
                    vertex_field:             "DynamicGraphicsPipelineDesc.vertex_shader",
                    fragment_field:           "DynamicGraphicsPipelineDesc.fragment_shader",
                    debug_name:               desc.debug_name,
                    prepared_vertex_shader:   &prepared_vertex[i],
                    prepared_fragment_shader: &prepared_fragment[i],
                    batch_vertex_shader:      null,
                    batch_fragment_shader:    null,
                );
            if (catch create_fault = created_optional) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return create_fault~;
            }
            created[i] = created_optional;
            created_count++;
        }
        foreach (i, pipeline : created) out_pipelines[i] = pipeline;
    };
}

fn gpu::PipelineHandle? vk_create_graphics_pipeline(
    gpu::Device* device,
    gpu::GraphicsPipelineDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    validate_graphics_pipeline_desc(state, desc)!;
    validate_pipeline_shader_code(
        state:          state,
        code:           &desc.vertex_shader,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.vertex_shader",
    )!;
    validate_pipeline_shader_code(
        state:          state,
        code:           &desc.fragment_shader,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.fragment_shader",
    )!;
    if (!pipeline_handle_capacity_available(state, 1)) {
        return gpu::SLOT_TABLE_FULL~;
    }
    PreparedShaderIdentity[2] prepared;
    usz prepared_count = 0;
    defer release_prepared_shaders(state, prepared[:prepared_count]);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        prepared[0] = prepare_shader_identity_locked(
            state,
            &desc.vertex_shader,
        )!;
        prepared_count++;
        prepared[1] = prepare_shader_identity_locked(
            state,
            &desc.fragment_shader,
        )!;
        prepared_count++;
    }
    PipelineKey key = build_graphics_key(
        desc,
        prepared[0].id,
        prepared[1].id,
    );
    return create_graphics_pipeline_from_modules(
        device:                   device,
        requested_key:            &key,
        vertex_code:              &desc.vertex_shader,
        fragment_code:            &desc.fragment_shader,
        operation:                "create_graphics_pipeline",
        vertex_field:             "GraphicsPipelineDesc.vertex_shader",
        fragment_field:           "GraphicsPipelineDesc.fragment_shader",
        debug_name:               desc.debug_name,
        prepared_vertex_shader:   &prepared[0],
        prepared_fragment_shader: &prepared[1],
        batch_vertex_shader:      null,
        batch_fragment_shader:    null,
    );
}

fn void? vk_create_graphics_pipelines(
    gpu::Device* device,
    gpu::GraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) @private {
    if (out_pipelines.len != descs.len) return gpu::INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();

    foreach (&desc : descs) {
        validate_graphics_pipeline_desc(state, desc)!;
    }
    if (!pipeline_handle_capacity_available(state, descs.len)) {
        return gpu::SLOT_TABLE_FULL~;
    }

    @pool() {
        PreparedShaderIdentity[] prepared_vertex =
            mem::talloc_array(PreparedShaderIdentity, descs.len);
        PreparedShaderIdentity[] prepared_fragment =
            mem::talloc_array(PreparedShaderIdentity, descs.len);
        $if env::TESTING:
            poison_unprepared_shader_identities(state, prepared_vertex);
            poison_unprepared_shader_identities(state, prepared_fragment);
        $endif
        usz prepared_vertex_count = 0;
        usz prepared_fragment_count = 0;
        defer release_prepared_shaders(
            state,
            prepared_fragment[:prepared_fragment_count],
        );
        defer release_prepared_shaders(
            state,
            prepared_vertex[:prepared_vertex_count],
        );
        foreach (i, &desc : descs) {
            validate_pipeline_shader_code(
                state:          state,
                code:           &desc.vertex_shader,
                operation:      "create_graphics_pipelines",
                rejected_field: "GraphicsPipelineDesc.vertex_shader",
            )!;
            {
                state.resource_mutex.lock()!!;
                defer state.resource_mutex.unlock();
                prepared_vertex[i] = prepare_shader_identity_locked(
                    state,
                    &desc.vertex_shader,
                )!;
            }
            prepared_vertex_count++;
            validate_pipeline_shader_code(
                state:          state,
                code:           &desc.fragment_shader,
                operation:      "create_graphics_pipelines",
                rejected_field: "GraphicsPipelineDesc.fragment_shader",
            )!;
            {
                state.resource_mutex.lock()!!;
                defer state.resource_mutex.unlock();
                prepared_fragment[i] = prepare_shader_identity_locked(
                    state,
                    &desc.fragment_shader,
                )!;
            }
            prepared_fragment_count++;
        }

        PipelineBatchShader[] shaders =
            mem::talloc_array(PipelineBatchShader, descs.len * 2);
        PipelineShaderModule*[] vertex_shaders =
            mem::talloc_array(PipelineShaderModule*, descs.len);
        PipelineShaderModule*[] fragment_shaders =
            mem::talloc_array(PipelineShaderModule*, descs.len);
        uint shader_count = 0;
        defer destroy_pipeline_batch_shaders(state, shaders, shader_count);
        foreach (i, &desc : descs) {
            vertex_shaders[i] = null;
            fragment_shaders[i] = null;
            PipelineKey key = build_graphics_key(
                desc,
                prepared_vertex[i].id,
                prepared_fragment[i].id,
            );
            if (pipeline_key_is_cached(state, &key)) continue;
            vertex_shaders[i] = get_pipeline_batch_shader(
                state:          state,
                id:             prepared_vertex[i].id,
                debug_name:     desc.vertex_shader.debug_name,
                expected_stage: gpu::ShaderStage.VERTEX,
                operation:      "create_graphics_pipelines",
                rejected_field: "GraphicsPipelineDesc.vertex_shader",
                shaders:        shaders,
                shader_count:   &shader_count,
            )!;
            fragment_shaders[i] = get_pipeline_batch_shader(
                state:          state,
                id:             prepared_fragment[i].id,
                debug_name:     desc.fragment_shader.debug_name,
                expected_stage: gpu::ShaderStage.FRAGMENT,
                operation:      "create_graphics_pipelines",
                rejected_field: "GraphicsPipelineDesc.fragment_shader",
                shaders:        shaders,
                shader_count:   &shader_count,
            )!;
        }

        gpu::PipelineHandle[] created = mem::talloc_array(gpu::PipelineHandle, descs.len);
        usz created_count = 0;
        foreach (i, &desc : descs) {
            if (pipeline_creation_should_fail(state, (usz)i)) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return gpu::PIPELINE_CREATE_FAILED~;
            }
            PipelineKey key = build_graphics_key(
                desc,
                prepared_vertex[i].id,
                prepared_fragment[i].id,
            );
            gpu::PipelineHandle? created_optional = create_graphics_pipeline_from_modules(
                device:                   device,
                requested_key:            &key,
                vertex_code:              &desc.vertex_shader,
                fragment_code:            &desc.fragment_shader,
                operation:                "create_graphics_pipelines",
                vertex_field:             "GraphicsPipelineDesc.vertex_shader",
                fragment_field:           "GraphicsPipelineDesc.fragment_shader",
                debug_name:               desc.debug_name,
                prepared_vertex_shader:   &prepared_vertex[i],
                prepared_fragment_shader: &prepared_fragment[i],
                batch_vertex_shader:      vertex_shaders[i],
                batch_fragment_shader:    fragment_shaders[i],
            );
            if (catch create_fault = created_optional) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return create_fault~;
            }
            created[i] = created_optional;
            created_count++;
        }
        foreach (i, pipeline : created) out_pipelines[i] = pipeline;
    };
}
