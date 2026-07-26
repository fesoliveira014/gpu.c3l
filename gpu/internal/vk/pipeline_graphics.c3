module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::thread;
import std::core::mem;

alias GraphicsPipelineCreateFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::GraphicsPipelineCreateInfo* create_info,
    vk::Pipeline* pipeline,
);

const uint GRAPHICS_ROOT_PUSH_SIZE = gpu::GraphicsRootPush::size;

struct GraphicsPipelineCreateOps @private {
    ShaderModuleCreateOps    shader_module;
    void*                    user_data;
    GraphicsPipelineCreateFn create;
}

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
    if (desc == null
        || desc.color_formats.len > gpu::MAX_COLOR_ATTACHMENTS
        || (desc.color_formats.len > 0 && desc.color_formats.ptr == null)
        || !gpu::internal::sample_count_valid(desc.sample_count)
        || !polygon_mode_valid(desc.polygon_mode)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.depth_format != gpu::Format.UNDEFINED
        && desc.depth_format != gpu::Format.D32_FLOAT) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.color_formats.len == 0
        && desc.depth_format == gpu::Format.UNDEFINED) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::TextureUsage color_usage = { .color_attach };
    foreach (format : desc.color_formats) {
        if (format == gpu::Format.UNDEFINED
            || format == gpu::Format.D32_FLOAT
            || format_to_vk(format) == vk::Format.UNDEFINED) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (!format_supports_sample_count(
            state,
            format,
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

fn vk::Result create_graphics_pipeline_real(
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::GraphicsPipelineCreateInfo* info,
    vk::Pipeline* out_pipeline,
) @private {
    (void)user_data;
    return vk::create_graphics_pipelines(
        device:            device,
        pipeline_cache:    pipeline_cache,
        create_info_count: 1,
        p_create_infos:    info,
        p_allocator:       null,
        p_pipelines:       out_pipeline,
    );
}

fn GraphicsPipelineCreateOps graphics_pipeline_create_ops_real() @private {
    return {
        .shader_module = shader_module_create_ops_real(),
        .create        = &create_graphics_pipeline_real,
    };
}

fn vk::Result create_graphics_pipeline_native(
    VkDeviceState* state,
    vk::GraphicsPipelineCreateInfo* info,
    vk::Pipeline* out_pipeline,
    GraphicsPipelineCreateOps* create_ops,
) {
    return create_ops.create(
        user_data:      create_ops.user_data,
        device:         state.device,
        pipeline_cache: state.driver_cache,
        create_info:    info,
        pipeline:       out_pipeline,
    );
}

<* Create-info structs are assigned directly rather than through by-value
   builder setters (c3c 0.8.0 codegen crash). *>
fn gpu::PipelineHandle? create_graphics_pipeline_from_modules(
    VkDeviceState* state,
    PipelineKey* requested_key,
    gpu::ShaderDesc* vertex_desc,
    gpu::ShaderDesc* fragment_desc,
    ZString operation,
    ZString vertex_field,
    ZString fragment_field,
    ZString debug_name,
    PreparedShaderIdentity* prepared_vertex_shader,
    PreparedShaderIdentity* prepared_fragment_shader,
    GraphicsPipelineCreateOps* create_ops,
) {
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

    ShaderIdentityInput vertex_code;
    shader_store_input_for_retained_id(
        state:      state,
        id:         prepared_vertex_shader.id,
        debug_name: vertex_desc.debug_name,
        out_input:  &vertex_code,
    );
    PipelineShaderModule vertex_shader = create_pipeline_shader_module_with_ops(
        state:          state,
        code:           &vertex_code,
        operation:      operation,
        rejected_field: vertex_field,
        create_ops:      &create_ops.shader_module,
    )!;
    defer destroy_pipeline_shader_module(state, &vertex_shader);
    ShaderIdentityInput fragment_code;
    shader_store_input_for_retained_id(
        state:      state,
        id:         prepared_fragment_shader.id,
        debug_name: fragment_desc.debug_name,
        out_input:  &fragment_code,
    );
    PipelineShaderModule fragment_shader = create_pipeline_shader_module_with_ops(
        state:          state,
        code:           &fragment_code,
        operation:      operation,
        rejected_field: fragment_field,
        create_ops:      &create_ops.shader_module,
    )!;
    defer destroy_pipeline_shader_module(state, &fragment_shader);
    @pool() {
        vk::Format[] color_formats = {};
        if (key.color_target_count > 0) {
            color_formats = mem::talloc_array(
                vk::Format,
                key.color_target_count,
            );
            foreach (i, &format : color_formats) {
                vk::Format vk_format = format_to_vk(
                    key.color_formats[i],
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
        if (key.depth_format != gpu::Format.UNDEFINED) {
            rendering_info.depth_attachment_format = format_to_vk(
                key.depth_format,
            );
        }

        vk::PipelineVertexInputStateCreateInfo vertex_input = vk::pipeline_vertex_input_state_create_info();

        vk::PipelineInputAssemblyStateCreateInfo input_assembly = vk::pipeline_input_assembly_state_create_info();
        input_assembly.topology = vk::PrimitiveTopology.TRIANGLE_LIST;

        vk::PipelineViewportStateCreateInfo viewport_state = vk::pipeline_viewport_state_create_info();
        viewport_state.viewport_count = 1;
        viewport_state.scissor_count = 1;

        vk::PipelineRasterizationStateCreateInfo raster = vk::pipeline_rasterization_state_create_info();
        raster.polygon_mode = key.polygon_mode == gpu::PolygonMode.LINE
            ? vk::PolygonMode.LINE
            : vk::PolygonMode.FILL;
        raster.cull_mode  = vk::CullModeFlagBits.NONE;
        raster.front_face = vk::FrontFace.COUNTER_CLOCKWISE;
        raster.line_width = 1.0;

        vk::PipelineMultisampleStateCreateInfo multisample = vk::pipeline_multisample_state_create_info();
        multisample.rasterization_samples = sample_count_to_vk(
            key.sample_count,
        );

        vk::PipelineDepthStencilStateCreateInfo depth_stencil = vk::pipeline_depth_stencil_state_create_info();

        vk::PipelineColorBlendStateCreateInfo blend_state = vk::pipeline_color_blend_state_create_info();
        if (color_formats.len > 0) {
            vk::PipelineColorBlendAttachmentState[] blend_attachments = mem::talloc_array(vk::PipelineColorBlendAttachmentState, color_formats.len);
            foreach (&attachment : blend_attachments) {
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
        dynamic_states[10] = vk::DynamicState.COLOR_BLEND_ENABLE_EXT;
        dynamic_states[11] = vk::DynamicState.COLOR_BLEND_EQUATION_EXT;
        dynamic_states[12] = vk::DynamicState.COLOR_WRITE_MASK_EXT;
        vk::PipelineDynamicStateCreateInfo dynamic_state = vk::pipeline_dynamic_state_create_info();
        dynamic_state.dynamic_state_count = (uint)dynamic_states.len;
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

        vk::Pipeline pipeline;
        check_pipeline_create_device_result(
            state: state,
            result: create_graphics_pipeline_native(
                state:        state,
                info:         &pipe_info,
                out_pipeline: &pipeline,
                create_ops:   create_ops,
            ),
        )!;
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

fn gpu::PipelineHandle? vk_create_graphics_pipeline_with_ops(
    VkDeviceState* state,
    gpu::GraphicsPipelineDesc* desc,
    GraphicsPipelineCreateOps* create_ops,
) {
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    validate_graphics_pipeline_desc(state, desc)!;
    ShaderIdentityInput[2] shader_inputs;
    normalize_shader_input(
        state:          state,
        desc:           &desc.vertex_shader,
        role:           gpu::internal::ShaderRole.VERTEX,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.vertex_shader",
        out_input:      &shader_inputs[0],
    )!;
    normalize_shader_input(
        state:          state,
        desc:           &desc.fragment_shader,
        role:           gpu::internal::ShaderRole.FRAGMENT,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.fragment_shader",
        out_input:      &shader_inputs[1],
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
            &shader_inputs[0],
        )!;
        prepared_count++;
        prepared[1] = prepare_shader_identity_locked(
            state,
            &shader_inputs[1],
        )!;
        prepared_count++;
    }
    PipelineKey key = build_graphics_key(
        desc,
        prepared[0].id,
        prepared[1].id,
    );
    return create_graphics_pipeline_from_modules(
        state:                    state,
        requested_key:            &key,
        vertex_desc:              &desc.vertex_shader,
        fragment_desc:            &desc.fragment_shader,
        operation:                "create_graphics_pipeline",
        vertex_field:             "GraphicsPipelineDesc.vertex_shader",
        fragment_field:           "GraphicsPipelineDesc.fragment_shader",
        debug_name:               desc.debug_name,
        prepared_vertex_shader:   &prepared[0],
        prepared_fragment_shader: &prepared[1],
        create_ops:               create_ops,
    );
}

fn gpu::PipelineHandle? vk_create_graphics_pipeline(
    VkDeviceState* state,
    gpu::GraphicsPipelineDesc* desc,
) {
    GraphicsPipelineCreateOps create_ops =
        graphics_pipeline_create_ops_real();
    return vk_create_graphics_pipeline_with_ops(
        state:      state,
        desc:       desc,
        create_ops: &create_ops,
    );
}
