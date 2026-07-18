module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

const uint GRAPHICS_ROOT_PUSH_SIZE = gpu::GraphicsRootPush::size;

const vk::ColorComponentFlags COLOR_WRITE_ALL = (vk::ColorComponentFlags)(
    (uint)vk::ColorComponentFlagBits.R_BIT
    | (uint)vk::ColorComponentFlagBits.G_BIT
    | (uint)vk::ColorComponentFlagBits.B_BIT
    | (uint)vk::ColorComponentFlagBits.A_BIT);

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

fn void? check_polygon_mode_support(gpu::PolygonMode mode, bool line_supported) {
    if (mode == gpu::PolygonMode.LINE && !line_supported) return gpu::UNSUPPORTED_FEATURE~;
}

<* Create-info structs are assigned directly rather than through by-value
   builder setters (c3c 0.8.0 codegen crash). *>
fn gpu::PipelineHandle? vk_create_graphics_pipeline(gpu::Device* device, gpu::GraphicsPipelineDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    if (desc.color_formats.len > gpu::MAX_COLOR_ATTACHMENTS) return gpu::INVALID_ARGUMENT~;
    if (desc.depth_format != gpu::Format.UNDEFINED && desc.depth_format != gpu::Format.D32_FLOAT) return gpu::INVALID_ARGUMENT~;
    if (desc.color_formats.len == 0 && desc.depth_format == gpu::Format.UNDEFINED) return gpu::INVALID_ARGUMENT~;
    check_polygon_mode_support(desc.raster.polygon_mode, state.caps.line_polygon_mode)!;
    ShaderPipelineReference vertex_reference =
        retain_shader_for_pipeline(state, desc.vertex_shader)!;
    defer release_shader_for_pipeline(state, vertex_reference);
    ShaderSlot* vertex_shader = vertex_reference.slot;
    ShaderPipelineReference fragment_reference =
        retain_shader_for_pipeline(state, desc.fragment_shader)!;
    defer release_shader_for_pipeline(state, fragment_reference);
    ShaderSlot* fragment_shader = fragment_reference.slot;
    validate_shader_stage(
        state:          state,
        slot:           vertex_shader,
        expected:       gpu::ShaderStage.VERTEX,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.vertex_shader",
        handle:         desc.vertex_shader,
    )!;
    validate_shader_stage(
        state:          state,
        slot:           fragment_shader,
        expected:       gpu::ShaderStage.FRAGMENT,
        operation:      "create_graphics_pipeline",
        rejected_field: "GraphicsPipelineDesc.fragment_shader",
        handle:         desc.fragment_shader,
    )!;

    PipelineKey key = build_graphics_key(desc);
    ulong hash = hash_key(&key);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint hit = find_entry(&state.pipeline_cache, hash, &key);
        if (hit != PIPELINE_ENTRY_NONE) {
            PipelineCacheEntry* entry = &state.pipeline_cache.entries[hit];
            entry.refcount++;
            if (try handle = state.pipelines.alloc({
                .pipeline = entry.pipeline,
                .layout   = entry.layout,
                .entry    = hit,
                .raster   = desc.raster,
                .depth    = desc.depth,
            })) {
                return handle;
            }
            entry.refcount--;
            return gpu::SLOT_TABLE_FULL~;
        }
    }

    @pool() {
        vk::Format[] color_formats = {};
        if (desc.color_formats.len > 0) {
            color_formats = mem::talloc_array(vk::Format, desc.color_formats.len);
            foreach (i, format : desc.color_formats) {
                vk::Format vk_format = format_to_vk(format);
                if (vk_format == vk::Format.UNDEFINED) return gpu::INVALID_ARGUMENT~;
                color_formats[i] = vk_format;
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
        if (desc.depth_format != gpu::Format.UNDEFINED) {
            rendering_info.depth_attachment_format = format_to_vk(desc.depth_format);
        }

        vk::PipelineVertexInputStateCreateInfo vertex_input = vk::pipeline_vertex_input_state_create_info();

        vk::PipelineInputAssemblyStateCreateInfo input_assembly = vk::pipeline_input_assembly_state_create_info();
        input_assembly.topology = topology_to_vk(desc.topology);

        vk::PipelineViewportStateCreateInfo viewport_state = vk::pipeline_viewport_state_create_info();
        viewport_state.viewport_count = 1;
        viewport_state.scissor_count = 1;

        vk::PipelineRasterizationStateCreateInfo raster = vk::pipeline_rasterization_state_create_info();
        raster.polygon_mode = desc.raster.polygon_mode == gpu::PolygonMode.LINE ? vk::PolygonMode.LINE : vk::PolygonMode.FILL;
        raster.line_width   = 1.0;
        if (desc.raster.depth_bias_constant != 0 || desc.raster.depth_bias_slope != 0 || desc.raster.depth_bias_clamp != 0) {
            raster.depth_bias_enable          = vk::TRUE;
            raster.depth_bias_constant_factor = desc.raster.depth_bias_constant;
            raster.depth_bias_slope_factor    = desc.raster.depth_bias_slope;
            raster.depth_bias_clamp           = desc.raster.depth_bias_clamp;
        }

        vk::PipelineMultisampleStateCreateInfo multisample = vk::pipeline_multisample_state_create_info();
        multisample.rasterization_samples = vk::SampleCountFlagBits.COUNT_1_BIT;

        vk::PipelineDepthStencilStateCreateInfo depth_stencil = vk::pipeline_depth_stencil_state_create_info();

        vk::PipelineColorBlendStateCreateInfo blend_state = vk::pipeline_color_blend_state_create_info();
        if (color_formats.len > 0) {
            vk::PipelineColorBlendAttachmentState[] blend_attachments = mem::talloc_array(vk::PipelineColorBlendAttachmentState, color_formats.len);
            foreach (&attachment : blend_attachments) {
                *attachment = {
                    .blend_enable           = desc.blend.enable ? vk::TRUE : vk::FALSE,
                    .src_color_blend_factor = blend_factor_to_vk(desc.blend.src_color),
                    .dst_color_blend_factor = blend_factor_to_vk(desc.blend.dst_color),
                    .color_blend_op         = blend_op_to_vk(desc.blend.color_op),
                    .src_alpha_blend_factor = blend_factor_to_vk(desc.blend.src_alpha),
                    .dst_alpha_blend_factor = blend_factor_to_vk(desc.blend.dst_alpha),
                    .alpha_blend_op         = blend_op_to_vk(desc.blend.alpha_op),
                    .color_write_mask       = COLOR_WRITE_ALL,
                };
            }
            blend_state.attachment_count = (uint)blend_attachments.len;
            blend_state.p_attachments = blend_attachments.ptr;
        }

        vk::DynamicState[7] dynamic_states = {
            vk::DynamicState.VIEWPORT,
            vk::DynamicState.SCISSOR,
            vk::DynamicState.CULL_MODE,
            vk::DynamicState.FRONT_FACE,
            vk::DynamicState.DEPTH_TEST_ENABLE,
            vk::DynamicState.DEPTH_WRITE_ENABLE,
            vk::DynamicState.DEPTH_COMPARE_OP,
        };
        vk::PipelineDynamicStateCreateInfo dynamic_state = vk::pipeline_dynamic_state_create_info();
        dynamic_state.dynamic_state_count = dynamic_states.len;
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
            result: vk::create_graphics_pipelines(
                device:            state.device,
                pipeline_cache:    state.driver_cache,
                create_info_count: 1,
                p_create_infos:    &pipe_info,
                p_allocator:       null,
                p_pipelines:       &pipeline,
            ),
        )!;
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.PIPELINE,
            object_handle: (ulong)pipeline,
            name:          desc.debug_name,
        );

        // Compiled outside the lock; a same-key racer may have inserted
        // meanwhile — the loser destroys its pipeline and refs the winner.
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint entry_index;
        uint raced = find_entry(&state.pipeline_cache, hash, &key);
        if (raced != PIPELINE_ENTRY_NONE) {
            vk::destroy_pipeline(state.device, pipeline, null);
            state.pipeline_cache.entries[raced].refcount++;
            entry_index = raced;
            pipeline = state.pipeline_cache.entries[raced].pipeline;
        } else if (try idx = insert_entry(&state.pipeline_cache, {
            .hash     = hash,
            .key      = key,
            .pipeline = pipeline,
            .layout   = state.graphics_layout,
        })) {
            entry_index = idx;
        } else {
            vk::destroy_pipeline(state.device, pipeline, null);
            return gpu::SLOT_TABLE_FULL~;
        }

        if (try handle = state.pipelines.alloc({
            .pipeline = pipeline,
            .layout   = state.graphics_layout,
            .entry    = entry_index,
            .raster   = desc.raster,
            .depth    = desc.depth,
        })) {
            return handle;
        }
        release_entry(state, entry_index);
        return gpu::SLOT_TABLE_FULL~;
    };
}
