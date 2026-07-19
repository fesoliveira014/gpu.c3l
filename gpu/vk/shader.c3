module gpu::vk @private;

import gpu @public;
import vk;
import spvreflect;
import std::core::mem;

struct PipelineShaderModule {
    vk::ShaderModule shader_module;
    char[gpu::SHADER_ENTRY_POINT_CAP] entry_point;
}

struct PipelineBatchShader {
    gpu::ShaderCode      code;
    gpu::ShaderStage     expected_stage;
    PipelineShaderModule native;
}

<* Fault SHADER_INVALID on any set other than 0 or a set-0 binding that breaks
   the heap convention; buffers-only shaders pass. *>
fn void? check_heap_convention(spvreflect::ShaderModule* reflect_module) @private {
    uint count = 0;
    if (reflect_module.enumerate_descriptor_bindings(&count, null) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::DescriptorBinding*[] bindings =
            mem::talloc_array(spvreflect::DescriptorBinding*, (sz)count);
        if (reflect_module.enumerate_descriptor_bindings(&count, bindings.ptr)
            != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (binding : bindings) {
            if (binding.set_index() != HEAP_SET) return gpu::SHADER_INVALID~;
            int descriptor_type = (int)binding.descriptor_type();
            switch (binding.binding_index()) {
                case HEAP_BINDING_SAMPLED:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLED_IMAGE) {
                        return gpu::SHADER_INVALID~;
                    }
                case HEAP_BINDING_STORAGE:
                    if (descriptor_type != (int)vk::DescriptorType.STORAGE_IMAGE) {
                        return gpu::SHADER_INVALID~;
                    }
                case HEAP_BINDING_SAMPLERS:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLER) {
                        return gpu::SHADER_INVALID~;
                    }
                default:
                    return gpu::SHADER_INVALID~;
            }
        }
    };
}

fn gpu::ShaderStage? gpu_stage_from_spv(spvreflect::ShaderStage stage) @private {
    switch (stage) {
        case spvreflect::ShaderStage.SHADER_STAGE_COMPUTE:
            return gpu::ShaderStage.COMPUTE;
        case spvreflect::ShaderStage.SHADER_STAGE_VERTEX:
            return gpu::ShaderStage.VERTEX;
        case spvreflect::ShaderStage.SHADER_STAGE_FRAGMENT:
            return gpu::ShaderStage.FRAGMENT;
        default:
            return gpu::INVALID_ARGUMENT~;
    }
}

fn void? check_push_constant_range(
    spvreflect::ShaderModule* reflect_module,
    gpu::ShaderStage stage,
) @private {
    usz expected_size = stage == gpu::ShaderStage.COMPUTE
        ? gpu::RootPush::size
        : gpu::GraphicsRootPush::size;
    uint count = 0;
    if (reflect_module.enumerate_push_constant_blocks(&count, null)
        != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::BlockVariable*[] blocks =
            mem::talloc_array(spvreflect::BlockVariable*, (sz)count);
        if (reflect_module.enumerate_push_constant_blocks(&count, blocks.ptr)
            != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (block : blocks) {
            if ((usz)block.byte_offset() + (usz)block.byte_size() > expected_size) {
                return gpu::INVALID_ARGUMENT~;
            }
        }
    };
}

fn void? copy_entry_point(char* dst, ZString source) @private {
    usz i = 0;
    while (source[i] != 0) {
        if (i >= gpu::SHADER_ENTRY_POINT_CAP - 1) return gpu::INVALID_ARGUMENT~;
        dst[i] = source[i];
        i++;
    }
    dst[i] = 0;
}

fn fault report_shader_contract_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    ZString debug_name,
    ZString rejected_field,
    ZString invariant,
) @private {
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        public_fault,
        resource_kind:       gpu::DebugResourceKind.SHADER,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          debug_name,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        null,
    );
}

<* Validate one prepared shader and create the temporary native module used by
   a single pipeline build. *>
fn PipelineShaderModule? create_pipeline_shader_module(
    VkDeviceState* state,
    gpu::ShaderCode* code,
    gpu::ShaderStage expected_stage,
    ZString operation,
    ZString rejected_field,
) {
    if (code == null || !code.is_valid()) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     code == null ? null : code.debug_name,
            rejected_field: rejected_field,
            invariant:      "shader code must come from prepare_shader_code",
        )~;
    }
    if (code.stage != expected_stage) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "shader stage must match its pipeline role",
        )~;
    }

    PipelineShaderModule shader;
    if (catch entry_fault = copy_entry_point(
        &shader.entry_point[0],
        code.entry_point,
    )) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   entry_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "entry point must fit the backend name capacity",
        )~;
    }

    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module =
        (spvreflect::ShaderModule*)&module_storage;
    if (spvreflect::create_shader_module(
        code.spirv.len,
        code.spirv.ptr,
        reflect_module,
    ) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "SPIR-V reflection must succeed",
        )~;
    }
    defer reflect_module.destroy();

    if (catch heap_fault = check_heap_convention(reflect_module)) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   heap_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "descriptor bindings must match the gpu heap convention",
        )~;
    }
    spvreflect::EntryPoint* entry =
        reflect_module.get_entry_point((ZString)&shader.entry_point[0]);
    if (entry == null) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "entry point must exist in SPIR-V module",
        )~;
    }
    gpu::ShaderStage? reflected_stage_optional = gpu_stage_from_spv(entry.stage());
    if (catch reflected_fault = reflected_stage_optional) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   reflected_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "reflected execution model must be supported",
        )~;
    }
    gpu::ShaderStage reflected_stage = reflected_stage_optional;
    if (reflected_stage != code.stage) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "prepared stage must match the reflected entry point",
        )~;
    }
    if (catch push_fault = check_push_constant_range(reflect_module, code.stage)) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   push_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "push constants must fit the shader-stage root ABI",
        )~;
    }

    @pool() {
        uint[] aligned_code = mem::talloc_array(uint, code.spirv.len / uint::size);
        mem::copy(aligned_code.ptr, code.spirv.ptr, code.spirv.len);
        vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
            .set_code_size(code.spirv.len)
            .set_code(aligned_code.ptr);
        check_shader_create_device_result(
            state: state,
            result: vk::create_shader_module(
                device:          state.device,
                p_create_info:   &info,
                p_allocator:     null,
                p_shader_module: &shader.shader_module,
            ),
        )!;
    };
    state.pipeline_shader_modules_created++;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SHADER_MODULE,
        object_handle: (ulong)shader.shader_module,
        name:          code.debug_name,
    );
    return shader;
}

fn void destroy_pipeline_shader_module(
    VkDeviceState* state,
    PipelineShaderModule* shader,
) {
    if (shader == null || shader.shader_module == null) return;
    vk::destroy_shader_module(state.device, shader.shader_module, null);
    shader.shader_module = (vk::ShaderModule)null;
}

fn PipelineShaderModule*? get_pipeline_batch_shader(
    VkDeviceState* state,
    gpu::ShaderCode* code,
    gpu::ShaderStage expected_stage,
    ZString operation,
    ZString rejected_field,
    PipelineBatchShader[] shaders,
    uint* shader_count,
) {
    foreach (&shader : shaders[:*shader_count]) {
        if (shader.code == *code && shader.expected_stage == expected_stage) {
            return &shader.native;
        }
    }
    PipelineShaderModule native = create_pipeline_shader_module(
        state:          state,
        code:           code,
        expected_stage: expected_stage,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    shaders[*shader_count] = {
        .code           = *code,
        .expected_stage = expected_stage,
        .native         = native,
    };
    PipelineShaderModule* result = &shaders[*shader_count].native;
    (*shader_count)++;
    return result;
}

fn void destroy_pipeline_batch_shaders(
    VkDeviceState* state,
    PipelineBatchShader[] shaders,
    uint shader_count,
) {
    foreach (&shader : shaders[:shader_count]) {
        destroy_pipeline_shader_module(state, &shader.native);
    }
}
