module gpu_test;

import gpu;

// Headless (lavapipe): reflection validation of shader modules against the
// heap convention. Fixtures come from test/shaders/ via build_shaders.sh.

const char[*] REFLECT_HEAP_SPIRV        = $embed("shaders/heap_write.comp.spv");
const char[*] REFLECT_BUFFERS_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] REFLECT_BAD_SET_SPIRV     = $embed("shaders/bad_set.comp.spv");
const char[*] REFLECT_BAD_BINDING_SPIRV = $embed("shaders/bad_binding.comp.spv");
const char[*] REFLECT_OVERSIZED_PUSH_SPIRV = $embed("shaders/oversized_push.comp.spv");

const ZString ENTRY_POINT_OVER_CAP = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

fn gpu::Device make_reflection_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_shader_reflection_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void heap_conforming_shader_accepted() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_HEAP_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &desc)!!;
    assert(shader.is_valid(), "heap-conforming shader is accepted");
    gpu::destroy_shader(&device, shader)!!;
}

fn void buffers_only_shader_accepted() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &desc)!!;
    assert(shader.is_valid(), "buffers-only shader with no bindings is accepted");
    gpu::destroy_shader(&device, shader)!!;
}

fn void foreign_set_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BAD_SET_SPIRV[..], .entry_point = "main",
    };
    fault set_fault = @catch(gpu::create_shader(&device, &desc));
    assert(set_fault == gpu::SHADER_INVALID, "a set-1 binding faults SHADER_INVALID");
}

fn void convention_violating_binding_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BAD_BINDING_SPIRV[..], .entry_point = "main",
    };
    fault binding_fault = @catch(gpu::create_shader(&device, &desc));
    assert(binding_fault == gpu::SHADER_INVALID, "a combined-image-sampler at set-0 binding 0 faults SHADER_INVALID");
}

fn void bogus_entry_point_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "not_main",
    };
    fault entry_fault = @catch(gpu::create_shader(&device, &desc));
    assert(entry_fault == gpu::INVALID_ARGUMENT, "a requested entry point absent from the module faults INVALID_ARGUMENT");
}

fn void wrong_stage_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.VERTEX, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "main",
    };
    fault stage_fault = @catch(gpu::create_shader(&device, &desc));
    assert(stage_fault == gpu::INVALID_ARGUMENT, "requesting VERTEX for a compute module faults INVALID_ARGUMENT");
}

fn void oversized_push_constant_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_OVERSIZED_PUSH_SPIRV[..], .entry_point = "main",
    };
    fault push_fault = @catch(gpu::create_shader(&device, &desc));
    assert(push_fault == gpu::INVALID_ARGUMENT, "a push-constant block past the root-push ABI faults INVALID_ARGUMENT");
}

fn void entry_point_over_cap_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = ENTRY_POINT_OVER_CAP,
    };
    fault cap_fault = @catch(gpu::create_shader(&device, &desc));
    assert(cap_fault == gpu::INVALID_ARGUMENT, "an entry point name past SHADER_ENTRY_POINT_CAP faults INVALID_ARGUMENT");
}
