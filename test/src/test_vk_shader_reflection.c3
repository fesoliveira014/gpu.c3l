module gpu_test;

import gpu;

// Headless (lavapipe): reflection validation of shader modules against the
// heap convention. Fixtures come from test/shaders/ via build_shaders.sh.

const char[*] REFLECT_HEAP_SPIRV        = $embed("shaders/heap_write.comp.spv");
const char[*] REFLECT_BUFFERS_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] REFLECT_BAD_SET_SPIRV     = $embed("shaders/bad_set.comp.spv");
const char[*] REFLECT_BAD_BINDING_SPIRV = $embed("shaders/bad_binding.comp.spv");

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
    bool faulted = false;
    if (catch gpu::create_shader(&device, &desc)) faulted = true;
    assert(faulted, "a set-1 binding must fault SHADER_INVALID");
}

fn void convention_violating_binding_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BAD_BINDING_SPIRV[..], .entry_point = "main",
    };
    bool faulted = false;
    if (catch gpu::create_shader(&device, &desc)) faulted = true;
    assert(faulted, "a combined-image-sampler at set-0 binding 0 must fault SHADER_INVALID");
}
