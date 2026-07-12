module gpu_consumer;

import gpu;

const char[*] COMPUTE_SHADER_SPIRV = $embed("shader.comp.spv");

fn int main() {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_consumer_package",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = COMPUTE_SHADER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!!;
    gpu::destroy_shader(&device, shader)!!;
    return 0;
}
