module retired_create_shader;

import gpu;

fn void? use_retired_create_shader(gpu::Device* device, gpu::ShaderDesc* desc) {
    (void)gpu::create_shader(device, desc)!;
}
