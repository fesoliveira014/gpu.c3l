module retired_shader_handle;

import gpu;

fn void use_retired_shader_handle() {
    gpu::ShaderHandle shader = {};
    (void)shader;
}
