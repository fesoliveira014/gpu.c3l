module retired_prepare_shader_code;

import gpu;

fn void use_retired_prepare_shader_code(gpu::ShaderDesc* desc) {
    (void)gpu::prepare_shader_code(desc);
}
