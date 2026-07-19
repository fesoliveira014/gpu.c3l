module retired_destroy_shader;

import gpu;

fn void? use_retired_destroy_shader(gpu::Device* device) {
    gpu::destroy_shader(device, {})!;
}
