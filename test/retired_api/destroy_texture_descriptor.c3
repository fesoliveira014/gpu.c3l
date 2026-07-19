module retired_destroy_tex_desc;

import gpu;

fn void? use_retired(gpu::Device* device, gpu::TextureIndex index) {
    gpu::destroy_texture_descriptor(device, index)!;
}
