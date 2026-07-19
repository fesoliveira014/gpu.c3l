module retired_create_tex_desc;

import gpu;

fn void? use_retired(
    gpu::Device* device,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* desc,
) {
    gpu::create_texture_descriptor(device, texture, desc)!;
}
