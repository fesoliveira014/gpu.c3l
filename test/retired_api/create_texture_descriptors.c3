module retired_create_tex_descs;

import gpu;

fn void? use_retired(
    gpu::Device* device,
    gpu::TextureViewCreateDesc[] descs,
    gpu::TextureView[] views,
) {
    gpu::create_texture_descriptors(device, descs, views)!;
}
