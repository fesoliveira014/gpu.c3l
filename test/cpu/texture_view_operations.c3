module texture_view_operations;

import gpu;

fn gpu::TextureIndex? texture_view_operations(
    gpu::Device* device,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* desc,
) {
    gpu::TextureView view = gpu::create_texture_view(device, texture, desc)!;
    defer (void)gpu::destroy_texture_view(device, view);
    return view.index;
}

fn void? texture_view_batch_operations(
    gpu::Device* device,
    gpu::TextureHandle[] textures,
    gpu::TextureIndex[] out_indices,
) {
    if (textures.len != out_indices.len) return gpu::INVALID_ARGUMENT~;
    gpu::TextureViewCreateDesc[4] descs;
    gpu::TextureView[4] views;
    if (textures.len > descs.len) return gpu::INVALID_ARGUMENT~;
    foreach (i, texture : textures) descs[i] = { .texture = texture };
    gpu::create_texture_views(
        device:    device,
        descs:     descs[..textures.len],
        out_views: views[..textures.len],
    )!;
    foreach (i, view : views[..textures.len]) {
        out_indices[i] = view.index;
        gpu::destroy_texture_view(device, view)!;
    }
}
