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

<* @require textures.len == out_indices.len : "out_indices.len must equal textures.len"
   @require textures.len <= 4 : "textures.len must not exceed the batch capacity" *>
fn void? texture_view_batch_operations(
    gpu::Device* device,
    gpu::TextureHandle[] textures,
    gpu::TextureIndex[] out_indices,
) {
    gpu::TextureViewCreateDesc[4] descs;
    gpu::TextureView[4] views;
    foreach (i, texture : textures) descs[i] = { .texture = texture };
    gpu::create_texture_views(
        device:    device,
        descs:     descs[..textures.len],
        out_views: views[..textures.len],
    )!;
    foreach (i, view : views[..textures.len]) {
        out_indices[i] = view.index;
        gpu::destroy_texture_view(device, view);
    }
}

fn usz texture_transition_operations(
    gpu::TextureHandle texture,
    gpu::TextureDesc* desc,
    gpu::TextureBarrier* out_barrier,
) {
    gpu::TextureState before = { .layout = gpu::TextureLayout.UNDEFINED };
    *out_barrier = gpu::texture_transition(texture, before, gpu::sampled_at({ .compute }));
    return gpu::texture_mip_bytes(desc, 0);
}
