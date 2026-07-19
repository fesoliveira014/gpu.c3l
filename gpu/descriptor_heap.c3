module gpu;

const uint DEFAULT_TEXTURE_DESCRIPTORS = 4096;
const uint DEFAULT_SAMPLER_DESCRIPTORS = 256;
const uint MAX_DESCRIPTOR_SLOTS = 65_536;

<* Publish one texture view in the strict shader-visible heap. The texture must
   carry sampled or storage usage. The returned owner-bearing view contains the
   raw generation-free index stored in shader data. *>
fn TextureView? create_texture_view(
    Device* device,
    TextureHandle texture,
    TextureViewDesc* desc,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, texture.owner, ResourceOwnerKind.TEXTURE)!;
    TextureView view = operation.vtable.create_texture_view(device, texture, desc)!;
    retain_device_children(&operation);
    return view;
}

<* Release one texture view and recycle its shader-visible index immediately.
   GPU references to the index must already be complete. *>
fn void? destroy_texture_view(Device* device, TextureView view) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, view.owner, ResourceOwnerKind.TEXTURE_VIEW)!;
    operation.vtable.destroy_texture_view(device, view)!;
    release_device_children(&operation);
}

<* One create_texture_views input. A zero-initialized view description selects
   the default view, matching a null description passed to create_texture_view. *>
struct TextureViewCreateDesc {
    TextureHandle   texture;
    TextureViewDesc view;
}

<* Batch-publish texture views. `out_views.len` must equal `descs.len`; an empty
   batch succeeds. A fault leaves heap allocation, generations, cached native
   views, and outputs unchanged. *>
fn void? create_texture_views(
    Device* device,
    TextureViewCreateDesc[] descs,
    TextureView[] out_views,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    foreach (&desc : descs) {
        validate_resource_owner(&operation, desc.texture.owner, ResourceOwnerKind.TEXTURE)!;
    }
    operation.vtable.create_texture_views(device, descs, out_views)!;
    retain_device_children(&operation, (ulong)descs.len);
}

<* Publish an interned sampler to the strict shader-visible heap. Requires
   strict capability; repeated publication returns the same device-lifetime index. *>
fn SamplerIndex? publish_sampler(Device* device, Sampler sampler) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, sampler.owner, ResourceOwnerKind.SAMPLER)!;
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
    return operation.vtable.publish_sampler(device, sampler);
}
