module gpu;

const uint DEFAULT_TEXTURE_HEAP_CAPACITY = 4096;
const uint DEFAULT_SAMPLER_HEAP_CAPACITY = 256;
const uint MAX_SHADER_HEAP_CAPACITY = 65_536;

<* One create_texture_views input. A zero-initialized view description selects
   the default view, matching a null description passed to create_texture_view. *>
struct TextureViewCreateDesc {
    TextureHandle   texture;
    TextureViewDesc view;
}

<* Publish one texture view in the strict shader-visible heap. Requires strict
   capability, and the texture must carry sampled or storage usage. The returned
   owner-bearing view contains the raw generation-free index stored in shader data. *>
fn TextureView? create_texture_view(
    Device* device,
    TextureHandle texture,
    TextureViewDesc* desc,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, texture.owner, ResourceOwnerKind.TEXTURE)!;
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
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

<* Batch-publish texture views. Requires strict capability.
   `out_views.len` must equal `descs.len`; an empty batch succeeds. A fault leaves
   heap allocation, generations, cached native views, and outputs unchanged. *>
fn void? create_texture_views(
    Device* device,
    TextureViewCreateDesc[] descs,
    TextureView[] out_views,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (out_views.len != descs.len) {
        return report_device_contract_failure(
            operation:      &operation,
            operation_name: "create_texture_views",
            public_fault:   INVALID_ARGUMENT,
            resource_kind:  DebugResourceKind.TEXTURE_DESCRIPTOR,
            rejected_field: "out_views",
            invariant:      "output count must match descriptor count",
        )~;
    }
    if (descs.len == 0) return;
    foreach (&desc : descs) {
        validate_resource_owner(&operation, desc.texture.owner, ResourceOwnerKind.TEXTURE)!;
    }
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
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
