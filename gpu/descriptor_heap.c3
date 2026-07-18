module gpu;

const uint DEFAULT_TEXTURE_DESCRIPTORS = 4096;
const uint DEFAULT_SAMPLER_DESCRIPTORS = 256;
// Ceiling set by the 16-bit slot field in TextureIndex/SamplerIndex.
const uint MAX_DESCRIPTOR_SLOTS = 65_536;

<* Register a texture in the shader-visible heap; the texture must carry
   sampled or storage usage (INVALID_ARGUMENT otherwise). The texture stays
   owned by its TextureHandle. Ownership: release with
   destroy_texture_descriptor. *>
fn TextureIndex? create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, texture.owner, ResourceOwnerKind.TEXTURE)!;
    TextureIndex index = operation.vtable.create_texture_descriptor(device, texture, view)!;
    retain_device_children(&operation);
    return index;
}

<* Release ownership and recycle the slot immediately. *>
fn void? destroy_texture_descriptor(Device* device, TextureIndex index) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    operation.vtable.destroy_texture_descriptor(device, index)!;
    release_device_children(&operation);
}

<* One create_texture_descriptors input; a zero-initialized view is the
   default view (same collapse as create_texture_descriptor's null view). *>
struct TextureDescriptorDesc {
    TextureHandle   texture;
    TextureViewDesc view;
}

<* Batch-create texture descriptors; out_indices.len must equal descs.len
   (INVALID_ARGUMENT otherwise), empty descs is a no-op. All-or-nothing: a
   fault leaves descriptor allocation, texture view caches, Vulkan view
   ownership, and out_indices unchanged. Ownership on success: release each
   index with destroy_texture_descriptor. *>
fn void? create_texture_descriptors(Device* device, TextureDescriptorDesc[] descs, TextureIndex[] out_indices) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    foreach (&desc : descs) {
        validate_resource_owner(&operation, desc.texture.owner, ResourceOwnerKind.TEXTURE)!;
    }
    operation.vtable.create_texture_descriptors(device, descs, out_indices)!;
    retain_device_children(&operation, (ulong)descs.len);
}

<* Samplers are pure heap entries; the descriptor is the resource.
   Ownership: release with destroy_sampler. *>
fn SamplerIndex? create_sampler(Device* device, SamplerDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    SamplerIndex index = operation.vtable.create_sampler(device, desc)!;
    retain_device_children(&operation);
    return index;
}

<* Release the sampler and recycle its heap slot immediately. *>
fn void? destroy_sampler(Device* device, SamplerIndex index) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    operation.vtable.destroy_sampler(device, index)!;
    release_device_children(&operation);
}
