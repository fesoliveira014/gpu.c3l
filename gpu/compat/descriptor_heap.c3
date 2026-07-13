module gpu::compat;

const uint DEFAULT_TEXTURE_DESCRIPTORS = 4096;
const uint DEFAULT_SAMPLER_DESCRIPTORS = 256;
// Ceiling set by the 16-bit slot field in TextureIndex/SamplerIndex.
const uint MAX_DESCRIPTOR_SLOTS = 65_536;

<* Register a texture in the shader-visible heap; the texture must carry
   sampled or storage usage (INVALID_ARGUMENT otherwise). The texture stays
   owned by its TextureHandle. Ownership: release with
   destroy_texture_descriptor. *>
fn TextureIndex? create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.create_texture_descriptor(device, texture, view);
}

<* The slot is recycled only after in-flight frames retire. *>
fn void? destroy_texture_descriptor(Device* device, TextureIndex index) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_texture_descriptor(device, index);
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
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.create_texture_descriptors(device, descs, out_indices);
}

<* Samplers are pure heap entries; the descriptor is the resource.
   Ownership: release with destroy_sampler. *>
fn SamplerIndex? create_sampler(Device* device, SamplerDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.create_sampler(device, desc);
}

<* Counterpart of create_sampler; the heap slot recycles after in-flight frames retire. *>
fn void? destroy_sampler(Device* device, SamplerIndex index) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_sampler(device, index);
}
