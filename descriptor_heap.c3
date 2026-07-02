module gpu;

const uint DEFAULT_TEXTURE_DESCRIPTORS = 4096;
const uint DEFAULT_SAMPLER_DESCRIPTORS = 256;
// Ceiling set by the 16-bit slot field in TextureIndex/SamplerIndex.
const uint MAX_DESCRIPTOR_SLOTS = 65_536;

<* Zero means default; capacities above the packed-slot ceiling fault. *>
fn uint? resolve_descriptor_capacity(uint requested, uint default_capacity) {
    uint capacity = requested == 0 ? default_capacity : requested;
    if (capacity > MAX_DESCRIPTOR_SLOTS) return INVALID_ARGUMENT~;
    return capacity;
}

<* Register a texture in the shader-visible heap. The texture stays owned by
   its TextureHandle. Ownership: release with destroy_texture_descriptor. *>
fn TextureIndex? create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_texture_descriptor(device, texture, view);
}

<* The slot is recycled only after in-flight frames retire. *>
fn void? destroy_texture_descriptor(Device* device, TextureIndex index) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_texture_descriptor(device, index);
}

<* Samplers are pure heap entries; the descriptor is the resource.
   Ownership: release with destroy_sampler. *>
fn SamplerIndex? create_sampler(Device* device, SamplerDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_sampler(device, desc);
}

fn void? destroy_sampler(Device* device, SamplerIndex index) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_sampler(device, index);
}
