module gpu;

const uint DEFAULT_TEXTURE_CAPACITY = 1024;

<* Texel format for textures and swapchain images. *>
enum Format : int {
    UNDEFINED,
    R8_UNORM,
    R8_UINT,
    RG8_UNORM,
    RGBA8_UNORM,
    RGBA8_SRGB,
    BGRA8_UNORM,
    BGRA8_SRGB,
    R16_UINT,
    R16_FLOAT,
    RG16_FLOAT,
    RGBA16_FLOAT,
    R32_UINT,
    R32_FLOAT,
    RG32_FLOAT,
    RGBA32_FLOAT,
    D32_FLOAT,
    // Backend-unsupported: creation faults INVALID_ARGUMENT (no
    // format-properties query exists yet to gate it per device).
    D24_UNORM_S8_UINT,
}

<* Texture dimensionality. Only TEX_2D creates today; the others are
   declared for API stability and fault INVALID_ARGUMENT at creation. *>
enum TextureDimension : int {
    TEX_1D,
    TEX_2D,
    TEX_3D,
    CUBE,
}

<* Sampler texel filtering. *>
enum Filter : int {
    NEAREST,
    LINEAR,
}

<* How samplers treat coordinates outside [0, 1]. *>
enum AddressMode : int {
    REPEAT,
    MIRRORED_REPEAT,
    CLAMP_TO_EDGE,
    CLAMP_TO_BORDER,
}

// Texture usage flags. Bitstruct of bools, composed by field-set.
bitstruct TextureUsage : uint {
    // Sampled through the descriptor heap.
    bool sampled      : 0;
    // Storage-image access from shaders.
    bool storage      : 1;
    // Render-pass color target.
    bool color_attach : 2;
    // Render-pass depth target.
    bool depth_attach : 3;
    // Source of copies/readbacks.
    bool transfer_src : 4;
    // Destination of uploads/copies.
    bool transfer_dst : 5;
    // Concurrent sharing across queue families; required when GRAPHICS and
    // COMPUTE both touch it and caps.async_compute is set.
    bool shared_queues : 6;
}

<* Creation parameters for create_texture. *>
struct TextureDesc {
    TextureDimension dimension;
    uint             width;
    uint             height;
    uint             depth;
    // 0 means 1.
    uint             mip_levels;
    // 0 means 1.
    uint             array_layers;
    Format           format;
    TextureUsage     usage;
    ZString          debug_name;
}

<* Subresource view of a texture; a zero-initialized desc is the full default view. *>
struct TextureViewDesc {
    // UNDEFINED means the texture's own format; reinterpretation is unsupported.
    Format format;
    uint   base_mip;
    // 0 means all mips from base_mip.
    uint   mip_count;
    uint   base_layer;
    // 0 means all layers from base_layer.
    uint   layer_count;
}

<* Creation parameters for create_sampler. *>
struct SamplerDesc {
    Filter      min_filter;
    Filter      mag_filter;
    Filter      mip_filter;
    AddressMode address_u;
    AddressMode address_v;
    AddressMode address_w;
    float       mip_lod_bias;
    float       min_lod;
    float       max_lod;
    // Requires caps.max_sampler_anisotropy > 0 and max_anisotropy >= 1.
    bool        anisotropy_enable;
    // Must be >= 1 when anisotropy_enable is set (INVALID_ARGUMENT
    // otherwise); upper-bounded by caps.max_sampler_anisotropy.
    float       max_anisotropy;
    // With `compare`, enables depth-compare (shadow) sampling.
    bool        compare_enable;
    CompareOp   compare;
    ZString     debug_name;
}

<* The handle owns the image; shader access goes through a TextureIndex.
   Ownership: release with destroy_texture. *>
fn TextureHandle? create_texture(Device* device, TextureDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_texture(device, desc);
}

<* Counterpart of create_texture; descriptors created from the texture must be destroyed first. *>
fn void? destroy_texture(Device* device, TextureHandle texture) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_texture(device, texture);
}
