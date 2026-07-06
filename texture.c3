module gpu;

const uint MAX_TEXTURES = 1024;

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
    D24_UNORM_S8_UINT,
}

enum TextureDimension : int {
    TEX_1D,
    TEX_2D,
    TEX_3D,
    CUBE,
}

enum Filter : int {
    NEAREST,
    LINEAR,
}

enum AddressMode : int {
    REPEAT,
    MIRRORED_REPEAT,
    CLAMP_TO_EDGE,
    CLAMP_TO_BORDER,
}

// Texture usage flags. Bitstruct of bools, composed by field-set.
bitstruct TextureUsage : uint {
    bool sampled      : 0;
    bool storage      : 1;
    bool color_attach : 2;
    bool depth_attach : 3;
    bool transfer_src : 4;
    bool transfer_dst : 5;
}

struct TextureDesc {
    TextureDimension dimension;
    uint             width;
    uint             height;
    uint             depth;
    uint             mip_levels;
    uint             array_layers;
    Format           format;
    TextureUsage     usage;
    ZString          debug_name;
}

struct TextureViewDesc {
    Format format;
    uint   base_mip;
    uint   mip_count;
    uint   base_layer;
    uint   layer_count;
}

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
    bool        anisotropy_enable;
    float       max_anisotropy;
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

fn void? destroy_texture(Device* device, TextureHandle texture) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_texture(device, texture);
}
