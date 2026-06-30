module gpu;

enum Format : int {
    UNDEFINED,
    R8_UNORM,
    R8_UINT,
    RG8_UNORM,
    RGBA8_UNORM,
    RGBA8_SRGB,
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
