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
    // Declared for API stability; query support before use.
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

<* Individually creatable optimal-tiling format features. Feature
   combinations must be checked with supports_texture_desc. *>
bitstruct TextureFormatFeatures : uint {
    bool sampled       : 0;
    bool storage       : 1;
    bool color_attach  : 2;
    bool depth_attach  : 3;
    bool transfer_src  : 4;
    bool transfer_dst  : 5;
    bool linear_filter : 6;
}

<* Texture dimensions implemented end-to-end by the backend. *>
bitstruct TextureDimensionSupport : uint {
    bool tex_1d : 0;
    bool tex_2d : 1;
    bool tex_3d : 2;
    bool cube   : 3;
}

<* Texture sample counts implemented end-to-end by the backend. *>
bitstruct TextureSampleCountSupport : uint {
    bool one        : 0;
    bool two        : 1;
    bool four       : 2;
    bool eight      : 3;
    bool sixteen    : 4;
    bool thirty_two : 5;
    bool sixty_four : 6;
}

<* Library-creatable support for one format. *>
struct TextureFormatSupport {
    TextureFormatFeatures     features;
    TextureDimensionSupport   dimensions;
    TextureSampleCountSupport sample_counts;
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
fn uint texture_mip_levels(TextureDesc* desc) @private {
    return desc.mip_levels == 0 ? 1 : desc.mip_levels;
}

fn uint texture_array_layers(TextureDesc* desc) @private {
    return desc.array_layers == 0 ? 1 : desc.array_layers;
}

fn uint texture_full_mip_chain(uint width, uint height) @private {
    uint levels = 1;
    uint size = width > height ? width : height;
    while (size > 1) {
        size >>= 1;
        levels++;
    }
    return levels;
}

fn bool texture_format_in_backend_profile(Format format) @private {
    switch (format) {
        case Format.R8_UNORM:
        case Format.R8_UINT:
        case Format.RG8_UNORM:
        case Format.RGBA8_UNORM:
        case Format.RGBA8_SRGB:
        case Format.BGRA8_UNORM:
        case Format.BGRA8_SRGB:
        case Format.R16_UINT:
        case Format.R16_FLOAT:
        case Format.RG16_FLOAT:
        case Format.RGBA16_FLOAT:
        case Format.R32_UINT:
        case Format.R32_FLOAT:
        case Format.RG32_FLOAT:
        case Format.RGBA32_FLOAT:
        case Format.D32_FLOAT:
        case Format.D24_UNORM_S8_UINT:
            return true;
        default:
            return false;
    }
}

fn bool texture_desc_valid(TextureDesc* desc) @private {
    if (desc.dimension != TextureDimension.TEX_2D) return false;
    if (desc.width == 0 || desc.height == 0 || desc.depth > 1) return false;
    if (texture_mip_levels(desc) > texture_full_mip_chain(desc.width, desc.height)) return false;
    if (!texture_format_in_backend_profile(desc.format)) return false;
    bool has_usage = desc.usage.sampled || desc.usage.storage || desc.usage.color_attach
        || desc.usage.depth_attach || desc.usage.transfer_src || desc.usage.transfer_dst;
    if (!has_usage) return false;
    bool is_depth = desc.format == Format.D32_FLOAT
        || desc.format == Format.D24_UNORM_S8_UINT;
    if (is_depth && (desc.usage.color_attach || desc.usage.storage)) return false;
    if (desc.format == Format.D24_UNORM_S8_UINT
        && (desc.usage.transfer_src || desc.usage.transfer_dst)) return false;
    if (!is_depth && desc.usage.depth_attach) return false;
    return true;
}

<* Query library-creatable features for a format. Individual feature bits do
   not guarantee that a combination is supported. *>
fn TextureFormatSupport? get_texture_format_support(Device* device, Format format) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_texture_format_support(device, format);
}

<* Return whether an exact descriptor is creatable by the backend and adapter.
   This checks capabilities only; allocation can still fail for lack of memory. *>
fn bool? supports_texture_desc(Device* device, TextureDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.supports_texture_desc(device, desc);
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
