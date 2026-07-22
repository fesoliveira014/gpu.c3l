module gpu;

<* Opaque device-owned compatibility value for texture placement. *>
typedef TextureCompatibility = uint128;

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
}

<* Rasterization samples per texel. The zero value is single-sampled. *>
enum SampleCount : int {
    ONE,
    TWO,
    FOUR,
    EIGHT,
    SIXTEEN,
    THIRTY_TWO,
    SIXTY_FOUR,
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
    TextureSampleCountSupport sample_counts;
}

<* Storage requirements for one texture descriptor. *>
struct TextureRequirements {
    usz                  size;
    usz                  alignment;
    TextureCompatibility compatibility;
    bool                 dedicated_only;
}

<* Independently owned texture and allocation returned by dedicated creation. *>
struct DedicatedTexture {
    TextureHandle texture;
    GpuAllocation allocation;
}

<* Creation parameters for create_texture. `access` must be a non-empty subset
   of the device's selected queue roles. *>
struct TextureDesc {
    uint             width;
    uint             height;
    // 0 means 1.
    uint             mip_levels;
    // 0 means 1.
    uint             array_layers;
    Format           format;
    TextureUsage     usage;
    QueueRoles       access;
    SampleCount      sample_count;
    ZString          debug_name;
}

<* Subresource view of a texture; a zero-initialized desc is the full default view. *>
struct TextureViewDesc {
    uint   base_mip;
    // 0 means all mips from base_mip.
    uint   mip_count;
    uint   base_layer;
    // 0 means all layers from base_layer.
    uint   layer_count;
}

<* Semantic parameters for intern_sampler. *>
struct SamplerDesc {
    Filter      min_filter;
    Filter      mag_filter;
    Filter      mip_filter;
    AddressMode address_u;
    AddressMode address_v;
    AddressMode address_w;
    // LOD values must be finite; min_lod must not exceed max_lod. The absolute
    // mip_lod_bias value must not exceed caps.max_sampler_lod_bias.
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

fn bool TextureCompatibility.is_valid(self) {
    uint128 raw = (uint128)self;
    ulong payload = (ulong)raw;
    return (ulong)(raw >> 64) != 0
        && (uint)payload != 0
        && payload >> 32 == 0;
}

fn ulong texture_compatibility_owner(TextureCompatibility compatibility) @private {
    return (ulong)((uint128)compatibility >> 64);
}

fn uint texture_compatibility_value(TextureCompatibility compatibility) @private {
    return (uint)(uint128)compatibility;
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
            return true;
        default:
            return false;
    }
}

fn bool sample_count_valid(SampleCount sample_count) @private {
    switch (sample_count) {
        case SampleCount.ONE:
        case SampleCount.TWO:
        case SampleCount.FOUR:
        case SampleCount.EIGHT:
        case SampleCount.SIXTEEN:
        case SampleCount.THIRTY_TWO:
        case SampleCount.SIXTY_FOUR:
            return true;
        default:
            return false;
    }
}

fn bool texture_desc_valid(TextureDesc* desc) @private {
    if (desc.width == 0 || desc.height == 0) return false;
    if (texture_mip_levels(desc) > texture_full_mip_chain(desc.width, desc.height)) return false;
    if (!texture_format_in_backend_profile(desc.format)) return false;

    bool has_usage = desc.usage.sampled || desc.usage.storage || desc.usage.color_attach
        || desc.usage.depth_attach || desc.usage.transfer_src || desc.usage.transfer_dst;
    if (!has_usage) return false;

    if (!sample_count_valid(desc.sample_count)) return false;
    if (desc.sample_count != SampleCount.ONE
        && (texture_mip_levels(desc) != 1
            || desc.usage.sampled
            || desc.usage.storage
            || desc.usage.transfer_src
            || desc.usage.transfer_dst
            || (!desc.usage.color_attach && !desc.usage.depth_attach))) {
        return false;
    }

    bool is_depth = desc.format == Format.D32_FLOAT;
    if (is_depth && (desc.usage.color_attach || desc.usage.storage)) return false;
    if (!is_depth && desc.usage.depth_attach) return false;

    return true;
}

<* Query library-creatable features for a format. Individual feature bits do
   not guarantee that a combination is supported. *>
fn TextureFormatSupport? get_texture_format_support(Device* device, Format format) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_texture_format_support(device, format);
}

<* Return whether an exact descriptor is creatable by the backend and adapter.
   This checks capabilities only; allocation can still fail for lack of memory. *>
fn bool? supports_texture_desc(Device* device, TextureDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    if (!queue_access_is_valid(&operation, desc.access)) return false;
    return operation.vtable.supports_texture_desc(device, desc);
}

<* Return exact storage requirements for a texture descriptor. *>
fn TextureRequirements? get_texture_requirements(Device* device, TextureDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;
    return operation.vtable.get_texture_requirements(device, desc);
}

<* The handle owns the image; shader access goes through a TextureIndex.
   Ownership: release with destroy_texture. *>
fn TextureHandle? create_texture(Device* device, TextureDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;
    TextureHandle texture = operation.vtable.create_texture(device, desc)!;
    retain_device_children(&operation);
    return texture;
}

<* Create a texture in caller-owned texture memory. *>
fn TextureHandle? create_placed_texture(
    Device* device,
    TextureDesc* desc,
    GpuAllocation allocation,
    usz offset,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    TextureHandle texture = operation.vtable.create_placed_texture(
        device:     device,
        desc:       desc,
        allocation: allocation,
        offset:     offset,
    )!;
    retain_device_children(&operation);
    return texture;
}

<* Create a texture and its independently owned dedicated storage atomically. *>
fn DedicatedTexture? create_dedicated_texture(
    Device* device,
    TextureDesc* desc,
    AllocationDesc* allocation_desc,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null || allocation_desc == null
        || allocation_desc.size == 0
        || allocation_desc.memory_class != MemoryClass.TEXTURE) {
        return INVALID_ARGUMENT~;
    }
    validate_queue_access(&operation, desc.access)!;
    validate_queue_access(&operation, allocation_desc.access)!;
    if (!queue_roles_are_subset(desc.access, allocation_desc.access)) {
        return INVALID_ARGUMENT~;
    }

    AllocationDesc normalized = *allocation_desc;
    normalized.alignment = normalized_allocation_alignment(
        allocation_desc.alignment,
    )!;
    validate_texture_allocation(
        operation:       &operation,
        desc:            allocation_desc,
        alignment:       &normalized.alignment,
        allow_dedicated: true,
    )!;
    DedicatedTexture created = operation.vtable.create_dedicated_texture(
        device,
        desc,
        &normalized,
    )!;
    retain_device_children(&operation, 2);
    return created;
}

<* Destroy a texture after its descriptors. Placed storage remains caller-owned. *>
fn void? destroy_texture(Device* device, TextureHandle texture) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, texture.owner, ResourceOwnerKind.TEXTURE)!;
    operation.vtable.destroy_texture(device, texture)!;
    release_device_children(&operation);
}

<* Intern and publish an immutable device-lifetime sampler index. Identical
   semantic descriptions return the same value; debug_name is not semantic. *>
fn SamplerIndex? intern_sampler(Device* device, SamplerDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
    return operation.vtable.intern_sampler(device, desc);
}
