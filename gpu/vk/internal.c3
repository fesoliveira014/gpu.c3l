module gpu::vk @private;

import gpu @public;
import std::hash::fnv64a;

const uint HANDLE_GENERATION_MAX = 0xFF_FFFF;

struct IndexingHeapLimits {
    uint max_sampled_images;
    uint max_storage_images;
    uint max_samplers;
    uint max_per_stage_sampled_images;
    uint max_per_stage_storage_images;
    uint max_per_stage_samplers;
    uint max_per_stage_resources;
    uint max_all_pools_descriptors;
}

struct HeapDescriptorTotals {
    ulong per_stage_resources;
    ulong all_pools_descriptors;
}

struct SamplerKey {
    gpu::Filter      min_filter;
    gpu::Filter      mag_filter;
    gpu::Filter      mip_filter;
    gpu::AddressMode address_u;
    gpu::AddressMode address_v;
    gpu::AddressMode address_w;
    float            mip_lod_bias;
    float            min_lod;
    float            max_lod;
    bool             anisotropy_enable;
    float            max_anisotropy;
    bool             compare_enable;
    gpu::CompareOp   compare;
}
$assert SamplerKey::size == 52;

fn float canonical_sampler_float(float value) {
    return value == 0.0f ? 0.0f : value;
}

fn bool sampler_key_equal(SamplerKey* left, SamplerKey* right) {
    return left.min_filter == right.min_filter
        && left.mag_filter == right.mag_filter
        && left.mip_filter == right.mip_filter
        && left.address_u == right.address_u
        && left.address_v == right.address_v
        && left.address_w == right.address_w
        && left.mip_lod_bias == right.mip_lod_bias
        && left.min_lod == right.min_lod
        && left.max_lod == right.max_lod
        && left.anisotropy_enable == right.anisotropy_enable
        && left.max_anisotropy == right.max_anisotropy
        && left.compare_enable == right.compare_enable
        && left.compare == right.compare;
}

fn ulong hash_sampler_key(SamplerKey* key) {
    return fnv64a::hash(((char*)key)[:SamplerKey::size]);
}

fn bool sampler_filter_is_valid(gpu::Filter filter) {
    return filter == gpu::Filter.NEAREST || filter == gpu::Filter.LINEAR;
}

fn bool sampler_address_mode_is_valid(gpu::AddressMode mode) {
    switch (mode) {
        case gpu::AddressMode.REPEAT:
        case gpu::AddressMode.MIRRORED_REPEAT:
        case gpu::AddressMode.CLAMP_TO_EDGE:
        case gpu::AddressMode.CLAMP_TO_BORDER:
            return true;
        default:
            return false;
    }
}

fn bool sampler_compare_op_is_valid(gpu::CompareOp op) {
    switch (op) {
        case gpu::CompareOp.NEVER:
        case gpu::CompareOp.LESS:
        case gpu::CompareOp.EQUAL:
        case gpu::CompareOp.LESS_EQUAL:
        case gpu::CompareOp.GREATER:
        case gpu::CompareOp.NOT_EQUAL:
        case gpu::CompareOp.GREATER_EQUAL:
        case gpu::CompareOp.ALWAYS:
            return true;
        default:
            return false;
    }
}

fn ZString sampler_desc_invalid_enum_field(gpu::SamplerDesc* desc) {
    if (!sampler_filter_is_valid(desc.min_filter)) return "SamplerDesc.min_filter";
    if (!sampler_filter_is_valid(desc.mag_filter)) return "SamplerDesc.mag_filter";
    if (!sampler_filter_is_valid(desc.mip_filter)) return "SamplerDesc.mip_filter";
    if (!sampler_address_mode_is_valid(desc.address_u)) return "SamplerDesc.address_u";
    if (!sampler_address_mode_is_valid(desc.address_v)) return "SamplerDesc.address_v";
    if (!sampler_address_mode_is_valid(desc.address_w)) return "SamplerDesc.address_w";
    if (desc.compare_enable && !sampler_compare_op_is_valid(desc.compare)) {
        return "SamplerDesc.compare";
    }
    return null;
}

fn SamplerKey canonical_sampler_key(
    gpu::SamplerDesc* desc,
    float device_max_anisotropy,
) {
    float max_anisotropy;
    if (desc.anisotropy_enable) {
        max_anisotropy = desc.max_anisotropy > device_max_anisotropy
            ? device_max_anisotropy
            : desc.max_anisotropy;
    }
    SamplerKey key = {};
    key.min_filter        = desc.min_filter;
    key.mag_filter        = desc.mag_filter;
    key.mip_filter        = desc.mip_filter;
    key.address_u         = desc.address_u;
    key.address_v         = desc.address_v;
    key.address_w         = desc.address_w;
    key.mip_lod_bias      = canonical_sampler_float(desc.mip_lod_bias);
    key.min_lod           = canonical_sampler_float(desc.min_lod);
    key.max_lod           = canonical_sampler_float(desc.max_lod);
    key.anisotropy_enable = desc.anisotropy_enable;
    key.max_anisotropy    = canonical_sampler_float(max_anisotropy);
    key.compare_enable    = desc.compare_enable;
    key.compare           = desc.compare_enable ? desc.compare : {};
    return key;
}

fn HeapDescriptorTotals descriptor_heap_totals(uint tex_capacity, uint smp_capacity) {
    return {
        .per_stage_resources   = (ulong)tex_capacity * 2,
        .all_pools_descriptors = (ulong)tex_capacity * 2 + (ulong)smp_capacity,
    };
}

fn bool indexing_heap_fits_limits(
    uint tex_capacity,
    uint smp_capacity,
    IndexingHeapLimits limits,
) {
    HeapDescriptorTotals totals = descriptor_heap_totals(tex_capacity, smp_capacity);
    return tex_capacity <= limits.max_sampled_images
        && tex_capacity <= limits.max_storage_images
        && tex_capacity <= limits.max_per_stage_sampled_images
        && tex_capacity <= limits.max_per_stage_storage_images
        && smp_capacity <= limits.max_samplers
        && smp_capacity <= limits.max_per_stage_samplers
        && totals.per_stage_resources <= (ulong)limits.max_per_stage_resources
        && totals.all_pools_descriptors <= (ulong)limits.max_all_pools_descriptors;
}

fn uint? next_handle_generation(uint generation) {
    if (generation >= HANDLE_GENERATION_MAX) return gpu::SLOT_TABLE_FULL~;
    return generation + 1;
}

fn bool can_allocate_handle_generation(uint generation) {
    return generation < HANDLE_GENERATION_MAX;
}

fn bool should_recycle_handle_generation(uint generation) {
    return generation < HANDLE_GENERATION_MAX - 1;
}

fn bool is_pow2(usz value) {
    return value != 0 && (value & (value - 1)) == 0;
}

fn uint? resolve_descriptor_capacity(
    uint requested,
    uint default_capacity,
) {
    uint capacity = requested == 0 ? default_capacity : requested;
    if (capacity > gpu::MAX_SHADER_HEAP_CAPACITY) return gpu::INVALID_ARGUMENT~;
    return capacity;
}
