module gpu::internal::vk @private;

import gpu @public;
import std::hash::fnv64a;

const uint HANDLE_GENERATION_MAX = 0xFF_FFFF;

enum IndexingHeapLimitFailure : char {
    NONE,
    SAMPLED_IMAGES,
    STORAGE_IMAGES,
    SAMPLERS,
    PER_STAGE_RESOURCES,
    ALL_POOLS_DESCRIPTORS,
}

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
    ulong sampled_images;
    ulong storage_images;
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
    uint             anisotropy_enable;
    float            max_anisotropy;
    uint             compare_enable;
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

fn SamplerKey canonical_sampler_key(gpu::SamplerDesc* desc) {
    return {
        .min_filter        = desc.min_filter,
        .mag_filter        = desc.mag_filter,
        .mip_filter        = desc.mip_filter,
        .address_u         = desc.address_u,
        .address_v         = desc.address_v,
        .address_w         = desc.address_w,
        .mip_lod_bias      = canonical_sampler_float(desc.mip_lod_bias),
        .min_lod           = canonical_sampler_float(desc.min_lod),
        .max_lod           = canonical_sampler_float(desc.max_lod),
        .anisotropy_enable = desc.anisotropy_enable ? 1 : 0,
        .max_anisotropy    = desc.anisotropy_enable
            ? canonical_sampler_float(desc.max_anisotropy)
            : 0.0f,
        .compare_enable    = desc.compare_enable ? 1 : 0,
        .compare           = desc.compare_enable ? desc.compare : {},
    };
}

fn HeapDescriptorTotals descriptor_heap_totals(uint tex_capacity, uint smp_capacity) {
    // Heap-local checks cannot reserve device-wide capacity used by other
    // update-after-bind pools or pipeline layouts.
    return {
        .sampled_images        = (ulong)tex_capacity * 2,
        .storage_images        = (ulong)tex_capacity * 2,
        .per_stage_resources   = (ulong)tex_capacity * 4,
        .all_pools_descriptors = (ulong)tex_capacity * 4
            + (ulong)smp_capacity,
    };
}

fn bool indexing_heap_fits_limits(
    uint tex_capacity,
    uint smp_capacity,
    IndexingHeapLimits limits,
) {
    return indexing_heap_limit_failure(
        tex_capacity,
        smp_capacity,
        limits,
    ) == IndexingHeapLimitFailure.NONE;
}

fn IndexingHeapLimitFailure indexing_heap_limit_failure(
    uint tex_capacity,
    uint smp_capacity,
    IndexingHeapLimits limits,
) {
    HeapDescriptorTotals totals = descriptor_heap_totals(tex_capacity, smp_capacity);
    if (totals.sampled_images > (ulong)limits.max_sampled_images
        || totals.sampled_images
            > (ulong)limits.max_per_stage_sampled_images) {
        return IndexingHeapLimitFailure.SAMPLED_IMAGES;
    }
    if (totals.storage_images > (ulong)limits.max_storage_images
        || totals.storage_images
            > (ulong)limits.max_per_stage_storage_images) {
        return IndexingHeapLimitFailure.STORAGE_IMAGES;
    }
    if (smp_capacity > limits.max_samplers
        || smp_capacity > limits.max_per_stage_samplers) {
        return IndexingHeapLimitFailure.SAMPLERS;
    }
    if (totals.per_stage_resources
        > (ulong)limits.max_per_stage_resources) {
        return IndexingHeapLimitFailure.PER_STAGE_RESOURCES;
    }
    if (totals.all_pools_descriptors
        > (ulong)limits.max_all_pools_descriptors) {
        return IndexingHeapLimitFailure.ALL_POOLS_DESCRIPTORS;
    }
    return IndexingHeapLimitFailure.NONE;
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
