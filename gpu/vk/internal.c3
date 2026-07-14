module gpu::vk @private;

import gpu @public;

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

fn uint next_handle_generation(uint generation) {
    return generation >= HANDLE_GENERATION_MAX ? 1 : generation + 1;
}

fn bool is_pow2(usz value) {
    return value != 0 && (value & (value - 1)) == 0;
}

fn uint? resolve_descriptor_capacity(
    uint requested,
    uint default_capacity,
) {
    uint capacity = requested == 0 ? default_capacity : requested;
    if (capacity > gpu::MAX_DESCRIPTOR_SLOTS) return gpu::INVALID_ARGUMENT~;
    return capacity;
}
