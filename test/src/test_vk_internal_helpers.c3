module gpu::internal::vk;

import gpu @public;

fn void pow2_helper() @test {
    assert(is_pow2(1) && is_pow2(16) && is_pow2(4096), "powers of two accepted");
    assert(!is_pow2(0) && !is_pow2(3) && !is_pow2(48), "zero and non-powers rejected");
}

fn void descriptor_capacity_resolution() @test {
    uint defaulted = resolve_descriptor_capacity(0, gpu::DEFAULT_TEXTURE_HEAP_CAPACITY)!!;
    assert(defaulted == gpu::DEFAULT_TEXTURE_HEAP_CAPACITY, "zero requests the default capacity");

    uint explicit_capacity = resolve_descriptor_capacity(16, gpu::DEFAULT_TEXTURE_HEAP_CAPACITY)!!;
    assert(explicit_capacity == 16, "explicit capacity is honored");

    uint at_ceiling = resolve_descriptor_capacity(gpu::MAX_SHADER_HEAP_CAPACITY, gpu::DEFAULT_TEXTURE_HEAP_CAPACITY)!!;
    assert(at_ceiling == gpu::MAX_SHADER_HEAP_CAPACITY, "the descriptor heap capacity limit itself is allowed");

    fault ceiling_fault = @catch(resolve_descriptor_capacity(
        gpu::MAX_SHADER_HEAP_CAPACITY + 1,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    ));
    assert(ceiling_fault == gpu::INVALID_ARGUMENT,
        "capacity above the descriptor heap limit faults INVALID_ARGUMENT");
}

fn IndexingHeapLimits exact_indexing_heap_limits(
    uint texture_capacity,
    uint sampler_capacity,
) {
    return {
        .max_sampled_images           = texture_capacity * 2,
        .max_storage_images           = texture_capacity * 2,
        .max_samplers                 = sampler_capacity,
        .max_per_stage_sampled_images = texture_capacity * 2,
        .max_per_stage_storage_images = texture_capacity * 2,
        .max_per_stage_samplers       = sampler_capacity,
        .max_per_stage_resources      = texture_capacity * 4,
        .max_all_pools_descriptors    = texture_capacity * 4
            + sampler_capacity,
    };
}

fn void indexing_heap_limit_failures_are_exact() @test {
    uint texture_capacity = 100;
    uint sampler_capacity = 50;
    IndexingHeapLimits limits = exact_indexing_heap_limits(
        texture_capacity,
        sampler_capacity,
    );
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.NONE);

    limits.max_sampled_images--;
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.SAMPLED_IMAGES);
    limits = exact_indexing_heap_limits(texture_capacity, sampler_capacity);

    limits.max_storage_images--;
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.STORAGE_IMAGES);
    limits = exact_indexing_heap_limits(texture_capacity, sampler_capacity);

    limits.max_samplers--;
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.SAMPLERS);
    limits = exact_indexing_heap_limits(texture_capacity, sampler_capacity);

    limits.max_per_stage_resources--;
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.PER_STAGE_RESOURCES);
    limits = exact_indexing_heap_limits(texture_capacity, sampler_capacity);

    limits.max_all_pools_descriptors--;
    assert(indexing_heap_limit_failure(
        texture_capacity,
        sampler_capacity,
        limits,
    ) == IndexingHeapLimitFailure.ALL_POOLS_DESCRIPTORS);
}

fn void indexing_heap_limit_arithmetic_widens() @test {
    IndexingHeapLimits limits = {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
    assert(indexing_heap_limit_failure(
        0x4000_0000,
        0,
        limits,
    ) == IndexingHeapLimitFailure.PER_STAGE_RESOURCES);
}

fn void shader_heap_hard_ceiling_is_separate_from_native_limits() @test {
    uint above_hard_ceiling = gpu::MAX_SHADER_HEAP_CAPACITY + 1;
    IndexingHeapLimits limits = {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
    assert(indexing_heap_limit_failure(
        above_hard_ceiling,
        0,
        limits,
    ) == IndexingHeapLimitFailure.NONE);
    assert(@catch(resolve_descriptor_capacity(
        above_hard_ceiling,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    )) == gpu::INVALID_ARGUMENT);
}

fn void resource_handle_generation_exhaustion_has_no_successor() @test {
    assert(next_handle_generation(HANDLE_GENERATION_MAX - 1)!! == HANDLE_GENERATION_MAX,
        "generation can advance to the final public value");
    fault exhausted = @catch(next_handle_generation(HANDLE_GENERATION_MAX));
    assert(exhausted == gpu::SLOT_TABLE_FULL,
        "exhausted generation has no reusable successor");
}
