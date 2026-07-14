module gpu::vk;

import gpu @public;

fn void pow2_helper() @test {
    assert(is_pow2(1) && is_pow2(16) && is_pow2(4096), "powers of two accepted");
    assert(!is_pow2(0) && !is_pow2(3) && !is_pow2(48), "zero and non-powers rejected");
}

fn void descriptor_capacity_resolution() @test {
    uint defaulted = resolve_descriptor_capacity(0, gpu::DEFAULT_TEXTURE_DESCRIPTORS)!!;
    assert(defaulted == gpu::DEFAULT_TEXTURE_DESCRIPTORS, "zero requests the default capacity");

    uint explicit_capacity = resolve_descriptor_capacity(16, gpu::DEFAULT_TEXTURE_DESCRIPTORS)!!;
    assert(explicit_capacity == 16, "explicit capacity is honored");

    uint at_ceiling = resolve_descriptor_capacity(gpu::MAX_DESCRIPTOR_SLOTS, gpu::DEFAULT_TEXTURE_DESCRIPTORS)!!;
    assert(at_ceiling == gpu::MAX_DESCRIPTOR_SLOTS, "the packed-slot ceiling itself is allowed");

    fault ceiling_fault = @catch(resolve_descriptor_capacity(
        gpu::MAX_DESCRIPTOR_SLOTS + 1,
        gpu::DEFAULT_TEXTURE_DESCRIPTORS,
    ));
    assert(ceiling_fault == gpu::INVALID_ARGUMENT,
        "capacity above the packed-slot ceiling faults INVALID_ARGUMENT");
}

fn void strict_indexing_heap_capacity_support_matches_creation_limits() @test {
    gpu::vk::IndexingHeapLimits limits = {
        .max_sampled_images           = gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        .max_storage_images           = gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        .max_samplers                 = gpu::DEFAULT_SAMPLER_DESCRIPTORS,
        .max_per_stage_sampled_images = gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        .max_per_stage_storage_images = gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        .max_per_stage_samplers       = gpu::DEFAULT_SAMPLER_DESCRIPTORS,
        .max_per_stage_resources      = gpu::DEFAULT_TEXTURE_DESCRIPTORS * 2,
        .max_all_pools_descriptors    = gpu::DEFAULT_TEXTURE_DESCRIPTORS * 2
            + gpu::DEFAULT_SAMPLER_DESCRIPTORS,
    };
    assert(gpu::vk::indexing_heap_fits_limits(
        tex_capacity: gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        smp_capacity: gpu::DEFAULT_SAMPLER_DESCRIPTORS,
        limits:       limits,
    ));

    limits.max_sampled_images--;
    assert(!gpu::vk::indexing_heap_fits_limits(
        tex_capacity: gpu::DEFAULT_TEXTURE_DESCRIPTORS,
        smp_capacity: gpu::DEFAULT_SAMPLER_DESCRIPTORS,
        limits:       limits,
    ));
}

fn void resource_handle_generation_wraps_to_public_width() @test {
    assert(next_handle_generation(HANDLE_GENERATION_MAX) == 1,
        "resource handle generation wraps before exceeding its 24-bit field");
}
