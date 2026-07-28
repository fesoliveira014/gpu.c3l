module gpu::internal::vk;

import vk;

fn void descriptor_heap_layout_and_pool_plans_are_exact() @test {
    uint texture_capacity = 100;
    uint sampler_capacity = 50;
    DescriptorHeapLayoutPlan layout = descriptor_heap_layout_plan(
        texture_capacity,
        sampler_capacity,
    );
    uint expected_flags =
        (uint)vk::DescriptorBindingFlagBits.UPDATE_AFTER_BIND_BIT
        | (uint)vk::DescriptorBindingFlagBits.UPDATE_UNUSED_WHILE_PENDING_BIT
        | (uint)vk::DescriptorBindingFlagBits.PARTIALLY_BOUND_BIT;
    uint[HEAP_BINDING_COUNT] expected_bindings = {
        HEAP_BINDING_SAMPLED_2D,
        HEAP_BINDING_STORAGE_2D,
        HEAP_BINDING_SAMPLERS,
        HEAP_BINDING_SAMPLED_3D,
        HEAP_BINDING_STORAGE_3D,
    };
    vk::DescriptorType[HEAP_BINDING_COUNT] expected_types = {
        vk::DescriptorType.SAMPLED_IMAGE,
        vk::DescriptorType.STORAGE_IMAGE,
        vk::DescriptorType.SAMPLER,
        vk::DescriptorType.SAMPLED_IMAGE,
        vk::DescriptorType.STORAGE_IMAGE,
    };
    uint[HEAP_BINDING_COUNT] expected_counts = {
        texture_capacity,
        texture_capacity,
        sampler_capacity,
        texture_capacity,
        texture_capacity,
    };
    for (uint i = 0; i < HEAP_BINDING_COUNT; i++) {
        assert(layout.bindings[i].binding == expected_bindings[i],
            "heap layout binding number is exact");
        assert(layout.bindings[i].descriptor_type == expected_types[i],
            "heap layout descriptor type is exact");
        assert(layout.bindings[i].descriptor_count == expected_counts[i],
            "heap layout logical capacity is exact");
        assert(layout.bindings[i].stage_flags == vk::ShaderStageFlagBits.ALL,
            "every heap binding is visible to all shader stages");
        assert((uint)layout.binding_flags[i] == expected_flags,
            "every heap binding retains update-after-bind flags");
    }

    DescriptorHeapPoolPlan pool = descriptor_heap_pool_plan(
        texture_capacity,
        sampler_capacity,
    );
    assert(pool.sizes[0].type == vk::DescriptorType.SAMPLED_IMAGE
            && pool.sizes[0].descriptor_count == texture_capacity * 2,
        "sampled pool capacity covers both dimensional bindings");
    assert(pool.sizes[1].type == vk::DescriptorType.STORAGE_IMAGE
            && pool.sizes[1].descriptor_count == texture_capacity * 2,
        "storage pool capacity covers both dimensional bindings");
    assert(pool.sizes[2].type == vk::DescriptorType.SAMPLER
            && pool.sizes[2].descriptor_count == sampler_capacity,
        "sampler pool capacity remains independent");

    HeapDescriptorTotals totals = descriptor_heap_totals(
        texture_capacity,
        sampler_capacity,
    );
    assert(totals.sampled_images == (ulong)texture_capacity * 2
            && totals.storage_images == (ulong)texture_capacity * 2,
        "native image totals cover both dimensional bindings");
    assert(totals.per_stage_resources
            == (ulong)texture_capacity * 4 + (ulong)sampler_capacity,
        "per-stage total includes all image bindings and samplers");
    assert(totals.all_pools_descriptors
            == (ulong)texture_capacity * 4 + (ulong)sampler_capacity,
        "pool total includes all image bindings and samplers");
}
