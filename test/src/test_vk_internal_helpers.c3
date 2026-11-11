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
        .max_sampled_images           = texture_capacity * 3,
        .max_storage_images           = texture_capacity * 2,
        .max_samplers                 = sampler_capacity,
        .max_per_stage_sampled_images = texture_capacity * 3,
        .max_per_stage_storage_images = texture_capacity * 2,
        .max_per_stage_samplers       = sampler_capacity,
        .max_per_stage_resources      = texture_capacity * 5,
        .max_all_pools_descriptors    = texture_capacity * 5
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

fn gpu::TextureHandle pending_test_handle(uint index) {
    return { .owner = 7, .index = index, .generation = 1 };
}

fn uint pending_test_outstanding(PendingTextureInits* list, uint role) {
    return list.outstanding[role].load();
}

fn bool pending_test_bucket_holds(PendingTextureInits* list, uint bucket, gpu::TextureHandle handle) {
    for (uint i = list.starts[bucket]; i < list.starts[bucket + 1]; i++) {
        if (list.handles[i] == handle) return true;
    }
    return false;
}

fn void pending_init_buckets_follow_access() @test {
    assert(pending_init_bucket({ .graphics }) == 0);
    assert(pending_init_bucket({ .compute }) == 1);
    assert(pending_init_bucket({ .graphics, .compute }) == 2);
    assert(pending_init_bucket({ .transfer }) == 3);
    assert(pending_init_bucket({ .graphics, .compute, .transfer }) == PENDING_INIT_BUCKETS - 1);
    assert(pending_init_bucket({}) == PENDING_INIT_BUCKETS - 1, "an empty access counts as every role");
    for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
        assert(pending_init_bucket(pending_init_bucket_roles(bucket)) == bucket);
    }
    assert(queue_roles_overlap({ .graphics, .compute }, { .compute, .transfer }));
    assert(!queue_roles_overlap({ .graphics, .compute }, { .transfer }));
    assert(!queue_roles_overlap({}, { .graphics, .compute, .transfer }));
}

fn void pending_inits_take_only_compatible_buckets() @test {
    gpu::TextureHandle[8] storage;
    PendingTextureInits list = { .handles = storage[..] };
    gpu::TextureHandle unshared = pending_test_handle(1);
    gpu::TextureHandle shared = pending_test_handle(2);
    gpu::TextureHandle transfer_only = pending_test_handle(3);
    assert(list.push(unshared, { .graphics, .compute }));
    assert(list.push(shared, { .graphics, .transfer }));
    assert(list.push(transfer_only, { .transfer }));
    assert(list.count() == 3);
    assert(pending_test_outstanding(&list, 0) == 2
        && pending_test_outstanding(&list, 1) == 1
        && pending_test_outstanding(&list, 2) == 2);

    gpu::TextureHandle[8] storage_taken;
    PendingInitTake take = { .handles = storage_taken[..] };
    assert(list.compatible_count({ .transfer }) == 2);
    assert(list.take({ .transfer }, &take) == 2, "a transfer queue takes the two transfer-accessible textures");
    gpu::TextureHandle[] taken = take.handles;
    assert(taken.len == 2, "the take shrinks to what it took");
    assert(taken[0] == transfer_only && taken[1] == shared, "a take lists its handles in bucket order");
    assert(take.bucket_counts[pending_init_bucket({ .transfer })] == 1
        && take.bucket_counts[pending_init_bucket({ .graphics, .transfer })] == 1);
    assert((uint)take.roles() == (uint)(gpu::QueueRoles){ .graphics, .transfer },
        "the take's roles are the union of its buckets");
    uint unshared_bucket = pending_init_bucket({ .graphics, .compute });
    assert(list.count() == 1 && pending_test_bucket_holds(&list, unshared_bucket, unshared),
        "the unshared texture stays queued");
    assert(list.compatible_count({ .transfer }) == 0);
    assert(list.has_outstanding({ .transfer }), "taken textures stay outstanding until settled");

    list.settle(&take);
    assert(!list.has_outstanding({ .transfer }), "settling ends the transfer role's count");
    assert(list.has_outstanding({ .graphics }) && list.has_outstanding({ .compute }));

    take = { .handles = storage_taken[..] };
    assert(list.take({ .compute }, &take) == 1 && take.handles[0] == unshared);
    assert(list.requeue(unshared, { .graphics, .compute }), "a failed submit puts the texture back");
    assert(list.count() == 1 && pending_test_outstanding(&list, 0) == 1 && pending_test_outstanding(&list, 1) == 1,
        "a requeue does not count the texture twice");
    assert(list.remove(unshared, { .graphics, .compute }), "destruction removes a queued texture");
    assert(!list.remove(unshared, { .graphics, .compute }), "removing twice finds nothing");
    assert(list.count() == 0 && !list.has_outstanding({ .graphics, .compute, .transfer }));
}

fn void pending_inits_full_list_drops_and_settles() @test {
    gpu::TextureHandle[2] storage;
    PendingTextureInits list = { .handles = storage[..] };
    assert(list.push(pending_test_handle(1), { .graphics }));
    assert(list.push(pending_test_handle(2), { .graphics }));
    assert(!list.push(pending_test_handle(3), { .graphics }), "a full list drops the texture");
    assert(pending_test_outstanding(&list, 0) == 2, "a dropped push is not counted");

    gpu::TextureHandle[2] taken;
    PendingInitTake take = { .handles = taken[..] };
    assert(list.take({ .graphics }, &take) == 2);
    assert(list.push(pending_test_handle(4), { .graphics }));
    assert(list.push(pending_test_handle(5), { .graphics }));
    assert(list.requeue(take.handles[0], { .graphics }) == false, "a requeue into a full list drops the texture");
    assert(pending_test_outstanding(&list, 0) == 3, "the dropped requeue is settled");
}

fn void pending_inits_match_a_reference_model() @test {
    const uint CAPACITY = 24;
    gpu::TextureHandle[CAPACITY] storage;
    PendingTextureInits list = { .handles = storage[..] };
    gpu::QueueRoles[CAPACITY * 4] reference_access;
    bool[CAPACITY * 4] queued;
    uint[QUEUE_ROLE_SLOTS] outstanding;
    uint next_index;
    uint random = 12345;
    for (uint step = 0; step < 4000; step++) {
        random = random * 1103515245 + 12345;
        uint choice = (random >> 16) % 4;
        if (choice <= 1 && next_index < reference_access.len) {
            gpu::QueueRoles access = pending_init_bucket_roles((random >> 8) % PENDING_INIT_BUCKETS);
            uint index = next_index++;
            bool pushed = list.push(pending_test_handle(index), access);
            uint queued_count;
            foreach (state : queued) {
                if (state) queued_count++;
            }
            assert(pushed == (queued_count < CAPACITY), "a push succeeds exactly while there is room");
            if (pushed) {
                queued[index] = true;
                reference_access[index] = access;
                for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
                    if (((uint)access & (1u << role)) != 0) outstanding[role]++;
                }
            }
        } else if (choice == 2) {
            uint index = next_index == 0 ? 0 : (random >> 4) % next_index;
            bool removed = list.remove(pending_test_handle(index), reference_access[index]);
            assert(removed == queued[index], "remove finds exactly the queued textures");
            if (removed) {
                queued[index] = false;
                for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
                    if (((uint)reference_access[index] & (1u << role)) != 0) outstanding[role]--;
                }
            }
        } else {
            gpu::QueueRoles roles = (gpu::QueueRoles)(1u << ((random >> 6) % QUEUE_ROLE_SLOTS));
            gpu::TextureHandle[CAPACITY] taken;
            PendingInitTake take = { .handles = taken[..] };
            uint expected = list.compatible_count(roles);
            uint count = list.take(roles, &take);
            assert(count == expected && take.handles.len == count);
            uint bucket = 0;
            uint left_in_bucket = take.bucket_counts[0];
            for (uint i = 0; i < count; i++) {
                while (left_in_bucket == 0) left_in_bucket = take.bucket_counts[++bucket];
                left_in_bucket--;
                uint index = take.handles[i].index;
                assert(queued[index] && queue_roles_overlap(reference_access[index], roles),
                    "take returns only queued textures the roles may access");
                assert(pending_init_bucket(reference_access[index]) == bucket,
                    "the bucket counts describe the handles in order");
                queued[index] = false;
                for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
                    if (((uint)reference_access[index] & (1u << role)) != 0) outstanding[role]--;
                }
            }
            list.settle(&take);
            assert(list.compatible_count(roles) == 0);
        }
        uint live;
        foreach (index, state : queued) {
            if (!state) continue;
            live++;
            assert(pending_test_bucket_holds(
                &list,
                pending_init_bucket(reference_access[index]),
                pending_test_handle((uint)index),
            ), "every queued texture sits in its access bucket");
        }
        assert(list.count() == live, "the buckets hold exactly the queued textures");
        for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
            assert(list.starts[bucket] <= list.starts[bucket + 1]);
        }
        for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
            assert(pending_test_outstanding(&list, role) == outstanding[role]);
        }
    }
}

fn void texture_init_targets_follow_identity_roles() @test {
    gpu::QueueRoles[3] distinct = { { .graphics }, { .compute }, { .transfer } };
    assert(texture_init_targets(distinct[..], 0, { .graphics, .transfer }) == 0b100,
        "a graphics init of a graphics and transfer texture orders the transfer queue only");
    assert(texture_init_targets(distinct[..], 0, { .graphics, .compute }) == 0b010);
    assert(texture_init_targets(distinct[..], 0, { .graphics }) == 0, "an unshared texture orders no other queue");
    assert(texture_init_targets(distinct[..], 2, { .graphics, .compute, .transfer }) == 0b011);

    gpu::QueueRoles[2] aliased = { { .graphics, .compute }, { .transfer } };
    assert(texture_init_targets(aliased[..], 0, { .compute }) == 0,
        "a role aliased onto the same queue needs no cross-queue wait");
    assert(texture_init_targets(aliased[..], 1, { .compute, .transfer }) == 0b01);
}

fn void texture_init_wait_rules() @test {
    assert(texture_init_wait_needed(5, 0, 0), "an unretired initialization is waited on");
    assert(!texture_init_wait_needed(0, 0, 0), "no initialization, no wait");
    assert(!texture_init_wait_needed(5, 5, 0), "a retired initialization needs no wait");
    assert(!texture_init_wait_needed(5, 0, 5), "an application wait at the same point covers it");
    assert(!texture_init_wait_needed(5, 0, 9), "an application wait at a later point covers it");
    assert(texture_init_wait_needed(5, 4, 4), "earlier points do not cover it");
}
