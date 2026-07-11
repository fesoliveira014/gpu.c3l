module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Headless (lavapipe): descriptor heap slot bookkeeping through the public API.

const uint DESCRIPTOR_BUFFER_USAGE = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;

fn void descriptor_buffer_create_info_uses_exclusive_for_one_family() @test {
    uint[1] family_storage = { 3 };

    vk::BufferCreateInfo info = gpu::vk::make_descriptor_buffer_create_info(
        4096,
        family_storage[..],
    );

    assert(info.size == 4096, "descriptor-buffer layout size is preserved");
    assert((uint)info.usage == DESCRIPTOR_BUFFER_USAGE, "descriptor-buffer usage remains exact");
    assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "one consumer family remains exclusive");
    assert(info.queue_family_index_count == 0, "exclusive creation publishes no family count");
    assert(info.p_queue_family_indices == null, "exclusive creation publishes no family pointer");
}

fn void descriptor_buffer_create_info_uses_only_graphics_and_compute_families() @test {
    uint[2] family_storage = { 5, 2 };

    vk::BufferCreateInfo info = gpu::vk::make_descriptor_buffer_create_info(
        8192,
        family_storage[..],
    );

    assert(info.size == 8192, "descriptor-buffer layout size is preserved");
    assert((uint)info.usage == DESCRIPTOR_BUFFER_USAGE, "descriptor-buffer usage remains exact");
    assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "distinct consumers use concurrent sharing");
    assert(info.queue_family_index_count == 2, "concurrent creation publishes exactly two families");
    assert(info.p_queue_family_indices == &family_storage[0], "create info points into caller-owned storage");
    assert(info.p_queue_family_indices[0] == 5, "graphics family remains first");
    assert(info.p_queue_family_indices[1] == 2, "compute family remains second");
}

fn gpu::Device make_heap_test_device(uint tex_capacity, uint smp_capacity) {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .texture_descriptor_capacity = tex_capacity,
        .sampler_descriptor_capacity = smp_capacity,
        .application_name            = "gpu_descriptor_heap_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::TextureHandle make_heap_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_heap_view_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .width        = 16,
        .height       = 16,
        .mip_levels   = 5,
        .array_layers = 4,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn uint cached_texture_view_count(vk::VkDeviceState* state, gpu::TextureHandle texture) {
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn void texture_descriptor_round_trip() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "created index is valid");
    gpu::destroy_texture_descriptor(&device, index)!!;

    fault stale_fault = @catch(gpu::destroy_texture_descriptor(&device, index));
    assert(stale_fault == gpu::INVALID_HANDLE, "destroying a stale index faults INVALID_HANDLE");
}

fn void texture_descriptor_exhaustion() @test {
    gpu::Device device = make_heap_test_device(2, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;

    fault full_fault = @catch(gpu::create_texture_descriptor(&device, texture, null));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "allocating past capacity faults DESCRIPTOR_HEAP_FULL");
    assert(a.is_valid() && b.is_valid(), "existing indices stay valid after exhaustion");

    gpu::destroy_texture_descriptor(&device, a)!!;
    gpu::destroy_texture_descriptor(&device, b)!!;
}

fn void texture_descriptor_exhaustion_does_not_cache_failed_views() @test {
    gpu::Device device = make_heap_test_device(1, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureIndex occupied = gpu::create_texture_descriptor(&device, texture, null)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    assert(cached_texture_view_count(state, texture) == 0, "the fixture starts with an empty view cache");

    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::TextureViewDesc view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        fault full = @catch(gpu::create_texture_descriptor(&device, texture, &view));
        assert(full == gpu::DESCRIPTOR_HEAP_FULL, "an exhausted heap rejects the descriptor");
        assert(cached_texture_view_count(state, texture) == 0,
            "a rejected descriptor leaves the texture view cache unchanged");
    }

    gpu::destroy_texture_descriptor(&device, occupied)!!;
}

fn void texture_descriptor_deferred_reuse() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex first = gpu::create_texture_descriptor(&device, texture, null)!!;
    ushort first_slot = first.slot;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::destroy_texture_descriptor(&device, first)!!;

    gpu::TextureIndex during = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(during.slot != first_slot, "slot is not reused while its frame is in flight");

    gpu::end_frame(&frame_token)!!;
    // The frame fence signals GPU-side; idle so the timeline has passed.
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureIndex recycled = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(recycled.slot == first_slot, "retired slot is recycled once the frame timeline passes");
    assert(recycled.generation != first.generation, "recycled slot carries a new generation");

    fault recycled_fault = @catch(gpu::destroy_texture_descriptor(&device, first));
    assert(recycled_fault == gpu::INVALID_HANDLE, "the pre-recycle index is stale after reuse");

    gpu::destroy_texture_descriptor(&device, during)!!;
    gpu::destroy_texture_descriptor(&device, recycled)!!;
}

fn void texture_descriptor_stale_texture_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, texture)!!;

    fault texture_fault = @catch(gpu::create_texture_descriptor(&device, texture, null));
    assert(texture_fault == gpu::INVALID_HANDLE, "registering a destroyed texture faults INVALID_HANDLE");
}

fn void texture_descriptor_bad_view_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewDesc reinterpret = { .format = gpu::Format.R32_UINT };
    fault view_fault = @catch(gpu::create_texture_descriptor(&device, texture, &reinterpret));
    assert(view_fault == gpu::INVALID_ARGUMENT, "format-reinterpreting views fault INVALID_ARGUMENT");
}

fn void destroy_texture_with_live_descriptor_faults_and_recovers() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;

    fault in_use_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::RESOURCE_IN_USE, "destroying a texture with a live descriptor faults RESOURCE_IN_USE");

    gpu::TextureIndex second = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(second.is_valid(), "handle stays valid and usable after the faulted destroy");
    gpu::destroy_texture_descriptor(&device, second)!!;

    gpu::destroy_texture_descriptor(&device, index)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_retired_undrained_descriptor_succeeds() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::destroy_texture_descriptor(&device, index)!!;

    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_blocks_while_any_descriptor_live() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;

    gpu::destroy_texture_descriptor(&device, a)!!;
    fault still_in_use = @catch(gpu::destroy_texture(&device, texture));
    assert(still_in_use == gpu::RESOURCE_IN_USE, "one remaining live descriptor still blocks destroy_texture");

    gpu::destroy_texture_descriptor(&device, b)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_unaffected_by_unrelated_descriptor() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture_a = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture_a)!!;
    gpu::TextureHandle texture_b = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture_a, null)!!;
    defer gpu::destroy_texture_descriptor(&device, index)!!;

    gpu::destroy_texture(&device, texture_b)!!;
}

fn void texture_descriptors_batch_happy_path() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[3] indices;
    gpu::create_texture_descriptors(&device, descs[..], indices[..])!!;

    foreach (index : indices) assert(index.is_valid(), "every batch-created index is valid");
    assert(indices[0].slot != indices[1].slot && indices[1].slot != indices[2].slot && indices[0].slot != indices[2].slot,
        "batch-created indices occupy distinct slots");

    foreach (index : indices) gpu::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_batch_fills_capacity_exactly() @test {
    gpu::Device device = make_heap_test_device(3, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[3] indices;
    gpu::create_texture_descriptors(&device, descs[..], indices[..])!!;

    fault full_fault = @catch(gpu::create_texture_descriptor(&device, texture, null));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "a batch that exactly fills capacity leaves none spare — the heap count rose by exactly N");

    foreach (index : indices) gpu::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_batch_zeroed_view_is_default() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[1] descs = { { .texture = texture } };
    gpu::TextureIndex[1] indices;
    gpu::create_texture_descriptors(&device, descs[..], indices[..])!!;
    assert(indices[0].is_valid(), "a zero-initialized view collapses to the default view and creates successfully");

    gpu::destroy_texture_descriptor(&device, indices[0])!!;
}

fn void texture_descriptors_mismatched_out_len_faults() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[1] indices;
    fault mismatch_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(mismatch_fault == gpu::INVALID_ARGUMENT, "a mismatched out-indices length faults INVALID_ARGUMENT");
}

fn void texture_descriptors_empty_batch_is_no_op() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDescriptorDesc[] descs;
    gpu::TextureIndex[] indices;
    gpu::create_texture_descriptors(&device, descs, indices)!!;
}

fn void texture_descriptors_batch_preflight_faults_untouched() @test {
    gpu::Device device = make_heap_test_device(4, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex existing = gpu::create_texture_descriptor(&device, texture, null)!!;

    // Capacity 4, one slot already used: 3 remain; the batch asks for 4.
    gpu::TextureDescriptorDesc[4] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[4] indices;
    fault full_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "a batch exceeding remaining capacity faults DESCRIPTOR_HEAP_FULL");
    assert(existing.is_valid(), "the pre-existing index survives the failed pre-flight");

    // Untouched: the failed pre-flight consumed nothing, so the remaining
    // headroom (3 slots) is still there to fill exactly.
    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex c = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(a.is_valid() && b.is_valid() && c.is_valid(), "the failed batch left the full remaining headroom available");

    gpu::destroy_texture_descriptor(&device, existing)!!;
    gpu::destroy_texture_descriptor(&device, a)!!;
    gpu::destroy_texture_descriptor(&device, b)!!;
    gpu::destroy_texture_descriptor(&device, c)!!;
}

fn void texture_descriptors_failed_batch_restores_full_view_capacity() @test {
    gpu::Device device = make_heap_test_device(32, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::TextureViewDesc failed_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        gpu::TextureDescriptorDesc[2] descs = {
            { .texture = texture, .view = failed_view },
            { .texture = stale_texture },
        };
        gpu::TextureIndex[2] indices;
        fault batch_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
        assert(batch_fault == gpu::INVALID_HANDLE, "a later stale texture faults the batch");
        assert(cached_texture_view_count(state, texture) == 0,
            "a failed batch restores the empty view cache");
    }

    gpu::TextureIndex[vk::MAX_CACHED_VIEWS] live;
    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::TextureViewDesc valid_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = 2 + i / 4,
            .layer_count = 1,
        };
        live[i] = gpu::create_texture_descriptor(&device, texture, &valid_view)!!;
    }
    assert(cached_texture_view_count(state, texture) == vk::MAX_CACHED_VIEWS,
        "all cache entries remain available after repeated failed batches");

    foreach (index : live) gpu::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_failed_batch_restores_multiple_created_views() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;

    gpu::TextureViewDesc first_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::TextureViewDesc second_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureDescriptorDesc[3] descs = {
        { .texture = texture, .view = first_view },
        { .texture = texture, .view = second_view },
        { .texture = stale_texture },
    };
    gpu::TextureIndex[3] indices;
    fault batch_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE, "a fault after two new views rejects the batch");
    assert(cached_texture_view_count(state, texture) == 0,
        "rollback removes every view created by the failed transaction");
    assert(!slot.views[0].used && !slot.views[1].used,
        "rollback restores both cache cells to their prior unused state");
}

fn void texture_descriptors_failed_batch_preserves_existing_and_duplicate_views() @test {
    gpu::Device device = make_heap_test_device(16, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;

    gpu::TextureViewDesc existing_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::TextureIndex existing_index = gpu::create_texture_descriptor(&device, texture, &existing_view)!!;
    vk::CachedView existing_cache = slot.views[0];

    gpu::TextureViewDesc new_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureDescriptorDesc[4] descs = {
        { .texture = texture, .view = existing_view },
        { .texture = texture, .view = new_view },
        { .texture = texture, .view = new_view },
        { .texture = stale_texture },
    };
    gpu::TextureIndex[4] indices;
    fault batch_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE, "a fault after a duplicate new view rejects the batch");
    assert(cached_texture_view_count(state, texture) == 1,
        "rollback removes the one transaction-created view and preserves the existing view");
    assert(slot.views[0].used && slot.views[0].view == existing_cache.view,
        "the pre-existing cached Vulkan view remains owned by the texture");
    assert(slot.views[0].base_mip == existing_cache.base_mip
        && slot.views[0].mip_count == existing_cache.mip_count
        && slot.views[0].base_layer == existing_cache.base_layer
        && slot.views[0].layer_count == existing_cache.layer_count,
        "the pre-existing cache entry remains unchanged");

    gpu::destroy_texture_descriptor(&device, existing_index)!!;
}

fn void texture_descriptors_mid_batch_fault_preserves_descriptor_state() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex[7] occupied;
    for (uint i = 0; i < 7; i++) {
        occupied[i] = gpu::create_texture_descriptor(&device, texture, null)!!;
    }
    gpu::TextureIndex retired = occupied[0];
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::destroy_texture_descriptor(&device, retired)!!;
    gpu::end_frame(&frame_token)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::DescriptorHeapState* heap = &state.heap;
    assert(heap.texture_retire.count > 0, "the fixture has a ready undrained descriptor retire");

    vk::TextureDescriptorCell[8] cells_before;
    uint[8] free_before;
    for (uint i = 0; i < 8; i++) {
        cells_before[i] = heap.texture_cells[i];
        free_before[i] = heap.texture_free[i];
    }
    uint free_count_before = heap.texture_free_count;
    uint next_new_before = heap.texture_next_new;
    uint retire_head_before = heap.texture_retire.head;
    uint retire_count_before = heap.texture_retire.count;

    gpu::TextureViewDesc view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureDescriptorDesc[2] descs = {
        { .texture = texture, .view = view },
        { .texture = stale_texture },
    };
    gpu::TextureIndex[2] indices = {
        { .slot = 6, .generation = 111 },
        { .slot = 7, .generation = 222 },
    };
    gpu::TextureIndex[2] outputs_before = indices;
    fault batch_fault = @catch(gpu::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE, "a mid-batch destroyed-texture reference faults INVALID_HANDLE");

    assert(heap.texture_free_count == free_count_before, "failure preserves free-list count");
    assert(heap.texture_next_new == next_new_before, "failure preserves the never-used slot cursor");
    assert(heap.texture_retire.head == retire_head_before
        && heap.texture_retire.count == retire_count_before,
        "failure leaves ready retires undrained");
    for (uint i = 0; i < 8; i++) {
        vk::TextureDescriptorCell before = cells_before[i];
        vk::TextureDescriptorCell after = heap.texture_cells[i];
        assert(after.used == before.used
            && after.generation == before.generation
            && after.owner.index == before.owner.index
            && after.owner.generation == before.owner.generation,
            "failure preserves every descriptor cell and generation");
        assert(heap.texture_free[i] == free_before[i], "failure preserves free-list ordering and contents");
    }
    assert(indices[0].slot == outputs_before[0].slot
        && indices[0].generation == outputs_before[0].generation
        && indices[1].slot == outputs_before[1].slot
        && indices[1].generation == outputs_before[1].generation,
        "failure leaves output indices untouched");

    gpu::TextureDescriptorDesc[2] valid = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::TextureIndex[2] committed;
    gpu::create_texture_descriptors(&device, valid[..], committed[..])!!;
    assert(committed[0].is_valid() && committed[1].is_valid(),
        "a two-item batch commits using one never-used slot and one ready retire");
    assert(heap.texture_retire.count == 0, "successful commit drains the ready retire");
    foreach (index : committed) gpu::destroy_texture_descriptor(&device, index)!!;
    for (uint i = 1; i < 7; i++) {
        gpu::destroy_texture_descriptor(&device, occupied[i])!!;
    }
}

fn void texture_descriptors_batch_created_descriptor_blocks_texture_destroy() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureDescriptorDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[2] indices;
    gpu::create_texture_descriptors(&device, descs[..], indices[..])!!;

    fault in_use_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::RESOURCE_IN_USE, "a batch-created descriptor blocks destroy_texture the same as a single-created one");

    foreach (index : indices) gpu::destroy_texture_descriptor(&device, index)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void sampler_round_trip_and_exhaustion() @test {
    gpu::Device device = make_heap_test_device(8, 2);
    defer gpu::destroy_device(&device)!!;

    gpu::SamplerDesc desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex a = gpu::create_sampler(&device, &desc)!!;
    gpu::SamplerIndex b = gpu::create_sampler(&device, &desc)!!;
    assert(a.is_valid() && b.is_valid(), "sampler indices valid");

    fault full_fault = @catch(gpu::create_sampler(&device, &desc));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "allocating past sampler capacity faults DESCRIPTOR_HEAP_FULL");

    gpu::destroy_sampler(&device, a)!!;
    fault stale_fault = @catch(gpu::destroy_sampler(&device, a));
    assert(stale_fault == gpu::INVALID_HANDLE, "destroying a stale sampler index faults INVALID_HANDLE");

    gpu::destroy_sampler(&device, b)!!;
}

fn void leaked_descriptors_survive_teardown() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::create_texture_descriptor(&device, texture, null)!!;

    gpu::SamplerDesc desc = {};
    gpu::create_sampler(&device, &desc)!!;

    // Leaked on purpose: teardown must report and stay clean.
    gpu::destroy_device(&device)!!;
}

// Pure: heap_fits_limits takes no device, so these synthesize HeapLimits
// values directly rather than depending on what the driver reports.

fn void heap_fits_limits_indexing_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(fits, "texture and sampler capacities at the limit fit");
}

fn void heap_fits_limits_indexing_exceeds_per_stage_aggregate() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 199,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "aggregate per-stage limit binds when all per-type limits fit");
}

fn void heap_fits_limits_indexing_excludes_samplers_from_per_stage_aggregate() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(fits, "plain samplers do not count against the per-stage resource aggregate");
}
fn void heap_fits_limits_indexing_exceeds_all_pools_aggregate() @test {
    gpu::vk::HeapLimits limits = {

        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 249,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "aggregate all-pools limit binds when all per-type limits fit");
}
fn void heap_fits_limits_indexing_widens_aggregate_arithmetic() @test {

    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 0x8000_0000,
        smp_capacity: 0,
        limits:       limits,
    );
    assert(!fits, "aggregate arithmetic does not wrap at uint width");
}

fn void heap_fits_limits_indexing_exceeds_sampled_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 99,
        .max_storage_images = 100,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the sampled-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_storage_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 99,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the storage-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_samplers() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 100,
        .max_samplers       = 49,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 49,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "sampler capacity over the sampler limit is rejected");
}

fn void heap_fits_limits_buffer_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size at the limit fits");
}

fn void heap_fits_limits_buffer_exceeds_resource_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 2048,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the resource descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_sampler_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 2048,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the sampler descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 2048,
        .sampler_descriptor_address_space_size  = 2048,
.descriptor_buffer_address_space_size = 2048,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the descriptor buffer address space is rejected");
}

fn void heap_fits_limits_indexing_exceeds_per_stage() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 99,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "per-stage limit binds even when the set-level limit fits");
}

fn void heap_fits_limits_buffer_exceeds_per_type_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = 4096,
        .max_sampler_descriptor_buffer_range     = 4096,
        .resource_descriptor_address_space_size  = 4096,
        .sampler_descriptor_address_space_size   = 2048,
        .descriptor_buffer_address_space_size    = 4096,
        .layout_size                             = 4096,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    );
    assert(!fits, "per-type address space binds even when the combined budget fits");
}

// Integration: a capacity above the library default, but still well within
// lavapipe's reported device limits, creates successfully.
fn void descriptor_heap_raised_capacity_creates() @test {
    gpu::Device device = make_heap_test_device(8192, 512);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    assert(device.caps.max_texture_descriptors == 8192, "caps report the created texture capacity");
    assert(device.caps.max_sampler_descriptors == 512, "caps report the created sampler capacity");
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "raised-capacity heap creates and serves a descriptor");
    gpu::destroy_texture_descriptor(&device, index)!!;
}
