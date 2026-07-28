module gpu::internal::vk;

import gpu @public;
import vk;
import std::atomic;

struct TimestampPoolProbe {
    vk::Result    create_result;
    vk::Result    read_result;
    uint          create_calls;
    uint          destroy_calls;
    uint          read_calls;
    vk::QueryType query_type;
    uint          query_count;
    vk::QueryPool destroyed;
    uint          first;
    uint          count;
    sz            data_size;
    vk::DeviceSize stride;
    vk::QueryResultFlags flags;
}

fn vk::Result probe_create_timestamp_pool(
    void* user_data,
    vk::Device device,
    vk::QueryPoolCreateInfo* info,
    vk::QueryPool* out_pool,
) {
    (void)device;
    TimestampPoolProbe* probe = (TimestampPoolProbe*)user_data;
    probe.create_calls++;
    probe.query_type = info.query_type;
    probe.query_count = info.query_count;
    if (probe.create_result == vk::Result.SUCCESS) {
        *out_pool = (vk::QueryPool)(uptr)(0x100 + probe.create_calls);
    }
    return probe.create_result;
}

fn void probe_destroy_timestamp_pool(
    void* user_data,
    vk::Device device,
    vk::QueryPool pool,
) {
    (void)device;
    TimestampPoolProbe* probe = (TimestampPoolProbe*)user_data;
    probe.destroy_calls++;
    probe.destroyed = pool;
}

fn vk::Result probe_read_timestamps(
    void* user_data,
    vk::Device device,
    vk::QueryPool pool,
    uint first,
    uint count,
    sz data_size,
    void* data,
    vk::DeviceSize stride,
    vk::QueryResultFlags flags,
) {
    (void)device;
    (void)pool;
    TimestampPoolProbe* probe = (TimestampPoolProbe*)user_data;
    probe.read_calls++;
    probe.first = first;
    probe.count = count;
    probe.data_size = data_size;
    probe.stride = stride;
    probe.flags = flags;
    if (probe.read_result == vk::Result.SUCCESS) {
        ulong* values = (ulong*)data;
        for (uint i = 0; i < count; i++) values[i] = 11 + (ulong)i * 6;
    }
    return probe.read_result;
}

fn TimestampPoolNativeOps timestamp_pool_probe_ops(
    TimestampPoolProbe* probe,
) {
    return {
        .user_data = probe,
        .create     = &probe_create_timestamp_pool,
        .destroy    = &probe_destroy_timestamp_pool,
        .read       = &probe_read_timestamps,
    };
}

fn void timestamp_pool_table_is_fixed_and_generation_checked() @test {
    assert(MAX_TIMESTAMP_POOLS == 256,
        "timestamp pool storage has the fixed backend capacity");
    TimestampPoolCell[1] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 41,
    };

    gpu::TimestampPoolHandle first = table.alloc({
        .pool     = (vk::QueryPool)(uptr)1,
        .capacity = 4,
    })!!;
    assert(first.owner == table.owner && first.index == 0,
        "allocation publishes the table owner and slot");
    assert(table.get(first)!!.capacity == 4,
        "the live handle resolves stored metadata");
    table.free(first)!!;
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "a freed generation is stale");

    gpu::TimestampPoolHandle second = table.alloc({
        .pool     = (vk::QueryPool)(uptr)2,
        .capacity = 8,
    })!!;
    assert(second.index == first.index
        && second.generation != first.generation,
        "slot reuse advances the public generation");
}

fn void timestamp_pool_table_rejects_foreign_stale_and_full_handles() @test {
    TimestampPoolCell[1] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 73,
    };
    gpu::TimestampPoolHandle live = table.alloc({})!!;
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "live capacity faults deterministically");

    gpu::TimestampPoolHandle foreign = live;
    foreign.owner++;
    assert(@catch(table.get(foreign)) == gpu::INVALID_HANDLE,
        "foreign owners cannot alias a local slot");
    gpu::TimestampPoolHandle stale = live;
    stale.generation++;
    assert(@catch(table.get(stale)) == gpu::INVALID_HANDLE,
        "generation mismatch cannot resolve a live slot");
}

fn void timestamp_pool_generation_exhaustion_retires_slot() @test {
    TimestampPoolCell[1] cells;
    cells[0].generation = HANDLE_GENERATION_MAX - 1;
    TimestampPoolTable table = {
        .slots     = cells[..],
        .owner     = 91,
        .free_head = 1,
        .next_new  = 1,
    };

    gpu::TimestampPoolHandle final = table.alloc({})!!;
    assert(final.generation == HANDLE_GENERATION_MAX,
        "the final timestamp generation can be published once");
    table.free(final)!!;
    assert(cells[0].generation == HANDLE_GENERATION_MAX
        && table.free_head == TIMESTAMP_POOL_FREE_NONE,
        "an exhausted timestamp slot is permanently retired");
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "an exhausted generation has no successor");
}

fn void timestamp_range_and_result_size_are_checked() @test {
    validate_timestamp_range(8, 0, 8)!!;
    validate_timestamp_range(8, 7, 1)!!;
    assert(@catch(validate_timestamp_range(8, 0, 0))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(validate_timestamp_range(8, 8, 1))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(validate_timestamp_range(8, uint::max, 2))
        == gpu::INVALID_ARGUMENT);

    assert(timestamp_result_bytes(3)!! == 3 * ulong::size);
    assert(@catch(timestamp_result_bytes(0)) == gpu::INVALID_ARGUMENT);
}

fn void timestamp_pool_creation_is_transactional_and_copies_name() @test {
    TimestampPoolCell[1] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 17,
    };
    TimestampPoolProbe probe;
    TimestampPoolNativeOps ops = timestamp_pool_probe_ops(&probe);
    char[16] name;
    name[0] = 'f';
    name[1] = 'r';
    name[2] = 'a';
    name[3] = 'm';
    name[4] = 'e';
    name[5] = 0;

    gpu::TimestampPoolHandle handle = create_timestamp_pool_with(
        table:      &table,
        device:     (vk::Device)null,
        capacity:   4,
        debug_name: (ZString)&name[0],
        ops:        &ops,
    )!!;
    name[0] = 'x';
    TimestampPoolSlot* slot = table.get(handle)!!;
    assert(slot.capacity == 4
        && ((ZString)&slot.debug_name[0]).str_view() == "frame",
        "creation publishes copied pool metadata");
    assert(probe.create_calls == 1
        && probe.query_type == vk::QueryType.TIMESTAMP
        && probe.query_count == 4,
        "creation requests an exact timestamp query pool");

    fault full = @catch(create_timestamp_pool_with(
        table:      &table,
        device:     (vk::Device)null,
        capacity:   2,
        debug_name: null,
        ops:        &ops,
    ));
    assert(full == gpu::SLOT_TABLE_FULL,
        "publication exhaustion preserves the table fault");
    assert(probe.create_calls == 2 && probe.destroy_calls == 1,
        "failed publication destroys the unpublished native pool");
}

fn void timestamp_pool_destruction_honors_tracked_references() @test {
    TimestampPoolCell[1] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 23,
    };
    TimestampPoolProbe probe;
    TimestampPoolNativeOps ops = timestamp_pool_probe_ops(&probe);
    gpu::TimestampPoolHandle handle = create_timestamp_pool_with(
        table:      &table,
        device:     (vk::Device)null,
        capacity:   2,
        debug_name: null,
        ops:        &ops,
    )!!;
    TimestampPoolSlot* slot = table.get(handle)!!;
    slot.tracked_command_references.store(1, AtomicOrdering.RELAXED);

    assert(@catch(destroy_timestamp_pool_with(
        &table,
        (vk::Device)null,
        handle,
        &ops,
    )) == gpu::RESOURCE_IN_USE);
    assert(probe.destroy_calls == 0 && table.get(handle)!! == slot,
        "an in-use pool remains live and native");

    slot.tracked_command_references.store(0, AtomicOrdering.RELAXED);
    vk::QueryPool native = slot.pool;
    destroy_timestamp_pool_with(
        &table,
        (vk::Device)null,
        handle,
        &ops,
    )!!;
    assert(probe.destroy_calls == 1 && probe.destroyed == native,
        "quiescent destruction releases the exact native pool");
    assert(@catch(table.get(handle)) == gpu::INVALID_HANDLE,
        "destruction retires the public generation");
}

fn void timestamp_pool_read_is_direct_and_nonblocking() @test {
    TimestampPoolCell[1] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 29,
    };
    TimestampPoolProbe probe;
    TimestampPoolNativeOps ops = timestamp_pool_probe_ops(&probe);
    gpu::TimestampPoolHandle handle = create_timestamp_pool_with(
        table:      &table,
        device:     (vk::Device)null,
        capacity:   4,
        debug_name: null,
        ops:        &ops,
    )!!;
    ulong[2] values;

    probe.read_result = vk::Result.NOT_READY;
    assert(@catch(read_timestamps_with(
        table:  &table,
        device: (vk::Device)null,
        pool:   handle,
        first:  1,
        count:  2,
        out:    values[..],
        ops:    &ops,
    )) == gpu::DEVICE_BUSY);
    assert(probe.first == 1
        && probe.count == 2
        && probe.data_size == 2 * ulong::size
        && probe.stride == ulong::size
        && probe.flags == (vk::QueryResultFlags)
            vk::QueryResultFlagBits.RESULT_64_BIT,
        "host reads request packed 64-bit results without WAIT");

    probe.read_result = vk::Result.SUCCESS;
    read_timestamps_with(
        table:  &table,
        device: (vk::Device)null,
        pool:   handle,
        first:  1,
        count:  2,
        out:    values[..],
        ops:    &ops,
    )!!;
    assert(values[0] == 11 && values[1] == 17,
        "successful reads write directly to caller storage");
}

fn void timestamp_pool_teardown_destroys_every_live_native_pool() @test {
    TimestampPoolCell[3] cells;
    TimestampPoolTable table = {
        .slots = cells[..],
        .owner = 31,
    };
    TimestampPoolProbe probe;
    TimestampPoolNativeOps ops = timestamp_pool_probe_ops(&probe);
    create_timestamp_pool_with(
        &table,
        (vk::Device)null,
        2,
        null,
        &ops,
    )!!;
    gpu::TimestampPoolHandle freed = create_timestamp_pool_with(
        &table,
        (vk::Device)null,
        2,
        null,
        &ops,
    )!!;
    create_timestamp_pool_with(
        &table,
        (vk::Device)null,
        2,
        null,
        &ops,
    )!!;
    destroy_timestamp_pool_with(
        &table,
        (vk::Device)null,
        freed,
        &ops,
    )!!;

    uint leaked = destroy_timestamp_pool_table_with(
        &table,
        (vk::Device)null,
        &ops,
    );
    assert(leaked == 2 && probe.destroy_calls == 3,
        "teardown reports and destroys exactly the live pools");
    assert(table.next_new == 0
        && table.free_head == TIMESTAMP_POOL_FREE_NONE,
        "teardown resets table bookkeeping");
}
