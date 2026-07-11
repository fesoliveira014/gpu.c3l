module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

fn void staging_buffer_create_info_covers_selected_families() @test {
    uint[3][3] topologies = {
        { 3, 3, 3 },
        { 5, 2, 5 },
        { 7, 4, 1 },
    };
    uint[3] expected_counts = { 1, 2, 3 };
    foreach (index, topology : topologies) {
        vk::VkDeviceState state = {
            .graphics_family = topology[0],
            .compute_family  = topology[1],
            .transfer_family = topology[2],
        };
        uint[3] families;
        uint family_count = gpu::vk::used_queue_families(&state, families[..]);
        gpu::BufferDesc desc = gpu::vk::staging_arena_desc(4096);
        vk::BufferCreateInfo info = gpu::vk::make_buffer_create_info(&desc, families[:family_count])!!;
        uint expected_count = expected_counts[index];
        assert(family_count == expected_count, "selected family count is exact");
        assert(info.size == 4096, "staging size is preserved");
        assert((uint)info.usage == (uint)vk::BufferUsageFlagBits.TRANSFER_SRC_BIT,
            "staging usage remains transfer source only");
        assert(families[0] == topology[0], "graphics family remains first");
        if (expected_count > 1) assert(families[1] == topology[1], "compute family remains second");
        if (expected_count > 2) assert(families[2] == topology[2], "transfer family remains third");
        if (expected_count == 1) {
            assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "one family stays exclusive");
            assert(info.queue_family_index_count == 0 && info.p_queue_family_indices == null,
                "exclusive creation publishes no family list");
        } else {
            assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "distinct families use concurrent sharing");
            assert(info.queue_family_index_count == expected_count, "concurrent family count is exact");
            assert(info.p_queue_family_indices == &families[0], "create info points to caller storage");
        }
    }
}

fn void transfer_single_family_buffers_stay_exclusive() @test {
    gpu::BufferDesc[2] descs = {
        gpu::vk::readback_arena_desc(2048),
        gpu::vk::staging_dedicated_desc(2048),
    };
    uint[2] expected_usage = {
        (uint)vk::BufferUsageFlagBits.TRANSFER_DST_BIT,
        (uint)vk::BufferUsageFlagBits.TRANSFER_SRC_BIT,
    };
    gpu::MemoryKind[2] expected_memory = {
        gpu::MemoryKind.READBACK,
        gpu::MemoryKind.STAGING,
    };
    foreach (index, &desc : descs) {
        vk::BufferCreateInfo info = gpu::vk::make_buffer_create_info(desc, {})!!;
        assert(!desc.usage.shared_queues, "single-family transfer descriptor does not request sharing");
        assert((uint)info.usage == expected_usage[index], "single-family transfer usage remains exact");
        assert(desc.memory_kind == expected_memory[index], "single-family transfer memory kind remains exact");
        assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "single-family transfer buffer stays exclusive");
        assert(info.queue_family_index_count == 0 && info.p_queue_family_indices == null,
            "single-family transfer buffer publishes no family list");
    }
}

fn void transfer_range_places_aligned_extent() @test {
    gpu::vk::TransferArenaRange range = gpu::vk::transfer_arena_range(
        virtual_head: 0,
        virtual_tail: 12,
        retire_count: 1,
        capacity:     64,
        size:         16,
        align:        16,
    )!!;
    assert(range.virtual_start == 16, "range starts at the aligned tail");
    assert(range.virtual_end == 32, "range end includes the requested size");
    assert(range.physical_offset == 16, "physical offset follows the aligned virtual start");
}

fn void transfer_range_wraps_to_physical_zero() @test {
    gpu::vk::TransferArenaRange range = gpu::vk::transfer_arena_range(
        virtual_head: 48,
        virtual_tail: 60,
        retire_count: 1,
        capacity:     64,
        size:         8,
        align:        4,
    )!!;
    assert(range.virtual_start == 64, "physical wrap gap advances the virtual start");
    assert(range.virtual_end == 72, "wrapped range stays monotonic");
    assert(range.physical_offset == 0, "wrapped range begins at physical zero");
}

fn void transfer_range_rejects_one_byte_beyond_live_window() @test {
    fault full = @catch(gpu::vk::transfer_arena_range(
        virtual_head: ulong::max - 64,
        virtual_tail: ulong::max - 16,
        retire_count: 1,
        capacity:     64,
        size:         17,
        align:        1,
    ));
    assert(full == gpu::ARENA_FULL, "one byte beyond remaining live capacity faults ARENA_FULL");
}

fn void transfer_range_rejects_alignment_overflow() @test {
    fault full = @catch(gpu::vk::transfer_arena_range(
        virtual_head: ulong::max - 16,
        virtual_tail: ulong::max,
        retire_count: 1,
        capacity:     64,
        size:         1,
        align:        2,
    ));
    assert(full == gpu::ARENA_FULL, "alignment beyond ulong max faults ARENA_FULL");
}

fn void transfer_range_accepts_exact_virtual_end() @test {
    gpu::vk::TransferArenaRange range = gpu::vk::transfer_arena_range(
        virtual_head: ulong::max - 64,
        virtual_tail: ulong::max - 1,
        retire_count: 1,
        capacity:     64,
        size:         1,
        align:        1,
    )!!;
    assert(range.virtual_end == ulong::max, "exact representable virtual end succeeds");
}

fn void transfer_range_rejects_extent_overflow() @test {
    fault full = @catch(gpu::vk::transfer_arena_range(
        virtual_head: ulong::max - 1,
        virtual_tail: ulong::max - 1,
        retire_count: 1,
        capacity:     64,
        size:         2,
        align:        1,
    ));
    assert(full == gpu::ARENA_FULL, "extent beyond ulong max faults ARENA_FULL");
}

fn void transfer_range_rejects_head_ahead_of_tail() @test {
    fault full = @catch(gpu::vk::transfer_arena_range(
        virtual_head: 10,
        virtual_tail: 9,
        retire_count: 1,
        capacity:     64,
        size:         1,
        align:        1,
    ));
    assert(full == gpu::ARENA_FULL, "head ahead of tail cannot underflow live accounting");
}

fn void empty_transfer_range_normalizes_high_cursors() @test {
    gpu::vk::TransferArenaRange range = gpu::vk::transfer_arena_range(
        virtual_head: ulong::max,
        virtual_tail: ulong::max,
        retire_count: 0,
        capacity:     64,
        size:         8,
        align:        16,
    )!!;
    assert(range.virtual_start == 0 && range.virtual_end == 8,
        "an empty retire queue plans from normalized zero cursors");
}

fn void live_transfer_range_preserves_virtual_identity() @test {
    gpu::vk::TransferArenaRange range = gpu::vk::transfer_arena_range(
        virtual_head: ulong::max - 64,
        virtual_tail: ulong::max - 16,
        retire_count: 1,
        capacity:     64,
        size:         8,
        align:        1,
    )!!;
    assert(range.virtual_start == ulong::max - 16,
        "a live retire queue never rebases its virtual identity");
    assert(range.virtual_end == ulong::max - 8,
        "live virtual end remains monotonic and representable");
}

fn void transfer_range_rejects_invalid_alignment() @test {
    fault invalid = @catch(gpu::vk::transfer_arena_range(
        virtual_head: 0,
        virtual_tail: 0,
        retire_count: 0,
        capacity:     64,
        size:         1,
        align:        3,
    ));
    assert(invalid == gpu::INVALID_ARGUMENT, "non-power-of-two alignment faults INVALID_ARGUMENT");
}

fn void transfer_range_rejects_zero_size() @test {
    fault invalid = @catch(gpu::vk::transfer_arena_range(
        virtual_head: 0,
        virtual_tail: 0,
        retire_count: 0,
        capacity:     64,
        size:         0,
        align:        1,
    ));
    assert(invalid == gpu::INVALID_ARGUMENT, "zero-size transfer ranges fault INVALID_ARGUMENT");
}
