module gpu_test;

import gpu;
import gpu::vk @public;

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
