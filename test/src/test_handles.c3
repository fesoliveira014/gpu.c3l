module gpu_test;

import gpu;
import gpu::slots;

fn void handle_pack_unpack() @test {
    BufferHandle h = { .index = 42, .generation = 7 };
    assert(h.index == 42, "index round-trips");
    assert(h.generation == 7, "generation round-trips");
    assert(h.is_valid(), "non-zero handle is valid");
    assert(!gpu::BUFFER_HANDLE_INVALID.is_valid(), "INVALID handle is not valid");
}

fn void handle_kinds_distinct_raw() @test {
    // index 0 + generation 1 is the smallest valid handle: never raw 0.
    BufferHandle h = { .index = 0, .generation = 1 };
    assert((ulong)h != 0, "live handle is never all-zero");
}

fn void slot_alloc_get_free() @test {
    Slot{BufferHandle, int}[8] storage;
    SlotTable{BufferHandle, int} t = { .slots = storage[..] };

    BufferHandle a = t.alloc(100)!!;
    assert(a.is_valid(), "alloc returns a valid handle");
    assert(*t.get(a)!! == 100, "get returns the stored value");

    t.free(a)!!;
    bool faulted = false;
    if (catch t.get(a)) faulted = true;
    assert(faulted, "freed handle no longer resolves");
}

fn void generation_mismatch_rejected() @test {
    Slot{BufferHandle, int}[8] storage;
    SlotTable{BufferHandle, int} t = { .slots = storage[..] };

    BufferHandle stale = t.alloc(1)!!;
    t.free(stale)!!;
    BufferHandle fresh = t.alloc(2)!!;

    assert(fresh.index == stale.index, "freed slot index is reused");
    assert(fresh.generation > stale.generation, "reused slot bumps generation");

    bool faulted = false;
    if (catch t.get(stale)) faulted = true;
    assert(faulted, "stale handle to reused slot is rejected");
    assert(*t.get(fresh)!! == 2, "fresh handle resolves");
}

fn void slot_table_full() @test {
    Slot{BufferHandle, int}[2] storage;
    SlotTable{BufferHandle, int} t = { .slots = storage[..] };

    t.alloc(1)!!;
    t.alloc(2)!!;
    bool faulted = false;
    if (catch t.alloc(3)) faulted = true;
    assert(faulted, "allocating past capacity raises SLOT_TABLE_FULL");
}
