module gpu_test;

import gpu @public;

struct SlotHandle {
    ulong owner;
    uint  index;
    uint  generation;
}

fn bool SlotHandle.is_valid(self) => self.owner != 0 && self.generation != 0;

fn void public_state_tokens_are_compact() @test {
    assert(Device::size == ulong::size, "Device is one opaque 64-bit token");
    assert(GpuAllocation::size <= ulong::size * 2,
        "GpuAllocation fits in two machine words");

}

fn void gpu_allocation_identity_is_generation_checked() @test {
    GpuAllocation first = { .owner = 7, .index = 3, .generation = 11 };
    GpuAllocation same = first;
    GpuAllocation stale = first;
    stale.generation++;

    assert(first.is_valid(), "owner and generation make an allocation valid");
    assert(first == same, "copied allocation identity compares equal");
    assert(first != stale, "generation participates in allocation identity");
    assert(!gpu::GPU_ALLOCATION_INVALID.is_valid(), "the zero allocation is invalid");
}

fn void index_pack_unpack() @test {
    TextureIndex t = { .slot = 42, .generation = 7 };
    assert(t.slot == 42, "slot round-trips");
    assert(t.generation == 7, "generation round-trips");
    assert((uint)t == (7u << 16) | 42u, "raw uint matches slot(0..15)|generation(16..31)");
    assert(((uint)t & 0xFFFFu) == 42, "shader-side slot mask recovers the slot");
    assert(t.is_valid(), "non-zero index is valid");
    assert(!gpu::TEXTURE_INDEX_INVALID.is_valid(), "INVALID index is not valid");
    assert(!gpu::SAMPLER_INDEX_INVALID.is_valid(), "INVALID sampler index is not valid");
}

fn void index_smallest_valid_never_zero() @test {
    SamplerIndex s = { .slot = 0, .generation = 1 };
    assert((uint)s != 0, "live index is never all-zero");
    assert(s.is_valid(), "slot 0 with generation 1 is valid");
}

// Concrete generation-checked slot table used to exercise handle invariants.
const uint FREE_NONE = 0;

struct IntCell {
    int  value;
    uint generation;
    uint next_free;
    bool used;
}

struct IntTable {
    IntCell[] slots;
    ulong     owner;
    uint      free_head;
    uint      next_new;
}

fn SlotHandle? IntTable.alloc(&self, int value) {
    uint idx;
    if (self.free_head != FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation++;
    self.slots[idx].used = true;
    return (SlotHandle){ .owner = self.owner, .index = idx, .generation = self.slots[idx].generation };
}

fn int*? IntTable.get(&self, SlotHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? IntTable.free(&self, SlotHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn void slot_alloc_get_free() @test {
    IntCell[8] storage;
    IntTable t = { .slots = storage[..], .owner = 1 };

    SlotHandle a = t.alloc(100)!!;
    assert(a.is_valid(), "alloc returns a valid handle");
    assert(*t.get(a)!! == 100, "get returns the stored value");

    t.free(a)!!;
    fault freed_fault = @catch(t.get(a));
    assert(freed_fault == gpu::INVALID_HANDLE, "freed handle returns INVALID_HANDLE");
}

fn void generation_mismatch_rejected() @test {
    IntCell[8] storage;
    IntTable t = { .slots = storage[..], .owner = 1 };

    SlotHandle stale = t.alloc(1)!!;
    t.free(stale)!!;
    SlotHandle fresh = t.alloc(2)!!;

    assert(fresh.index == stale.index, "freed slot index is reused");
    assert(fresh.generation > stale.generation, "reused slot bumps generation");

    fault stale_fault = @catch(t.get(stale));
    assert(stale_fault == gpu::INVALID_HANDLE, "stale handle to reused slot returns INVALID_HANDLE");
    assert(*t.get(fresh)!! == 2, "fresh handle resolves");
}

fn void slot_table_full() @test {
    IntCell[2] storage;
    IntTable t = { .slots = storage[..], .owner = 1 };

    t.alloc(1)!!;
    t.alloc(2)!!;
    fault full_fault = @catch(t.alloc(3));
    assert(full_fault == gpu::SLOT_TABLE_FULL, "allocating past capacity raises SLOT_TABLE_FULL");
}
