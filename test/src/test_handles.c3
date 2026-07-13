module gpu_test;

import gpu @public;

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


fn void public_state_tokens_are_compact() @test {
    assert(Device::size == ulong::size, "Device is one opaque 64-bit token");
    assert(ReadbackTicket::size == ulong::size, "ReadbackTicket is one generation-checked token");
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

// Concrete generation-checked slot table mirroring the backend's per-type
// tables (the generic gpu::slots module was removed to avoid a c3c 0.8.0
// codegen crash; see slot table comments in gpu/vk/*.c3).
const uint FREE_NONE = 0;

struct IntCell {
    int  value;
    uint generation;
    uint next_free;
    bool used;
}

struct IntTable {
    IntCell[] slots;
    uint      free_head;
    uint      next_new;
}

fn BufferHandle? IntTable.alloc(&self, int value) {
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
    return (BufferHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn int*? IntTable.get(&self, BufferHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? IntTable.free(&self, BufferHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn void slot_alloc_get_free() @test {
    IntCell[8] storage;
    IntTable t = { .slots = storage[..] };

    BufferHandle a = t.alloc(100)!!;
    assert(a.is_valid(), "alloc returns a valid handle");
    assert(*t.get(a)!! == 100, "get returns the stored value");

    t.free(a)!!;
    fault freed_fault = @catch(t.get(a));
    assert(freed_fault == gpu::INVALID_HANDLE, "freed handle returns INVALID_HANDLE");
}

fn void generation_mismatch_rejected() @test {
    IntCell[8] storage;
    IntTable t = { .slots = storage[..] };

    BufferHandle stale = t.alloc(1)!!;
    t.free(stale)!!;
    BufferHandle fresh = t.alloc(2)!!;

    assert(fresh.index == stale.index, "freed slot index is reused");
    assert(fresh.generation > stale.generation, "reused slot bumps generation");

    fault stale_fault = @catch(t.get(stale));
    assert(stale_fault == gpu::INVALID_HANDLE, "stale handle to reused slot returns INVALID_HANDLE");
    assert(*t.get(fresh)!! == 2, "fresh handle resolves");
}

fn void slot_table_full() @test {
    IntCell[2] storage;
    IntTable t = { .slots = storage[..] };

    t.alloc(1)!!;
    t.alloc(2)!!;
    fault full_fault = @catch(t.alloc(3));
    assert(full_fault == gpu::SLOT_TABLE_FULL, "allocating past capacity raises SLOT_TABLE_FULL");
}
