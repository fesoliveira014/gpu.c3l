// Generic generation-checked slot table over caller-provided storage, the
// backbone the backend resource tables instantiate. `Handle` is a handle
// bitstruct (exposing index/generation); `Value` is the stored payload.
module gpu::slots <Handle, Value>;

import gpu;

const uint FREE_NONE = 0;

struct Slot {
    Value value;
    uint  generation;
    uint  next_free;   // (index + 1) of next free slot, FREE_NONE if none
    bool  used;
}

struct SlotTable {
    Slot[] slots;
    uint   free_head;  // (index + 1) of first free slot, FREE_NONE if none
    uint   next_new;   // next never-yet-used slot index
}

<* Allocate a slot (reusing a freed one if available) and return a fresh handle.
   Faults SLOT_TABLE_FULL when capacity is exhausted. *>
fn Handle? SlotTable.alloc(&self, Value value) {
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
    self.slots[idx].generation++;   // 0 -> 1 on first use; bumped on reuse
    self.slots[idx].used = true;
    Handle h = { .index = idx, .generation = self.slots[idx].generation };
    return h;
}

<* Resolve a handle to its stored value, or INVALID_HANDLE if the slot is out of
   range, unused, or stale (generation mismatch). *>
fn Value*? SlotTable.get(&self, Handle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

<* Retire a slot and bump its generation, invalidating outstanding handles.
   Faults INVALID_HANDLE if the handle does not resolve. *>
fn void? SlotTable.free(&self, Handle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}
