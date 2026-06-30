// Generic, generation-checked slot table — the backend-agnostic backbone the
// resource tables (buffers, textures, …) instantiate. Pure CPU, no allocation:
// the caller provides the backing `Slot[]` storage.
//
// `Handle` is one of the handle bitstructs (it must expose `index` and
// `generation`); `Value` is the stored payload. Instantiate per resource, e.g.
//   import gpu::slots;
//   SlotTable{BufferHandle, BufferRecord} table = { .slots = storage[..] };
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

<* Allocate a slot for `value` and return a fresh handle.
   Reuses a freed slot if one is available, otherwise grows into unused
   storage. Faults with SLOT_TABLE_FULL when capacity is exhausted. *>
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

<* Resolve a handle to a pointer to its stored value.
   Faults with INVALID_HANDLE if the slot is out of range, unused, or the
   handle's generation no longer matches (a stale handle to a reused slot). *>
fn Value*? SlotTable.get(&self, Handle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

<* Retire the slot a handle refers to and bump its generation so any
   outstanding handle to it becomes stale. Faults with INVALID_HANDLE if the
   handle does not currently resolve. *>
fn void? SlotTable.free(&self, Handle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}
