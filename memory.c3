module gpu;

// A shader-visible 64-bit GPU address. Distinct from a plain ulong so it can't
// be confused with sizes, offsets, or handles.
typedef GpuAddress = ulong;

enum MemoryKind : int {
    FRAME_UPLOAD,
    PERSISTENT_UPLOAD,
    DEVICE,
    READBACK,
    STAGING,
}

// Host-side view of a GPU-visible range. `cpu` may be null for non-mappable
// memory; `gpu` is non-zero for shader-visible memory.
struct GpuSpan {
    GpuAddress   gpu;
    void*        cpu;
    usz          size;
    BufferHandle buffer;
    usz          offset;
    MemoryKind   kind;
}

const usz MAX_MEMORY_HEAPS = 16;
const usz ARENA_SIZE = 1 << 20;
const usz PERSISTENT_ARENA_SIZE = 64 << 20;

// Per-heap memory usage snapshot.
struct MemoryHeapBudget {
    ulong usage;
    ulong budget;
    ulong allocation_bytes;
    ulong block_bytes;
}

// Device memory snapshot. `heaps[0..heap_count]` are valid. Resource counts are
// 0 until the resource tables exist.
struct MemoryStats {
    MemoryHeapBudget[MAX_MEMORY_HEAPS] heaps;
    uint  heap_count;
    ulong buffer_count;
    ulong texture_count;
    ulong live_allocation_count;
}

// Round `value` up to the next multiple of `align` (a power of two).
fn usz align_up(usz value, usz align) => (value + align - 1) & ~(align - 1);

<* Begin the next frame: advance to the next frame-in-flight slot, wait for its
   arena to retire, reset it, and update the allocator's frame index. *>
fn void? begin_frame(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.begin_frame(device);
}

<* Bump-allocate a span from the current frame's upload arena. Returns ARENA_FULL
   when the arena is exhausted. The span is valid until the arena is reset. *>
fn GpuSpan? alloc_frame_span(Device* device, usz size, usz align) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.alloc_frame_span(device, size, align);
}

<* End the current frame, recording its timeline value. *>
fn void? end_frame(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.end_frame(device);
}

struct PersistentAllocDesc {
    usz         size;
    usz         align;
    BufferUsage usage;
    MemoryKind  memory_kind;
    ZString     debug_name;
}

struct PersistentArenaStats {
    ulong capacity_bytes;
    ulong used_bytes;
    ulong allocation_count;
}

<* Allocate a long-lived span from the persistent arena. Its GPU address is
   stable until freed. Ownership: free with free_persistent_span. *>
fn GpuSpan? alloc_persistent_span(Device* device, PersistentAllocDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.alloc_persistent_span(device, desc);
}

fn void? free_persistent_span(Device* device, GpuSpan span) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.free_persistent_span(device, span);
}

fn PersistentArenaStats? get_persistent_stats(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_persistent_stats(device);
}

<* Derive a sub-range starting `offset` bytes into this span. The GPU address
   and (when present) CPU pointer advance by `offset`; the backing buffer and
   memory kind are inherited. *>
fn GpuSpan GpuSpan.subspan(self, usz offset, usz size) {
    return {
        .gpu    = (GpuAddress)((ulong)self.gpu + offset),
        .cpu    = self.cpu == null ? null : (char*)self.cpu + offset,
        .size   = size,
        .buffer = self.buffer,
        .offset = self.offset + offset,
        .kind   = self.kind,
    };
}
