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

// Non-blocking readback: data at `span` becomes valid once the device
// timeline reaches `value`. `dedicated`/`range_end` are backend bookkeeping.
struct ReadbackTicket {
    GpuSpan        span;
    SemaphoreValue value;
    BufferHandle   dedicated;
    ulong          range_end;
}

const usz MAX_MEMORY_HEAPS = 16;
const usz ARENA_SIZE = 1 << 20;
const usz PERSISTENT_ARENA_SIZE = 64 << 20;
const usz DEFAULT_STAGING_ARENA_SIZE = 32 << 20;
const usz DEFAULT_READBACK_ARENA_SIZE = 8 << 20;

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

// Region for cmd_upload_texture; mirrors BufferTextureCopyDesc minus the
// source span (the staging arena provides it).
struct TextureUploadDesc {
    TextureHandle texture;
    uint          mip;
    uint          base_layer;
    uint          layer_count;
    uint          x;
    uint          y;
    uint          width;
    uint          height;
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

<* Record an upload into `dst` through the staging arena: staging span,
   memcpy, flush, copy. Records no barriers — destination preconditions and
   next-use transitions stay with the caller, exactly as with raw copies. *>
fn void? cmd_upload_buffer(CommandList* commands, BufferHandle dst, usz offset, char[] data) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_upload_buffer(commands, dst, offset, data);
}

<* Texture flavor of cmd_upload_buffer. The region follows the copy-desc rules
   (extent 0 = full mip); the tracked layout must be TRANSFER_DST. *>
fn void? cmd_upload_texture(CommandList* commands, TextureUploadDesc* desc, char[] data) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_upload_texture(commands, desc, data);
}

<* One-call blocking upload: own submission on the graphics queue, then the
   caller-named next-use barrier. Setup/tooling ergonomics — not a hot path. *>
fn void? upload_buffer_data(Device* device, BufferHandle dst, usz offset, char[] data, Stage next_stage, Hazard next_hazard) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.upload_buffer_data(device, dst, offset, data, next_stage, next_hazard);
}

<* Blocking full-mip texture upload. The texture must be tracked UNDEFINED
   (transitioned in) or already TRANSFER_DST; anything else faults
   INVALID_RESOURCE_STATE — re-uploading an in-use texture belongs in a
   command stream with explicit barriers. Ends at the named next-use. *>
fn void? upload_texture_data(Device* device, TextureHandle dst, uint mip, char[] data, Stage next_stage, Hazard next_hazard, TextureLayout next_layout) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.upload_texture_data(device, dst, mip, data, next_stage, next_hazard, next_layout);
}

<* One-call blocking readback. The from trio orders prior GPU writes before
   the copy; bytes land in `out` after an internal wait + invalidate. *>
fn void? readback_buffer_data(Device* device, BufferHandle src, usz offset, char[] out_data, Stage from_stage, Hazard from_hazard) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.readback_buffer_data(device, src, offset, out_data, from_stage, from_hazard);
}

<* Blocking full-mip texture readback; round-trips from_layout → TRANSFER_SRC
   → from_layout, leaving the texture as found. *>
fn void? readback_texture_data(Device* device, TextureHandle src, uint mip, char[] out_data, Stage from_stage, Hazard from_hazard, TextureLayout from_layout) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.readback_texture_data(device, src, mip, out_data, from_stage, from_hazard, from_layout);
}

<* Record a non-blocking buffer readback into the caller's command list.
   Source-side synchronization is the caller's barrier; data is resolvable
   after the recording frame ends (the device timeline advances at end_frame).
   Ownership: the range is held until resolve_readback releases it. *>
fn ReadbackTicket? cmd_readback_buffer(CommandList* commands, BufferHandle src, usz offset, usz size) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_readback_buffer(commands, src, offset, size);
}

<* Record a non-blocking full-mip texture readback. The texture must carry
   transfer_src usage and sit in the TRANSFER_SRC tracked layout; the caller
   owns the barriers on both sides. *>
fn ReadbackTicket? cmd_readback_texture(CommandList* commands, TextureHandle src, uint mip) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_readback_texture(commands, src, mip);
}

<* True once the ticket's data is resolvable; false for resolved tickets.
   Never faults. *>
fn bool poll_readback(Device* device, ReadbackTicket* ticket) {
    if (device.vtable == null) return false;
    return device.vtable.poll_readback(device, ticket);
}

<* Copy the ticket's bytes into `dest` and release the underlying range.
   Faults READBACK_NOT_READY before the ticket's timeline value signals and
   INVALID_ARGUMENT on an already-resolved ticket or short `dest`. *>
fn void? resolve_readback(Device* device, ReadbackTicket* ticket, char[] dest) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.resolve_readback(device, ticket, dest);
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
