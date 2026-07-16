module gpu;

// A shader-visible 64-bit GPU address. Distinct from a plain ulong so it can't
// be confused with sizes, offsets, or handles.
typedef GpuAddress = ulong;

// Where a resource's memory lives and how the host may access it.
enum MemoryKind : int {
    // Per-frame upload arena; contents live for one frame.
    FRAME_UPLOAD,
    // Host-visible with persistent lifetime; stable until freed.
    PERSISTENT_UPLOAD,
    // GPU-only; not host-mappable.
    DEVICE,
    // GPU-to-CPU readback; invalidate before reading.
    READBACK,
    // Transfer sources for staged uploads.
    STAGING,
}

<* Generation-checked identity for private readback state. *>
struct ReadbackTicket {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert ReadbackTicket::size <= ulong::size * 2;

<* True while the token names readback state; resolution consumes it. *>
fn bool ReadbackTicket.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool ReadbackTicket.equals(self, ReadbackTicket other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}

// Host-side view of a GPU-visible range. `cpu` may be null; `access` names
// the selected queue roles admitted to the range.
struct GpuSpan {
    GpuAddress       gpu;
    void*            cpu;
    usz              size;
    BufferHandle     buffer;
    AllocationHandle allocation;
    usz              offset;
    MemoryKind       kind;
    QueueRoles       access;
}

struct FrameToken {
    Device device;
    ulong  generation;
}
$assert FrameToken::size <= 16;

<* True while the token contains a frame generation. *>
fn bool FrameToken.is_valid(self) => (ulong)self.device != 0 && self.generation != 0;

const usz MAX_MEMORY_HEAPS = 16;
const usz MAX_PERSISTENT_ALLOCATIONS = 4096;
const usz DEFAULT_FRAME_ARENA_SIZE = 16 << 20;
const usz DEFAULT_PERSISTENT_ARENA_SIZE = 64 << 20;
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

<* Overflow-safe bounds check: offset + size must fit in limit. A zero size
   only checks the offset; each caller defines what zero means. *>
fn bool range_fits(usz limit, usz offset, usz size) @private {
    if (offset > limit) return false;
    return size <= limit - offset;
}

fn DeviceOperation? frame_operation(FrameToken* frame) @private {
    if (frame == null || frame.generation == 0) return INVALID_HANDLE~;
    return begin_device_operation(&frame.device);
}

<* Enter an idle frame. Double begin faults `INVALID_RESOURCE_STATE`; a retire
   wait may fault `WAIT_TIMEOUT`. Those rejections leave lifecycle and arena
   state unchanged. *>
fn FrameToken? begin_frame(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    FrameToken frame;
    operation.vtable.begin_frame(device, &frame)!;
    return frame;
}

<* `size` and power-of-two `align` must be nonzero (`INVALID_ARGUMENT`);
   capacity faults `ARENA_FULL`, invalid or stale tokens `INVALID_HANDLE`.
   Faults do not advance the arena; spans remain valid until its reset. *>
fn GpuSpan? alloc_frame_span(FrameToken* frame, usz size, usz align) {
    DeviceOperation operation = frame_operation(frame)!;
    defer end_device_operation(&operation);
    return operation.vtable.alloc_frame_span(frame, size, align);
}

<* End the identified frame. Invalid, stale, or consumed tokens fault
   `INVALID_HANDLE` without mutation; a failed end retains the token and
   lifecycle state for retry. *>
fn void? end_frame(FrameToken* frame) {
    DeviceOperation operation = frame_operation(frame)!;
    defer end_device_operation(&operation);
    operation.vtable.end_frame(frame)!;
    *frame = {};
}

<* Run `work` inside one frame. End faults take precedence and retain
   `out_frame`; successful end clears it. *>
macro void? @with_frame(FrameToken* out_frame, Device* device, #work, ...) {
    if (out_frame == null) return INVALID_HANDLE~;
    *out_frame = {};
    *out_frame = begin_frame(device)!;
    fault work_fault = @catch(#work(out_frame, ...$vaarg));
    fault end_fault = @catch(end_frame(out_frame));
    if (end_fault) return end_fault~;
    if (work_fault) return work_fault~;
}

// `memory_kind` must be PERSISTENT_UPLOAD; `align` 0 means no requirement;
// `usage` must fit the arena superset; `access` must name selected roles.
struct PersistentAllocDesc {
    usz         size;
    usz         align;
    BufferUsage usage;
    MemoryKind  memory_kind;
    QueueRoles  access;
    ZString     debug_name;
}

// Persistent-arena usage snapshot, returned by get_persistent_stats.
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

<* Allocate a long-lived span from the persistent arena. `size` must be
   nonzero; `align` 0 means no requirement, otherwise a power of two
   (INVALID_ARGUMENT). `access` must be a non-empty subset of selected roles
   and is preserved in the returned span. Its GPU address is stable until
   freed. Ownership: free with free_persistent_span. *>
fn GpuSpan? alloc_persistent_span(Device* device, PersistentAllocDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;
    return operation.vtable.alloc_persistent_span(device, desc);
}

<* Counterpart of alloc_persistent_span. *>
fn void? free_persistent_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.free_persistent_span(device, span);
}

<* Snapshot of persistent-arena usage and fragmentation counters. *>
fn PersistentArenaStats? get_persistent_stats(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_persistent_stats(device);
}

<* Record an upload into `dst` through the staging arena: staging span,
   memcpy, flush, copy. Records no barriers — destination preconditions and
   next-use transitions stay with the caller, exactly as with raw copies. *>
fn void? cmd_upload_buffer(CommandList* commands, BufferHandle dst, usz offset, char[] data) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    return operation.vtable.cmd_upload_buffer(commands, dst, offset, data);
}

<* Texture flavor of cmd_upload_buffer. The region follows the copy-desc rules
   (extent 0 = full mip); the tracked layout must be TRANSFER_DST. *>
fn void? cmd_upload_texture(CommandList* commands, TextureUploadDesc* desc, char[] data) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    return operation.vtable.cmd_upload_texture(commands, desc, data);
}

<* One-call blocking upload on the device's frame queue role, followed by the
   caller-named next-use barrier. dst must admit that role. Not a hot path. *>
fn void? upload_buffer_data(Device* device, BufferHandle dst, usz offset, char[] data, Stage next_stage, Hazard next_hazard) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.upload_buffer_data(device, dst, offset, data, next_stage, next_hazard);
}

<* Blocking full-mip texture upload. The texture must be tracked UNDEFINED
   (transitioned in) or already TRANSFER_DST; anything else faults
   INVALID_RESOURCE_STATE — re-uploading an in-use texture belongs in a
   command stream with explicit barriers. Ends at the named next-use. *>
fn void? upload_texture_data(Device* device, TextureHandle dst, uint mip, char[] data, Stage next_stage, Hazard next_hazard, TextureLayout next_layout) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.upload_texture_data(device, dst, mip, data, next_stage, next_hazard, next_layout);
}

<* One-call blocking readback on the device's frame queue role. src must admit
   that role. The from trio orders prior writes before copy and host read. *>
fn void? readback_buffer_data(Device* device, BufferHandle src, usz offset, char[] out_data, Stage from_stage, Hazard from_hazard) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.readback_buffer_data(device, src, offset, out_data, from_stage, from_hazard);
}

<* Blocking full-mip texture readback; round-trips from_layout → TRANSFER_SRC
   → from_layout, leaving the texture as found. *>
fn void? readback_texture_data(Device* device, TextureHandle src, uint mip, char[] out_data, Stage from_stage, Hazard from_hazard, TextureLayout from_layout) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.readback_texture_data(device, src, mip, out_data, from_stage, from_hazard, from_layout);
}

<* Record a non-blocking buffer readback into the caller's command list.
   Source-side synchronization is the caller's barrier; data is resolvable
   after the recording frame ends (the device timeline advances at end_frame).
   Ownership: the range is held until resolve_readback releases it. *>
fn ReadbackTicket? cmd_readback_buffer(CommandList* commands, BufferHandle src, usz offset, usz size) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    return operation.vtable.cmd_readback_buffer(commands, src, offset, size);
}

<* Record a non-blocking full-mip texture readback. The texture must carry
   transfer_src usage and sit in the TRANSFER_SRC tracked layout; the caller
   owns the barriers on both sides. *>
fn ReadbackTicket? cmd_readback_texture(CommandList* commands, TextureHandle src, uint mip) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    return operation.vtable.cmd_readback_texture(commands, src, mip);
}

<* True once the ticket's data is resolvable; false for the consumed
   original token. Stale aliases and invalid devices fault INVALID_HANDLE. *>
fn bool? poll_readback(Device* device, ReadbackTicket* ticket) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (ticket == null) return INVALID_HANDLE~;
    if (!ticket.is_valid()) return false;
    validate_resource_owner(&operation, ticket.owner, ResourceOwnerKind.READBACK_TICKET)!;
    return operation.vtable.poll_readback(device, ticket);
}

<* Copy the ticket's bytes into `dest` and release the underlying range.
   Faults READBACK_NOT_READY before the ticket's timeline value signals and
   INVALID_ARGUMENT on a consumed ticket or short `dest`; stale aliases fault
   INVALID_HANDLE. *>
fn void? resolve_readback(Device* device, ReadbackTicket* ticket, char[] dest) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (ticket == null) return INVALID_HANDLE~;
    if (!ticket.is_valid()) return INVALID_ARGUMENT~;
    validate_resource_owner(&operation, ticket.owner, ResourceOwnerKind.READBACK_TICKET)!;
    return operation.vtable.resolve_readback(device, ticket, dest);
}

<* Derive a sub-range without bounds or overflow checks. Use only when the
   caller has already proved the range, or deliberately needs an invalid span
   for validation testing. *>
fn GpuSpan GpuSpan.unchecked_subspan(self, usz offset, usz size) {
    return {
        .gpu    = (GpuAddress)((ulong)self.gpu + offset),
        .cpu    = self.cpu == null ? null : (char*)self.cpu + offset,
        .size       = size,
        .buffer     = self.buffer,
        .allocation = self.allocation,
        .offset     = self.offset + offset,
        .kind       = self.kind,
        .access     = self.access,
    };
}

<* Derive a non-empty exact-sized sub-range within this immediate parent.
   Faults INVALID_ARGUMENT for zero size, a range that escapes the parent, or
   any derived address/offset addition that would overflow. *>
fn GpuSpan? GpuSpan.checked_subspan(self, usz offset, usz size) {
    if (size == 0) return INVALID_ARGUMENT~;
    if (!range_fits(self.size, offset, size)) return INVALID_ARGUMENT~;

    ulong gpu_address = (ulong)self.gpu;
    if ((ulong)offset > ulong::max - gpu_address) return INVALID_ARGUMENT~;
    if (offset > usz::max - self.offset) return INVALID_ARGUMENT~;
    if (self.cpu != null && (uptr)offset > uptr::max - (uptr)self.cpu) {
        return INVALID_ARGUMENT~;
    }
    return self.unchecked_subspan(offset, size);
}
