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
enum MemoryClass : int {
    CPU_WRITE,
    GPU_PRIVATE,
    CPU_READ,
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

<* Non-owning identity and range into generic GPU data. *>
struct GpuSpan {
    ulong owner;
    uint  index;
    uint  generation;
    usz   offset;
    usz   size;
}

<* `size` must be nonzero; alignment zero selects 16 bytes. *>
struct AllocationDesc {
    usz         size;
    usz         alignment;
    MemoryClass memory_class;
    QueueRoles  access;
    ZString     debug_name;
}

<* Immutable allocation properties and actual mapping behavior. *>
struct AllocationInfo {
    usz         size;
    usz         alignment;
    MemoryClass memory_class;
    QueueRoles  access;
    bool        mapped;
    bool        coherent;
    bool        addressable;
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

fn bool memory_class_is_known(MemoryClass memory_class) @private {
    switch (memory_class) {
        case MemoryClass.CPU_WRITE:
        case MemoryClass.GPU_PRIVATE:
        case MemoryClass.CPU_READ:
            return true;
        default:
            return false;
    }
}

fn usz? normalized_allocation_alignment(usz alignment) @private {
    const usz MIN_ALLOCATION_ALIGNMENT = 16;
    if (alignment == 0) return MIN_ALLOCATION_ALIGNMENT;
    if ((alignment & (alignment - 1)) != 0) return INVALID_ARGUMENT~;
    return max(alignment, MIN_ALLOCATION_ALIGNMENT);
}

<* Allocate addressable generic GPU data. Free with `free_allocation`. *>
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null || desc.size == 0) return INVALID_ARGUMENT~;
    if (!memory_class_is_known(desc.memory_class)) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;

    AllocationDesc normalized = *desc;
    normalized.alignment = normalized_allocation_alignment(desc.alignment)!;
    GpuAllocation allocation = operation.vtable.allocate_memory(
        device,
        &normalized,
    )!;
    retain_device_children(&operation);
    return allocation;
}

<* Free quiescent storage; faults preserve the owning token. *>
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (allocation == null || !allocation.is_valid()) return INVALID_HANDLE~;
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    operation.vtable.free_allocation(device, *allocation)!;
    release_device_children(&operation);
    *allocation = GPU_ALLOCATION_INVALID;
}

<* Return immutable properties for a live allocation. *>
fn AllocationInfo? get_allocation_info(
    Device* device,
    GpuAllocation allocation,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_info(device, allocation);
}

<* Borrow the allocation's complete range. *>
fn GpuSpan? get_allocation_span(Device* device, GpuAllocation allocation) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_span(device, allocation);
}

<* Borrow the CPU mapping for a live host-visible span. *>
fn char[]? get_span_mapping(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_span_mapping(device, span);
}

<* Return the stable nonzero address of a live span. *>
fn GpuAddress? get_span_address(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_span_address(device, span);
}

<* Make CPU writes to a live mapped allocation range visible to the GPU.
   Coherent ranges are no-ops. *>
fn void? flush_mapped_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        span.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.flush_mapped_span(device, span);
}

<* After the relevant GPU completion point, make its writes visible to CPU
   reads. This call does not wait; coherent ranges are no-ops. *>
fn void? invalidate_mapped_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        span.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.invalidate_mapped_span(device, span);
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
    retain_device_children(&operation);
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
    release_device_children(&operation);
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
   and is recorded in the arena's immutable allocation identity. The span's
   GPU address is stable until freed. Ownership: free with free_persistent_span. *>
fn GpuSpan? alloc_persistent_span(Device* device, PersistentAllocDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;
    GpuSpan span = operation.vtable.alloc_persistent_span(device, desc)!;
    retain_device_children(&operation);
    return span;
}

<* Counterpart of alloc_persistent_span. *>
fn void? free_persistent_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    operation.vtable.free_persistent_span(device, span)!;
    release_device_children(&operation);
}

<* Snapshot of persistent-arena usage and fragmentation counters. *>
fn PersistentArenaStats? get_persistent_stats(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_persistent_stats(device);
}

<* Record an exact-span upload through the staging arena. `data.len` must equal
   `dst.size`; the caller records destination barriers. *>
fn void? cmd_upload_buffer(CommandList* commands, GpuSpan dst, char[] data) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    return operation.vtable.cmd_upload_buffer(commands, dst, data);
}

<* Texture flavor of cmd_upload_buffer. The region follows the copy-desc rules
   (extent 0 = full mip); the tracked layout must be TRANSFER_DST. *>
fn void? cmd_upload_texture(CommandList* commands, TextureUploadDesc* desc, char[] data) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    return operation.vtable.cmd_upload_texture(commands, desc, data);
}

<* Blocking exact-span upload followed by the caller-named next-use barrier.
   `data.len` must equal `dst.size`; `dst` must admit the frame queue role. *>
fn void? upload_buffer_data(
    Device* device,
    GpuSpan dst,
    char[] data,
    Stage next_stage,
    Hazard next_hazard,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.upload_buffer_data(
        device:      device,
        dst:         dst,
        data:        data,
        next_stage:  next_stage,
        next_hazard: next_hazard,
    );
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

<* Blocking exact-span readback. `out_data.len` must equal `src.size`; `src`
   must admit the frame queue role. The hazards order prior writes. *>
fn void? readback_buffer_data(
    Device* device,
    GpuSpan src,
    char[] out_data,
    Stage from_stage,
    Hazard from_hazard,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.readback_buffer_data(
        device:      device,
        src:         src,
        out_data:    out_data,
        from_stage:  from_stage,
        from_hazard: from_hazard,
    );
}

<* Blocking full-mip texture readback; round-trips from_layout → TRANSFER_SRC
   → from_layout, leaving the texture as found. *>
fn void? readback_texture_data(Device* device, TextureHandle src, uint mip, char[] out_data, Stage from_stage, Hazard from_hazard, TextureLayout from_layout) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.readback_texture_data(device, src, mip, out_data, from_stage, from_hazard, from_layout);
}

<* Record a non-blocking buffer readback into the caller's command list.
   Source-side synchronization is the caller's barrier. The ticket holds its
   range until resolve_readback, or until discard cancels the command. *>
fn ReadbackTicket? cmd_readback_buffer(CommandList* commands, GpuSpan src) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    ReadbackTicket ticket = operation.vtable.cmd_readback_buffer(commands, src)!;
    retain_device_children(&operation);
    return ticket;
}

<* Record a non-blocking full-mip texture readback. The texture must carry
   transfer_src usage and sit in the TRANSFER_SRC tracked layout; the caller
   owns the barriers on both sides. *>
fn ReadbackTicket? cmd_readback_texture(CommandList* commands, TextureHandle src, uint mip) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    ReadbackTicket ticket = operation.vtable.cmd_readback_texture(
        commands,
        src,
        mip,
    )!;
    retain_device_children(&operation);
    return ticket;
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
    operation.vtable.resolve_readback(device, ticket, dest)!;
    release_device_children(&operation);
}

<* Derive a sub-range without bounds or overflow checks. Use only when the
   caller has already proved the range, or deliberately needs an invalid span
   for validation testing. *>
fn GpuSpan GpuSpan.unchecked_subspan(self, usz offset, usz size) {
    return {
        .owner      = self.owner,
        .index      = self.index,
        .generation = self.generation,
        .offset     = self.offset + offset,
        .size       = size,
    };
}

<* Derive a non-empty exact-sized range within this immediate parent. *>
fn GpuSpan? GpuSpan.checked_subspan(self, usz offset, usz size) {
    if (size == 0 || !range_fits(self.size, offset, size)) {
        return INVALID_ARGUMENT~;
    }
    if (offset > usz::max - self.offset) return INVALID_ARGUMENT~;
    return self.unchecked_subspan(offset, size);
}
