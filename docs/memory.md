# gpu.c3l Memory Architecture

## 1. Purpose

Memory is the foundation of `gpu.c3l`. The library's shader ABI depends on user code being able to write root data and GPU data structures into memory that shaders can address. The Vulkan backend uses Vulkan Memory Allocator through `vma.c3l` for Vulkan memory allocation, while exposing a smaller public memory model.

The public API uses:

```text
GpuAllocation
GpuSpan
GpuAddress
MemoryClass
AllocationDesc
AllocationInfo
TextureHandle
MemoryKind
```

The backend uses:

```text
vma::Allocator
vma::Allocation
vma::AllocationInfo
vma::VirtualBlock
vk::Buffer
vk::Image
vk::DeviceAddress
```

`vma::` and `vk::` types are never public API types.

## 2. VMA integration boundary

Only the Vulkan backend imports `vma`.

Backend files that may import `vma`:

```text
gpu/vk/allocator.c3
gpu/vk/allocation.c3
gpu/vk/memory.c3
gpu/vk/buffer.c3
gpu/vk/texture.c3
gpu/vk/debug.c3
```

Public files must not import `vma`.

VMA responsibilities:

```text
create/destroy allocator
create/destroy buffer allocations
create/destroy image allocations
map/unmap host memory
flush/invalidate non-coherent memory
query allocation info
query heap budgets and stats
name allocations
provide virtual allocator for CPU-side suballocation
```

`gpu.c3l` responsibilities:

```text
define behavioral memory classes
track owning allocations and non-owning span identities
keep mappings, addresses, access, and native backing private
translate resource policy to Vulkan/VMA
own frame/persistent/staging/readback arenas
validate bounds, capabilities, and lifetimes
record explicit barriers
```

## 3. Backend allocator state

The Vulkan device state owns one VMA allocator.

```text
VkDeviceState
    vk::Instance instance
    vk::PhysicalDevice physical_device
    vk::Device device
    vma::Allocator allocator
    AllocationTable allocations
    FrameArenaState[] frame_arenas
    PersistentArenaState persistent_arena
    TransferArenaState staging_arena
    TransferArenaState readback_arena
    DedicatedRetireState dedicated_staging
```

The persistent arena is reserved for `PERSISTENT_UPLOAD`; device-local
persistent data uses `MemoryClass.GPU_PRIVATE` allocations.

Allocator creation happens after Vulkan device creation and before resource
creation. The allocation table is destroyed before the allocator.

## 4. VMA allocator creation policy

The backend must create `vma::Allocator` with:

```text
instance
physical_device
device
vulkan_api_version
allocator flags for buffer device address
allocator flags for memory budget when supported
```

If buffer device address is unavailable, device creation fails because every
generic allocation must provide a stable GPU address.

## 5. Independent allocations

`GpuAllocation` owns device-scoped data or texture storage. `GpuSpan` borrows
generic data from an allocation or arena and stores only
`{ owner, index, generation, offset, size }`. Mapping, GPU address, access,
memory class, and native backing remain in device state.

```text
MemoryClass.CPU_WRITE    mapped for host writes
MemoryClass.GPU_PRIVATE  unmapped, addressable GPU data
MemoryClass.CPU_READ     mapped for host reads
MemoryClass.TEXTURE      unmapped, non-addressable texture storage

AllocationDesc
    size
    alignment
    memory_class
    access
    texture_requirements
    debug_name

allocate_memory
free_allocation
get_allocation_info
get_allocation_span
get_span_mapping
get_span_address
flush_mapped_span
invalidate_mapped_span
```

Size must be nonzero. Alignment zero selects 16 bytes; explicit alignment must
be a power of two and is normalized to at least 16. `AllocationInfo` reports
the immutable size, actual alignment, class, access, mapping, coherence, and
address capabilities.

Generic classes create a private addressable buffer and VMA allocation.
`TEXTURE` creates raw device-local memory from queried compatibility masks.
Both publish an `AllocationTable` slot only after native work succeeds.
The table owns generation and liveness. Mapping, address, and visibility calls
validate owner, generation, bounds, and required mapping before native use.
`flush_mapped_span` and `invalidate_mapped_span` accept only independent
allocation spans. No Vulkan or VMA object crosses the public boundary.

`free_allocation` destroys the backing immediately and invalidates the token
only after success. The caller must ensure GPU use is quiescent. Faults preserve
the token. Allocation debug names are copied into backend state and included in
leak diagnostics.

Use `checked_subspan` to partition a span:

```c3
GpuSpan vertices = packed.checked_subspan(0, vertex_bytes)!;
GpuSpan indices = packed.checked_subspan(vertex_bytes, index_bytes)!;
```

Checked slicing preserves identity, changes only offset and size, and rejects
zero size, parent escape, and offset overflow. `unchecked_subspan` performs no
bounds or overflow checks.

## 6. Public memory policy

| Need | Public API | Contract |
|---|---|---|
| CPU-written generic data | `MemoryClass.CPU_WRITE` | map, write, flush, then submit |
| GPU-private generic data | `MemoryClass.GPU_PRIVATE` | upload or write from GPU commands |
| CPU-read generic data | `MemoryClass.CPU_READ` | wait, invalidate, then read |
| Placed textures | `MemoryClass.TEXTURE` | query requirements, allocate, create placed textures |
| Per-frame roots and tables | `alloc_frame_span` | mapped, host-coherent, valid for the frame generation |
| Long-lived CPU-written tables | `alloc_persistent_span` with `PERSISTENT_UPLOAD` | mapped and host-coherent until freed |
| Upload and readback scratch | transfer helpers | staging and readback storage stay private |

`MemoryKind` is not an independent-allocation selector. Public code supplies it
only through `PersistentAllocDesc`, where `PERSISTENT_UPLOAD` is mandatory.
`FRAME_UPLOAD`, `DEVICE`, `READBACK`, and `STAGING` describe private backing
policy selected by the corresponding API path.

## 7. Private buffer backing

Generic allocations and arena ranges use private addressable Vulkan buffers;
texture allocations contain only compatible image memory. Generic allocations
use a fixed native-usage superset. Queue-family sharing derives from the
immutable `QueueRoles` access set. Creation publishes an allocation or span
identity only after its native buffer, VMA allocation, mapping state, and
nonzero device address are complete.

Private `gpu::vk::BufferHandle`, `BufferDesc`, and `BufferUsage` declarations
support this implementation. They are not public resource types.

## 8. Texture allocation

Public descriptor:

```text
TextureDesc
    TextureDimension dimension
    uint width
    uint height
    uint depth
    uint mip_levels
    uint array_layers
    Format format
    TextureUsage usage
    QueueRoles access
    ZString debug_name
```

Backend translation:

```text
TextureUsage.SAMPLED      -> VK_IMAGE_USAGE_SAMPLED_BIT
TextureUsage.STORAGE      -> VK_IMAGE_USAGE_STORAGE_BIT
TextureUsage.COLOR_ATTACH -> VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
TextureUsage.DEPTH_ATTACH -> VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
TextureUsage.TRANSFER_SRC -> VK_IMAGE_USAGE_TRANSFER_SRC_BIT
TextureUsage.TRANSFER_DST -> VK_IMAGE_USAGE_TRANSFER_DST_BIT
```

Owned creation:

```text
validate descriptor and adapter support
create image and allocation transactionally
create the default view
publish TextureHandle
```

Placed creation:

```text
get_texture_requirements
allocate_memory with MemoryClass.TEXTURE and all required compatibility values
create_placed_texture at an aligned, non-overlapping offset
```

Dedicated creation:

```text
get_texture_requirements
create_dedicated_texture with an exact-size compatible AllocationDesc
```

Requirements are device-owned and opaque. Incompatible groups,
dedicated-only requirements, insufficient size or access, stale allocations,
and overlapping live placements fail before image creation.

## 9. Texture lifetime

Destroying an owned texture releases its image allocation immediately.
Destroying a placed texture releases the image but not its `GpuAllocation`.
`free_allocation` returns `RESOURCE_IN_USE` while a placed image is live.
Dedicated creation returns separate texture and allocation tokens; destroy the
texture before releasing its allocation.

## 10. Frame upload arena

Each frame-in-flight owns one or more VMA-backed buffers.

Frame upload allocation is governed by a strict lifecycle:

```text
IDLE --begin_frame(device)--> ACTIVE(token generation) --end_frame(token)--> IDLE
```

`begin_frame` returns a `FrameToken`; `alloc_frame_span` and `end_frame` require
that token rather than a bare device pointer. A copied token may allocate while
its generation remains active. The token embeds the owning `Device` value and
does not borrow the caller's device variable. Successful end clears the passed
copy and makes every alias stale. Failed end leaves the token, generation, frame
slot, retirement values, queue-use flags, and prospective signal value unchanged for
retry.

A malformed, consumed, or stale token faults `INVALID_HANDLE`. `begin_frame`
while active faults `INVALID_RESOURCE_STATE` before changing frame state. This
is particularly important with one frame in flight, where the next slot is the
active slot itself. `frame.is_valid()` reports whether the token contains a
generation; it does not validate a stale alias. Tokens are stack-only, at most
16 bytes, and add no heap allocation or new atomic operation to the allocation
path.
Retirement waits and counter queries complete before frame state is committed.
A timeout or backend query failure therefore leaves the frame index, arena
cursor, retirement counters, pool accounting, and output token unchanged.
Expected retire-wait timeouts return `WAIT_TIMEOUT` silently for retry,
matching other expected semaphore/fence timeout paths. Non-timeout backend
failures emit a structured diagnostic identifying the public operation but do
not represent a `FrameToken` as slot identity. Alignment validation
precedes zero-size validation when both inputs are invalid.

```text
FrameArenaState
    gpu::vk::BufferHandle backing_buffer
    GpuAddress gpu_base
    void* cpu_base
    usz size
    QueueRoles access
    Atomic{usz} cursor
    ulong frame_timeline_value
```

Frame-upload, persistent-arena, and descriptor-buffer storage require
host-coherent memory, so their mapped writes need no flush. Independent
allocations use the span visibility operations; transfer arenas handle
visibility internally.

Frame spans admit the selected graphics and compute roles, or transfer on a
transfer-only device. The backing buffer uses the exact deduplicated families for
those roles: one family stays exclusive and two or more use concurrent sharing.
Barriers, completion-point ordering, and lifetime remain explicit.

Allocation during `ACTIVE` is lock-free — the cursor is an atomic bumped with
a CAS loop, so worker threads allocate concurrently (see docs/threading.md):

```text
alloc_frame_span(token, size, align)
    if token is not the active generation: return INVALID_HANDLE
    retry:
    if cursor > arena.size: return ARENA_FULL
    remainder = cursor & (align - 1)
    padding = remainder == 0 ? 0 : align - remainder
    if padding > arena.size - cursor: return ARENA_FULL
    aligned = cursor + padding
    if size > arena.size - aligned: return ARENA_FULL
    next_cursor = aligned + size
    if !compare_exchange(cursor, next_cursor): goto retry
    span = backing_span.unchecked_subspan(aligned, size)
```

Reset:

```text
reset only after queue timeline >= frame_timeline_value
cursor = 0
```

## 11. Persistent arenas

Persistent arenas use VMA virtual allocator to suballocate ranges from large real buffers.

The backing buffer uses the fixed usage superset `transfer_src`,
`transfer_dst`, `uniform`, `storage`, `addressable`, `indirect`, and
`index`; `vertex` is excluded. It admits every selected role. Distinct
families use the exact ordered concurrent-sharing list; one family remains
exclusive.

`PersistentAllocDesc` specifies size, alignment, `PERSISTENT_UPLOAD`,
semantic access, and a debug name. `access` must be a non-empty subset of
selected roles. Each returned span carries generation-checked identity;
device-owned metadata stores its bounds and access.

Persistent backing is host-coherent, so mapped writes need no flush. Callers
still record barriers, order submissions with completion points, wait for
completion, and keep each span live until all referencing work retires.
`free_persistent_span` is valid only after that retirement.
Persistent virtual-allocation exhaustion keeps the `ARENA_FULL` fault and
reports the originating backend result. A free whose backing identity or complete owning
range was changed faults `INVALID_ARGUMENT`; an unknown, stale, or already-freed
allocation faults `INVALID_HANDLE`.

```text
PersistentArenaState
    gpu::vk::BufferHandle backing_buffer
    vma::VirtualBlock virtual_block
    GpuAddress gpu_base
    void* cpu_base
    usz size
    PersistentAllocationTable allocations
```

Allocation flow:

```text
1. Build VirtualAllocationCreateInfo with size and alignment.
2. Allocate from vma::VirtualBlock.
3. Publish immutable range metadata and return its GpuSpan.
```

Free flow:

```text
1. Resolve the live allocation identity and exact owning range.
2. After every referencing GPU access retires, free the virtual allocation.
3. Retire the allocation generation.
```

## 12. Readback arena

Buffer readback consumes an exact source span. `readback_buffer_data` requires
`out_data.len == src.size`. Use `checked_subspan` for a partial readback.
Internal readback storage supports CPU invalidation before reads.

Blocking flow (the `readback_buffer_data` / `readback_texture_data` helpers):

```text
1. Allocate an internal readback range.
2. Record the GPU copy into that range.
3. Record barrier transfer write -> host read if needed by backend policy.
4. Submit and wait for timeline.
5. Invalidate the allocation range if non-coherent.
6. Read the range returned by `get_span_mapping`.
```

For non-blocking readback, allocate `CPU_READ` memory, record the copy and a
`TRANSFER_WRITE` to `HOST_READ` barrier on its destination span, submit, and
poll the returned completion point. After completion,
invalidate the span and read its mapping. The caller controls allocation reuse
and release.

## 13. Staging arena

The staging arena uploads data without creating many short-lived native allocations.

`cmd_upload_buffer` and `upload_buffer_data` consume an exact destination span
and require `data.len == dst.size`. Use `checked_subspan` for a partial upload.

The staging and readback arenas place ranges with monotonic virtual offsets and
map them to physical offsets with `virtual_start % arena_size`. Placement is
planned before retire state is changed: alignment padding, the physical wrap
gap, live-window capacity, and `virtual_end` are each checked with
subtraction-first bounds before any addition is formed. An unrepresentable
range behaves as `ARENA_FULL`, so the existing dedicated-buffer fallback
remains available.

Live virtual offsets are never wrapped. When retirement empties the queue, the
next successful arena allocation normalizes the stale head and tail to zero.

Upload flow:

```text
1. Allocate staging span.
2. Copy source bytes to the range returned by `get_span_mapping`.
3. Flush if non-coherent.
4. Record copy to destination.
5. Caller records destination next-use barrier.
6. Staging span is recycled after submit timeline retires.
```

The staging and readback rings and dedicated transfer fallbacks are `EXCLUSIVE`
when their admitted roles select one family. With multiple admitted families
they are `CONCURRENT` across the exact deduplicated device order: graphics,
compute, then transfer.

Concurrent sharing removes queue-family ownership transfers only. Host
flushes, barriers, submission ordering, queue completion, timeline retirement,
and ring locking remain required. Each ring remains scoped to its owning
`Device`.

### 13.1 Which timeline retires a range

Every staging range, dedicated staging buffer, and readback range carries a
timeline tag decided at allocation time: `FRAME` (`frame_timeline`, tagged
`counter + 1`) for in-frame command-list paths (`cmd_upload_buffer`,
`cmd_upload_texture`), or `HELPER` (`helper_timeline`) for the blocking
helpers (`upload_buffer_data`, `upload_texture_data`, `readback_buffer_data`,
`readback_texture_data`). A blocking helper reserves its retire value once,
under `transfer_mutex`, before its (single) allocation, and every range or
dedicated buffer it tags carries that exact value — never the value another
concurrent helper reserved, and never `frame_timeline`. Drains compare each
entry against its own tagged timeline's current counter value, so one
helper's completion can never retire another helper's or an unsubmitted
list's ranges. See docs/threading.md §Helper timeline for the
completion-side turnstile.

Blocking helpers validate resource access against their single internal
transfer role rather than a recording queue's merged role set, so a span
admitted on a multi-role queue recording may still be rejected by a helper;
this stricter check is intentional and deterministic across queue topologies.

## 14. Mapped visibility

Call `flush_mapped_span` after CPU writes and before GPU use. After waiting or
polling the relevant completion point, call `invalidate_mapped_span` before CPU
reads. Neither operation waits.

Both operations require a live, mapped independent-allocation span. Coherent
memory returns success without native work. The backend rounds non-coherent
ranges to atom boundaries and clamps the final atom to the native allocation.

## 15. Memory budget and statistics

Public API:

```text
MemoryHeapBudget
    ulong usage
    ulong budget
    ulong allocation_bytes
    ulong block_bytes

MemoryStats
    MemoryHeapBudget[MAX_MEMORY_HEAPS] heaps
    uint heap_count
    ulong texture_count
    ulong live_allocation_count

get_memory_stats(Device* device) -> MemoryStats?
build_memory_report(Device* device, bool detailed) -> String?
```

Backend sources:

```text
vma::Allocator.heap_budgets
vma::Allocator.statistics
vma::Allocator.stats_string
live slot tables
```

Call current-frame-index update during `begin_frame` so VMA budget tracking remains useful.

## 16. Allocation names and user data

Debug builds should set:

```text
Vulkan object name
VMA allocation name
VMA allocation user data or backend side table
```

Recommended allocation name format:

```text
allocation:<debug_name>
buffer:<debug_name>
texture:<debug_name>
arena:frame_upload:<frame_index>
arena:persistent_upload
arena:staging
arena:readback
```

## 17. Immediate resource lifetime

`free_allocation` and non-WSI core resource destruction release native ownership
immediately. They never wait and never enqueue deferred release work. The caller
must first discard recording or executable command tokens and wait for every
submitted completion point that may reference the resource.

Validation tracks explicitly named spans, textures, and pipelines. A detected
reference returns `RESOURCE_IN_USE` without consuming ownership. References
reachable only through GPU addresses or shader indices cannot be enumerated and
remain a caller precondition.

A texture with live descriptors and an allocation with live placed or dedicated
textures also return `RESOURCE_IN_USE`. Descriptor and sampler indices recycle
immediately; stale shader data is therefore a caller lifetime violation.
`destroy_device` remains non-blocking and returns `DEVICE_BUSY` while queue
work is incomplete.

## 18. Defragmentation policy

There is no automatic defragmentation. GPU addresses may be stored in root
structs, tables, and indirect records, so allocations do not move.

## 19. Memory acceptance criteria

The memory layer is acceptable when:

```text
all Vulkan buffers/images are VMA-backed
no raw vkAllocateMemory path exists in backend code
independent allocations provide generation-checked ownership and range queries
addressable spans return non-zero GpuAddress
frame spans reset only after timeline retirement
persistent spans support allocation/free/reuse
readback path invalidates non-coherent memory
staging path flushes non-coherent memory
memory stats report VMA budget and live resources
allocation names appear in debug reports
resource destruction never waits or queues deferred work
```
