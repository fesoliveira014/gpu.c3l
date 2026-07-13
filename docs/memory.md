# gpu.c3l Memory Architecture

## 1. Purpose

Memory is the foundation of `gpu.c3l`. The library's shader ABI depends on user code being able to write root data and GPU data structures into memory that shaders can address. The Vulkan backend uses Vulkan Memory Allocator through `vma.c3l` for Vulkan memory allocation, while exposing a smaller public memory model.

The public API uses:

```text
GpuAddress
GpuSpan
BufferHandle
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
choose public memory kind
translate usage to Vulkan/VMA policy
track public handles
track GPU addresses
track CPU pointers
own frame/persistent/staging/readback arena policy
validate lifetimes
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
    FrameArenaState[] frame_arenas
    PersistentArenaState persistent_arena
    TransferArenaState staging_arena
    TransferArenaState readback_arena
    DedicatedRetireState dedicated_staging
```

(One persistent arena, PERSISTENT_UPLOAD only — device-local persistent data
goes through explicit DEVICE buffers, not an arena.)

Allocator creation happens after Vulkan device creation and before buffer/image creation. Allocator destruction happens after all resource destruction queues have been drained.

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

If buffer device address is not supported, device creation should fail. The primary shader ABI depends on addressable buffers.

## 5. Public memory kinds

### 5.1 `MemoryKind.FRAME_UPLOAD`

Short-lived, CPU-written, GPU-read data.

Use for:

```text
root structs
per-frame constants
small per-draw records
small dispatch records
small material updates
```

Properties:

```text
host-visible
persistently mapped
addressable
linear bump allocation
reset after frame timeline retires
```

Vulkan buffer usage:

```text
SHADER_DEVICE_ADDRESS
STORAGE_BUFFER
UNIFORM_BUFFER
TRANSFER_SRC
TRANSFER_DST if needed
```

VMA allocation policy:

```text
usage: AUTO or AUTO_PREFER_HOST
flags: HOST_ACCESS_SEQUENTIAL_WRITE, MAPPED
```

### 5.2 `MemoryKind.PERSISTENT_UPLOAD`

Long-lived CPU-updated, GPU-readable data.

Use for:

```text
material records
small persistent lookup tables
CPU-updated scene metadata
bindless material tables
```

Properties:

```text
host-visible when available
addressable
suballocated through vma::VirtualBlock
explicit allocation/free
```

VMA policy:

```text
usage: AUTO
flags: HOST_ACCESS_SEQUENTIAL_WRITE, MAPPED, HOST_ACCESS_ALLOW_TRANSFER_INSTEAD when appropriate
```

If an allocation is not host-visible because the backend allows transfer fallback, the API must route writes through staging.

### 5.3 `MemoryKind.DEVICE`

GPU-local memory.

Use for:

```text
large static buffers
GPU-written storage buffers
textures
render targets
storage images
large geometry data
```

Properties:

```text
prefer device-local
usually not mapped
updated by explicit copies
addressable for buffers that request BufferUsage.ADDRESSABLE
```

VMA policy:

```text
usage: AUTO_PREFER_DEVICE for buffers
usage: AUTO for images
flags: none unless buffer device address or dedicated allocation is required
```

### 5.4 `MemoryKind.READBACK`

GPU-written, CPU-read memory.

Use for:

```text
compute results
pixel readback
test validation
screenshot data
GPU-generated reports
```

Properties:

```text
host-visible
prefer host-cached
mapped or map-on-demand
requires invalidate before CPU read on non-coherent memory
```

VMA policy:

```text
usage: AUTO or AUTO_PREFER_HOST
flags: HOST_ACCESS_RANDOM, MAPPED
```

### 5.5 `MemoryKind.STAGING`

One-shot or batched upload memory.

Use for:

```text
texture uploads
large buffer uploads
asset streaming
```

Properties:

```text
host-visible
sequential write
TRANSFER_SRC
recycled after timeline retire
```

VMA policy:

```text
usage: AUTO
flags: HOST_ACCESS_SEQUENTIAL_WRITE, MAPPED
```

### 5.6 `GpuSpan` slicing

Use `checked_subspan` when partitioning a span:

```c3
GpuSpan vertices = packed.checked_subspan(0, vertex_bytes)!;
GpuSpan indices = packed.checked_subspan(vertex_bytes, index_bytes)!;
```

Validation is relative to the immediate parent. A nested child therefore
cannot escape an intermediate slice even when it would still fit the original
backing buffer. Bounds use `size <= parent.size - offset` after validating
the offset, avoiding `offset + size` overflow. Derived GPU, CPU, and backing
offset additions are checked separately.

`size == 0` faults `INVALID_ARGUMENT`, matching the other public span
producers and avoiding ambiguity with operations where zero means “to end.”

`unchecked_subspan` is reserved for already-proven hot paths and tests that
deliberately construct invalid spans; no historical unchecked alias remains.

## 6. Buffer allocation

Public descriptor:

```text
BufferDesc
    usz size
    BufferUsage usage
    MemoryKind memory_kind
    ZString debug_name
```

Backend translation:

```text
BufferUsage.ADDRESSABLE  -> VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
BufferUsage.STORAGE      -> VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
BufferUsage.UNIFORM      -> VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
BufferUsage.TRANSFER_SRC -> VK_BUFFER_USAGE_TRANSFER_SRC_BIT
BufferUsage.TRANSFER_DST -> VK_BUFFER_USAGE_TRANSFER_DST_BIT
BufferUsage.INDIRECT     -> VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
BufferUsage.INDEX        -> VK_BUFFER_USAGE_INDEX_BUFFER_BIT
BufferUsage.VERTEX       -> VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
```

Backend creation flow:

```text
1. Validate desc.size > 0.
2. Translate usage flags.
3. Add required transfer/address flags.
4. Build vk::BufferCreateInfo.
5. Build vma::AllocationCreateInfo from MemoryKind.
6. Call allocator.try_create_buffer.
7. Store vk::Buffer, vma::Allocation, vma::AllocationInfo.
8. Store mapped CPU pointer if available.
9. Query and store vk::DeviceAddress if ADDRESSABLE.
10. Set debug object name and allocation name.
11. Return BufferHandle.
```

## 7. Buffer slot

```text
BufferSlot
    vk::Buffer buffer
    vma::Allocation allocation
    vma::AllocationInfo allocation_info
    vk::DeviceAddress gpu_base
    void* cpu_base
    usz size
    BufferUsage usage
    MemoryKind memory_kind
    ushort generation
    bool used
    bool mapped
    bool coherent
    bool pending_destroy
```

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

Backend creation flow:

```text
1. Validate dimensions and format.
2. Build vk::ImageCreateInfo.
3. Build vma::AllocationCreateInfo.
4. Call allocator.try_create_image.
5. Create default vk::ImageView.
6. Store initial layout.
7. Set debug names.
8. Return TextureHandle.
```

## 9. Texture slot

```text
TextureSlot
    vk::Image image
    vma::Allocation allocation
    vma::AllocationInfo allocation_info
    vk::ImageView default_view
    vk::ImageLayout layout
    Format format
    TextureUsage usage
    uint width
    uint height
    uint depth
    uint mip_levels
    uint array_layers
    ushort generation
    bool used
    bool pending_destroy
```

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
    BufferHandle backing_buffer
    GpuAddress gpu_base
    void* cpu_base
    usz size
    Atomic{usz} cursor
    ulong frame_timeline_value
```

Frame-upload memory is allocated host-coherent by requirement (a
HOST_VISIBLE|HOST_COHERENT type is spec-guaranteed), so root writes need no
flush; the same holds for the descriptor-buffer storage. STAGING, READBACK,
and PERSISTENT_UPLOAD keep VMA's memory-type freedom and the explicit
`flush_buffer`/`invalidate_buffer` contract.

Frame-arena backing buffers are created for exactly the selected graphics and
compute families. When those family indices differ, the backend uses concurrent
sharing; when they alias, the buffer remains exclusive even if transfer uses
another family. Concurrent sharing is the correctness-first choice while frame
spans expose no family intent; the explicit-transfer alternative remains
unsupported. Callers do not set `BufferUsage.shared_queues` for frame
spans. This ownership policy does not replace barriers or semaphores: callers
still order host writes and cross-queue GPU accesses explicitly.

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
    span = { gpu_base + aligned, cpu_base + aligned,
             backing_buffer, aligned, size, FRAME_UPLOAD }
```

Reset:

```text
reset only after queue timeline >= frame_timeline_value
cursor = 0
```

## 11. Persistent arenas

Persistent arenas use VMA virtual allocator to suballocate ranges from large real buffers.

The backing buffer is created once with a fixed usage superset (`transfer_src`,
`transfer_dst`, `uniform`, `storage`, `addressable`, `indirect`, `index`, and
`shared_queues`; `vertex` is deliberately excluded). It is shared automatically
across the exact selected graphics, compute, and transfer families. Distinct
families use concurrent sharing with the ordered deduplicated family list; when
all three queues alias one family, the buffer remains exclusive with no family
list.

`PersistentAllocDesc.usage` is validated as a subset of that superset at
allocation time (`INVALID_ARGUMENT` on unsupported bits) rather than applied per
span. `shared_queues` is accepted as a no-op because sharing belongs to the
backing buffer, and empty usage remains a valid storage-style default. Explicit
buffers and textures still require their own `shared_queues` flag when consumed
across distinct queue families.

Concurrent sharing removes queue-family ownership transfers only. Callers must
still flush host writes as required, record barriers, order submissions with
semaphores, wait for completion, and keep each span live until all referencing
work retires. `free_persistent_span` is valid only after that retirement. The
arena and its spans remain scoped to the single supported live `Device`;
multi-device support is outside the contract.
Persistent virtual-allocation exhaustion keeps the `ARENA_FULL` fault and
reports the originating backend result. A free whose buffer is not the arena
backing faults `INVALID_ARGUMENT`; an unknown or already-freed offset faults
`INVALID_HANDLE`. Neither case claims public identity because persistent spans
are value ranges, not generation-checked public slots.

```text
PersistentArenaState
    BufferHandle backing_buffer
    vma::VirtualBlock virtual_block
    GpuAddress gpu_base
    void* cpu_base
    usz size
    PersistentAllocMap allocations   (offset -> virtual allocation, for free-by-span)
```

Allocation flow:

```text
1. Build VirtualAllocationCreateInfo with size and alignment.
2. Allocate from vma::VirtualBlock.
3. Return GpuSpan using returned offset.
```

Free flow:

```text
1. Validate span belongs to arena.
2. Find matching virtual allocation handle.
3. After every referencing GPU access retires, free virtual allocation.
4. Mark span invalid in debug tracking.
```

## 12. Readback arena

Readback buffers may be explicit buffers or arena spans. They must support CPU invalidation before reads.

Blocking flow (the `readback_buffer_data` / `readback_texture_data` helpers):

```text
1. Create readback buffer/span.
2. Record GPU copy into readback resource.
3. Record barrier transfer write -> host read if needed by backend policy.
4. Submit and wait for timeline.
5. Invalidate VMA allocation range if non-coherent.
6. Read CPU pointer.
```

Non-blocking flow (tickets): `cmd_readback_buffer` / `cmd_readback_texture`
record the copy into the caller's command list and return a `ReadbackTicket`
that identifies private readback state. The caller keeps rendering;
`poll_readback` answers readiness without blocking, and `resolve_readback`
copies out and releases the state (faulting
`READBACK_NOT_READY` if polled early). The ticket owns its range until
resolved.

## 13. Staging arena

The staging arena exists to upload large data without creating many short-lived buffers.

The staging and readback arenas place ranges with monotonic virtual offsets and
map them to physical offsets with `virtual_start % arena_size`. Placement is
planned before retire state is changed: alignment padding, the physical wrap
gap, live-window capacity, and `virtual_end` are each checked with
subtraction-first bounds before any addition is formed. An unrepresentable
range behaves as `ARENA_FULL`, so the existing dedicated-buffer fallback
remains available.

Live virtual offsets are never wrapped or rebased because `virtual_end` also
identifies pinned readback-ticket ranges. When retirement leaves the queue
empty, the next successful arena allocation normalizes the stale head and tail
to zero; an outstanding pinned ticket keeps the queue nonempty and therefore
prevents normalization.

Upload flow:

```text
1. Allocate staging span.
2. Copy source bytes to span.cpu.
3. Flush if non-coherent.
4. Record copy to destination.
5. Caller records destination next-use barrier.
6. Staging span is recycled after submit timeline retires.
```

The long-lived staging ring is `EXCLUSIVE` when graphics, compute, and
transfer select one family. With multiple selected families it is
`CONCURRENT` across the exact deduplicated order returned by the device
topology: graphics, compute, then transfer. The readback ring remains
exclusive because device copies overwrite its ranges before host-only reads;
one-shot dedicated staging fallbacks also remain exclusive to their single
consumer.

Concurrent sharing removes queue-family ownership transfers only. Host
flushes, barriers, submission ordering, queue completion, timeline retirement,
and ring locking remain required. This policy assumes one live `Device`.

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

## 14. Flush and invalidate policy

The backend should track whether memory is host-coherent.

CPU write path:

```text
if allocation is non-coherent:
    flush written range before GPU reads
```

CPU read path:

```text
if allocation is non-coherent:
    invalidate read range after GPU writes and before CPU reads
```

The public helpers should expose explicit flush/invalidate for mapped buffers and hide it only in high-level upload/readback convenience functions.

## 15. Memory budget and statistics

Public API:

```text
MemoryHeapBudget
    ulong usage
    ulong budget
    ulong allocation_bytes
    ulong block_bytes

MemoryStats
    MemoryHeapBudget[] heaps
    ulong buffer_count
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
buffer:<debug_name>
texture:<debug_name>
arena:frame_upload:<frame_index>
arena:persistent_upload
arena:staging
arena:readback
```

## 17. Deferred destruction

`destroy_buffer`/`destroy_texture`/`destroy_pipeline`/`destroy_shader`/`destroy_semaphore` free the public handle immediately but cannot free the backend VMA buffer/image, image view, `vk::Pipeline`, shader module, or semaphore right away — a frame already submitted may still reference it. The backend queues those objects (`gpu/vk/deferred.c3`) keyed by `retire_timeline_value` (gpu/memory.c3, the same "safe after" value the descriptor heap and transfer arenas use) and frees each once the frame timeline reaches it. The queue drains on every `begin_frame` (after its wait) and opportunistically on every enqueue; teardown drains everything unconditionally once the device is idle. Destroying a resource never faults on the frames-in-flight window alone — see `RESOURCE_IN_USE` in `gpu/faults.c3`. `destroy_texture` does fault `RESOURCE_IN_USE` when a live `TextureIndex` descriptor still owns the texture; destroy the descriptor first. Retired-but-undrained descriptors (destroyed, no frame boundary since) do not block, and device teardown stays lenient — `report_descriptor_leaks` reports leftovers instead of faulting.

`retire_timeline_value` also defers while off-frame work is pending: `submit`
outside a frame bracket (threading.md Tier E, sanctioned for frame-loop-free
apps) sets an off-frame-pending marker in addition to `begin_frame`/`end_frame`'s
`frame_active` flag, so a destroy of a resource an off-frame submit
referenced enqueues rather than freeing synchronously. `end_frame`'s
cross-queue chain (§ frame retirement across queues, threading.md) waits
every queue used since the last boundary — off-frame or in-frame — before
clearing the marker. The prospective frame value, slot retirement, queue-use
flags, and marker commit only after that signal submit succeeds, so a rejected
submit leaves the boundary unchanged for retry. A frame-loop-free app that
never calls `begin_frame` again holds deferred entries safely until teardown.

## 18. Defragmentation policy

VMA defragmentation is deferred.

Initial policy:

```text
no automatic defragmentation
no defragmentation of addressable buffers
no defragmentation while GPU work may reference allocations
```

Reason: GPU addresses can be stored inside root structs, material tables, and indirect records. Moving an allocation invalidates those addresses unless every reference is rebuilt.

Future policy may support:

```text
manual defrag after device idle
non-addressable resource defrag
rebuild all addressable references after move
```

## 19. Memory acceptance criteria

The memory layer is acceptable when:

```text
all Vulkan buffers/images are VMA-backed
no raw vkAllocateMemory path exists in backend code
addressable buffers return non-zero GpuAddress
frame spans reset only after timeline retirement
persistent spans support allocation/free/reuse
readback path invalidates non-coherent memory
staging path flushes non-coherent memory
memory stats report VMA budget and live resources
allocation names appear in debug reports
destroyed buffers/textures/pipelines/shaders/semaphores free their backend
    object only after retire_timeline_value passes
```
