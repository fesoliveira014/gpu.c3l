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
```

The backend uses:

```text
vma::Allocator
vma::Allocation
vma::AllocationInfo
vk::Buffer
vk::Image
vk::DeviceAddress
```

`vma::` and `vk::` types are never public API types.

## 2. VMA integration boundary

Only the Vulkan backend imports `vma`.

Backend files that may import `vma`:

```text
gpu/internal/vk/allocator.c3
gpu/internal/vk/allocation.c3
gpu/internal/vk/buffer.c3
gpu/internal/vk/texture.c3
gpu/internal/vk/debug.c3
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
```

`gpu.c3l` responsibilities:

```text
define behavioral memory classes
track owning allocations and non-owning span identities
keep mappings, addresses, access, and native backing private
translate resource policy to Vulkan/VMA
leave allocation reuse and pooling policy to callers
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
```

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
generic data from an allocation and stores only
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

Applications that keep mapped spans and their GPU addresses together can use
the root-module `MappedGpuSpan` convenience value:

```c3
gpu::MappedGpuSpan mapped = gpu::mapped_gpu_span(device, packed)!;
gpu::MappedGpuSpan vertices =
    mapped.checked_subspan(0, vertex_bytes)!;
vertices.bytes[..] = vertex_data[..];
gpu::flush_mapped_span(device, vertices.span)!;
```

`MappedGpuSpan` is a non-owning bundle of the span, its mapped byte slice, and
its GPU address. Its public invariant is `bytes.len == span.size`; checked
children reject inconsistent aggregates and keep all three ranges consistent
without another device lookup. Derived byte pointers carry no alignment
guarantee beyond `char`. The factory performs independent public mapping and
address queries, so the caller must keep the allocation live throughout the
call. The bundle does not retain the allocation or perform synchronization:
its fields expire when the allocation is freed, and callers still flush or
invalidate mapped ranges and wait for GPU completion where the memory policy
requires it.

## 6. Public memory policy

| Need | Public API | Contract |
|---|---|---|
| CPU-written generic data | `MemoryClass.CPU_WRITE` | map, write, flush, then `submit` |
| GPU-private generic data | `MemoryClass.GPU_PRIVATE` | upload or write from GPU commands |
| CPU-read generic data | `MemoryClass.CPU_READ` | wait, invalidate, then read |
| Placed textures | `MemoryClass.TEXTURE` | query requirements, allocate, create placed textures |
| Sparse texture pages | `MemoryClass.TEXTURE` | create sparse texture, query its page requirement, allocate compatible caller-owned pools |
| Long-lived CPU-written tables | `MemoryClass.CPU_WRITE` | map, write, flush, submit, wait or poll, then free or reuse |

Long-lived CPU-written storage follows the independent-allocation contract:
allocate `CPU_WRITE` memory, borrow its span, mapping, and address as needed,
write, flush, record and submit, wait for or poll the covering completion point,
then free the owning `GpuAllocation`. Do not infer coherence from the memory
class.

## 7. Private buffer backing

Generic allocations use private addressable Vulkan buffers;
texture allocations contain only compatible image memory. Generic allocations
use a fixed native-usage superset. Queue-family sharing derives from the
immutable `QueueRoles` access set. Creation publishes an allocation or span
identity only after its native buffer, VMA allocation, mapping state, and
nonzero device address are complete.

Private buffer records support this implementation. They are not public
resource types.

## 8. Texture allocation

Public descriptor:

```text
TextureDesc
    uint width
    uint height
    uint depth
    uint mip_levels
    uint array_layers
    Format format
    TextureUsage usage
    QueueRoles access
    SampleCount sample_count
    ZString debug_name
```

Zero depth selects a 2D image with native depth one. Positive depth selects a
3D image, including a genuine depth-one 3D image. Three-dimensional textures
use one normalized array layer and one sample, may be sampled, stored, or
transferred, and cannot be color or depth attachments. Their mip chain reduces
width, height, and depth independently.

Render attachments use a separate explicit child:

```text
AttachmentViewDesc
    TextureHandle texture
    uint mip_level
    uint array_layer
```

`create_attachment_view` validates the selected subresource, retains the
texture, and creates any required native image view before command recording.
The returned `AttachmentViewHandle` is immutable and device-owned. It is not a
shader-visible `TextureView` and consumes no descriptor-heap slot.

Backend translation:

```text
TextureUsage.SAMPLED      -> VK_IMAGE_USAGE_SAMPLED_BIT
TextureUsage.STORAGE      -> VK_IMAGE_USAGE_STORAGE_BIT
TextureUsage.COLOR_ATTACH -> VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
TextureUsage.DEPTH_ATTACH -> VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
TextureUsage.TRANSFER_SRC -> VK_IMAGE_USAGE_TRANSFER_SRC_BIT
TextureUsage.TRANSFER_DST -> VK_IMAGE_USAGE_TRANSFER_DST_BIT
SampleCount               -> VkSampleCountFlagBits
```

Multisample textures are attachment-only, have one mip, and require an
adapter-supported sample count; they are therefore 2D. Resolve destinations are separate
single-sample color-attachment textures.

Owned creation:

```text
validate descriptor and adapter support
create image and allocation transactionally
create the default view when usage is sampled, storage, or attachment-capable
publish TextureHandle
```

Transfer-only textures publish without a default image view. Their image,
allocation, placement, transfer, barrier, and destruction behavior is otherwise
unchanged for both dimensions. View-capable 3D textures create a full-volume
`TYPE_3D` default view; selected mip views retain full depth and one array layer.

Sparse creation:

```text
create_sparse_texture on a sparse-enabled device
query the immutable SparseTextureRequirements snapshot
allocate compatible MemoryClass.TEXTURE page pools as needed
bind color tiles and required color/metadata tails
order GPU use after the returned CompletionPoint
order unbind or replacement after every prior user
wait before reusing or freeing backing bytes
```

Sparse creation owns only a raw unbound image and optional default view. It
does not ask VMA to allocate or bind image memory. `virtual_size` describes the
image's virtual byte span and is not reported as physical allocation size.
`page_allocation.size` and `.alignment` both name one native sparse block;
compatibility retains the device owner and native memory-type mask, and
`dedicated_only` is false. COLOR is always the first aspect and METADATA is
optional. The snapshot is stored by value in the texture slot and later queries
perform no native call.

`bind_sparse_texture_memory` resolves each named allocation to its native
device memory and absolute offset while holding the resource lock. A live
backing must be non-dedicated `MemoryClass.TEXTURE` memory compatible with the
page requirement, selected memory type, texture access roles, alignment, and
requested range. The exact invalid allocation with zero offset is the unbind
sentinel. Tile backing charges a complete page for every covered tile,
including partial edge tiles; opaque tail sizes need not be page multiples.

Backing ownership never transfers. The call retains the sparse texture through
the returned completion but keeps no residency map, allocation reference, or
deferred backing free after `vkQueueBindSparse` returns. Keep allocations live
through the returned completion and keep resident bytes exclusive until an
ordered unbind or replacement and all prior GPU use have completed.
`free_allocation` does not implicitly unbind and cannot diagnose historical
sparse binding with `RESOURCE_IN_USE`.

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
Owned, placed, and dedicated 3D textures use these same compatibility,
alignment, overlap, and allocation-ownership rules.

## 9. Texture lifetime

Destroying an owned texture releases its image allocation immediately.
Destroying a placed texture releases the image but not its `GpuAllocation`.
`free_allocation` returns `RESOURCE_IN_USE` while a placed image is live.
Dedicated creation returns separate texture and allocation tokens; destroy the
texture before releasing its allocation.

Destroying a sparse texture releases its raw image and cached/default views.
It never frees, decrements, or otherwise mutates caller-owned page allocations.
Descriptor publication does not establish residency, and sparse creation adds
no implicit allocation, binding, completion wait, or resident-region tracking.
Use a region only after a successful ordered bind has established residency.
An incomplete bind retains the image and makes destruction return
`RESOURCE_IN_USE`. Destruction is safe only after all bind operations and GPU
users complete and still does not release caller-owned backing pools.

A live user-created attachment view retains its texture, so `destroy_texture`
returns `RESOURCE_IN_USE` until every view is destroyed. A borrowed swapchain
view does not add a retain because the swapchain owns and invalidates its texture
and view together. A view referenced by a recording, executable, or submitted
command list returns `RESOURCE_IN_USE` from its owning lifecycle operation. Wait
for covering completion before destroying a user-created view or resizing or
destroying a swapchain. Non-default native subresource views are destroyed
exactly once with their public child; pass recording never creates or caches
them.

Texture layout transitions do not change allocation ownership, mapping
visibility, command retention, or completion-based lifetime.
`TextureState.layout` remains caller-owned operational history for each
texture or independently transitioned subresource range. The backend stores no
global layout history; explicit texture barriers establish required transfer,
sampled, storage, attachment, initialization, and presentation layouts.

## 10. Caller-owned transient data

Roots, constants, transfer payloads, and other short-lived GPU data use ordinary
allocations. The library does not define an application work boundary, choose
how many copies exist, or reset caller storage.

For CPU-authored data:

```text
allocate CPU_WRITE storage
borrow its GpuSpan and mapping
write the mapped bytes
flush_mapped_span before GPU consumption
record and submit work
retain the returned CompletionPoint
wait or poll before reusing or freeing the allocation
```

For GPU-private data, copy from caller-owned `CPU_WRITE` storage and retain both
allocations until the copy's completion covers their last use. For readback,
copy into `CPU_READ` storage, wait for completion, invalidate the mapped span,
and only then read it.

Applications may build rings, pools, or workload-local allocators from
`GpuAllocation`, `GpuSpan`, and `CompletionPoint`. That policy remains
outside the root module. Each allocation declares the queue roles that may use
it; sharing mode follows those roles but does not replace barriers or
submission ordering.

## 11. Host transfers

Transfer storage is caller-owned. Uploads use `CPU_WRITE` allocations: borrow
the span, mapping, and address as needed, write, flush, record and submit the
copy, wait for or poll the covering completion point, then free or reuse the
owning allocation.
Readback uses `CPU_READ` allocations: record the copy and a global barrier with
`before.transfer` and `after.host`, submit, wait or poll, invalidate, then
read the mapping before freeing or reusing the owning allocation.

Buffer-texture copy spans can cover 2D array layers or 3D depth slices. Zero
width/height selects the remaining selected-mip extent from x/y; for 3D, zero
depth likewise selects the remainder from z. A nonzero row length adds padding
between rows, and the caller-owned span must contain every padded row across
every selected layer or depth slice. There is no independent slice-pitch
field: slices are separated by the copied height.

The core does not allocate transfer storage, choose fallback policy, or create
additional completion state. Applications may implement pooling and reuse over
allocations, spans, commands, and completion points.

## 12. Mapped visibility

Call `flush_mapped_span` after CPU writes and before GPU use. After waiting or
polling the relevant completion point, call `invalidate_mapped_span` before CPU
reads. Neither operation waits.

Both operations require a live, mapped independent-allocation span. Coherent
memory returns success without native work. The backend rounds non-coherent
ranges to atom boundaries and clamps the final atom to the native allocation.

## 13. Memory budget and statistics

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

## 14. Allocation names and user data

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
```

## 15. Immediate resource lifetime

`free_allocation` and non-WSI core resource destruction release native ownership
immediately. They never wait and never enqueue deferred release work. The caller
must first discard recording or executable command tokens and wait for every
submitted completion point that may reference the resource.

Validation tracks explicitly named spans, textures, and pipelines. A detected
reference returns `RESOURCE_IN_USE` without consuming ownership. References
reachable only through GPU addresses or shader indices cannot be enumerated and
remain a caller precondition.

A texture with live views and an allocation with live placed or dedicated
textures also return `RESOURCE_IN_USE`. Destroying a `TextureView` recycles its
raw shader index immediately; stale shader data is therefore a caller lifetime
violation. Sampler indices instead remain live until device teardown.

`destroy_device` remains non-blocking and returns `DEVICE_BUSY` while queue
work is incomplete.

## 16. Defragmentation policy

There is no automatic defragmentation. GPU addresses may be stored in root
structs, tables, and indirect records, so allocations do not move.

## 17. Memory acceptance criteria

The memory layer is acceptable when:

```text
all Vulkan buffers/images are VMA-backed
no raw vkAllocateMemory path exists in backend code
independent allocations provide generation-checked ownership and range queries
addressable spans return non-zero GpuAddress
caller-owned transient allocations remain live until their completion points finish
long-lived CPU-written data remains caller-owned through GPU completion
host transfer paths flush and invalidate non-coherent memory
memory stats report VMA budget and live resources
allocation names appear in debug reports
resource destruction never waits or queues deferred work
```
