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
vk/allocator.c3
vk/memory.c3
vk/buffer.c3
vk/texture.c3
vk/debug.c3
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
    PersistentArenaState persistent_upload_arena
    PersistentArenaState persistent_device_arena
    StagingArenaState staging_arena
    ReadbackArenaState readback_arena
```

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

```text
FrameArenaState
    BufferHandle backing_buffer
    GpuAddress gpu_base
    void* cpu_base
    usz size
    usz cursor
    ulong frame_timeline_value
```

Allocation:

```text
alloc_frame_span(size, align)
    aligned = align_up(cursor, align)
    if aligned + size > arena.size: return ARENA_FULL
    span.gpu = gpu_base + aligned
    span.cpu = cpu_base + aligned
    span.buffer = backing_buffer
    span.offset = aligned
    span.size = size
    cursor = aligned + size
```

Reset:

```text
reset only after queue timeline >= frame_timeline_value
cursor = 0
```

## 11. Persistent arenas

Persistent arenas use VMA virtual allocator to suballocate ranges from large real buffers.

```text
PersistentArenaState
    BufferHandle backing_buffer
    vma::VirtualBlock virtual_block
    GpuAddress gpu_base
    void* cpu_base
    usz size
    MemoryKind kind
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
3. Free virtual allocation.
4. Mark span invalid in debug tracking.
```

## 12. Readback arena

Readback buffers may be explicit buffers or arena spans. They must support CPU invalidation before reads.

Readback flow:

```text
1. Create readback buffer/span.
2. Record GPU copy into readback resource.
3. Record barrier transfer write -> host read if needed by backend policy.
4. Submit and wait for timeline.
5. Invalidate VMA allocation range if non-coherent.
6. Read CPU pointer.
```

## 13. Staging arena

The staging arena exists to upload large data without creating many short-lived buffers.

Upload flow:

```text
1. Allocate staging span.
2. Copy source bytes to span.cpu.
3. Flush if non-coherent.
4. Record copy to destination.
5. Caller records destination next-use barrier.
6. Staging span is recycled after submit timeline retires.
```

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

## 17. Defragmentation policy

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

## 18. Memory acceptance criteria

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
```
