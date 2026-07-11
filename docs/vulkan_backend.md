# gpu.c3l Vulkan Backend

## 1. Purpose

The Vulkan backend implements the `gpu` public API on Vulkan 1.3. It lives under:

```c3
module gpu::vk;
```

It imports:

```c3
import gpu;
import vk;
import vma;
import spvreflect;
```

`spvreflect` is used by `vk/shader.c3` for SPIR-V reflection.

No `vk::` or `vma::` type should appear in public `gpu` API signatures.

## 2. Backend files

```text
vk/backend.c3              loader/VMA link probes, backend availability
vk/instance.c3             instance creation, validation layers, debug messenger
vk/device.c3               physical device selection, logical device, feature chain
vk/queue.c3                queue family selection, queue handles, submit
vk/allocator.c3            vma::Allocator creation/destruction, stats
vk/memory.c3               memory kind policy, arenas, virtual allocator
vk/buffer.c3               VkBuffer + VMA allocation path
vk/texture.c3              VkImage + VMA allocation, views, layout tracking
vk/descriptor_heap.c3      descriptor buffer or descriptor indexing implementation
vk/shader.c3               SPIR-V modules and reflection validation
vk/pipeline_cache.c3       pipeline dedup cache and driver cache
vk/pipeline_compute.c3     compute pipeline creation
vk/pipeline_graphics.c3    graphics pipeline creation
vk/command.c3              command buffers and command recording
vk/transfer.c3             upload/readback helpers and staging arenas
vk/sync.c3                 barriers, timeline semaphores
vk/render_pass.c3          dynamic rendering
vk/swapchain.c3            WSI and swapchain
vk/debug.c3                debug names, leak reports
vk/helpers.c3              enum and flag translation helpers
```

## 3. Required Vulkan features

The backend should require:

```text
Vulkan 1.3
buffer device address
synchronization2
dynamic rendering
timeline semaphores
shaderInt64
multiDrawIndirect
shaderDrawParameters
```

Descriptor path:

```text
default (AUTO): descriptor indexing
opt-in: descriptor buffer (DescriptorHeapMode.DESCRIPTOR_BUFFER)
```

`AUTO` prefers indexing: lavapipe (Mesa 25.0.7) miscompiles descriptor-buffer image access, so descriptor buffer is never auto-selected (`resolve_heap_mode` in `vk/device.c3`).

Device creation should fail with `UNSUPPORTED_FEATURE` if required features are missing.

## 4. Instance creation

Instance creation responsibilities:

```text
select Vulkan API version
collect required instance extensions
collect validation layer names when enabled
create vk::Instance
load extension entry points
install debug utils messenger when enabled
```

The backend should support a headless path with no surface extensions and a windowed path with platform-specific surface extensions.

## 5. Physical device selection

Selection criteria:

```text
must support Vulkan 1.3
must support buffer device address
must support synchronization2
must support dynamic rendering
must support timeline semaphores
must support shaderInt64
must support multiDrawIndirect
must support shaderDrawParameters
must support the heap non-uniform-indexing features
must resolve a heap mode from the requested DescriptorHeapMode
must expose a queue family supporting graphics and compute
```

Scoring is by device type only (`score_device`):

```text
discrete > integrated > virtual > cpu > other
```

Device selection result:

```text
pick_physical_device(instance, desc) -> PhysicalDeviceSelection?
```

Each feature-compatible candidate's queue topology is resolved once during selection. The winning `PhysicalDeviceSelection` carries that cached `QueueFamilies` value into logical-device creation; queue topology remains a suitability filter rather than a scoring bonus.

## 6. Logical device creation

Logical device creation builds a Vulkan feature chain.

Required features (device rejected without them, always enabled):

```text
bufferDeviceAddress
synchronization2
dynamicRendering
timelineSemaphore
shaderInt64
multiDrawIndirect
shaderDrawParameters
runtimeDescriptorArray
shaderSampledImageArrayNonUniformIndexing
shaderStorageImageArrayNonUniformIndexing
shaderStorageImageReadWithoutFormat
shaderStorageImageWriteWithoutFormat
```

`maintenance4` is always enabled.

Optional base features are queried on the selected device and enabled only
when advertised:

```text
fillModeNonSolid -> DeviceCaps.line_polygon_mode
```

`PolygonMode.LINE` faults `UNSUPPORTED_FEATURE` before shader or cache lookup
when this cap is false. The feature is not a physical-device selection requirement.

Heap-path-dependent features:

```text
descriptorBuffer
descriptorBindingPartiallyBound
descriptorBindingUpdateAfterBind
```

The exact feature structs depend on the Vulkan headers exposed by `vk.c3l`.

## 7. VMA allocator integration

After logical device creation:

```text
create vma::Allocator
store in VkDeviceState
```

Allocator create info should include:

```text
physical_device
device
instance
vulkan_api_version
buffer-device-address allocator flag
memory-budget allocator flag when supported
```

All Vulkan buffer/image memory must be allocated through VMA.

Backend code should use idiomatic VMA wrappers where possible:

```text
vma::try_create_allocator
allocator.try_create_buffer
allocator.try_create_image
allocator.try_map
allocator.try_flush
allocator.try_invalidate
allocator.heap_budgets
allocator.stats_string
```

## 8. Buffer implementation

Creation flow:

```text
public BufferDesc
    -> validate
    -> translate BufferUsage to vk::BufferUsageFlags
    -> translate MemoryKind to vma::AllocationCreateInfo
    -> allocator.try_create_buffer
    -> query mapped pointer from allocation info
    -> query buffer device address if addressable
    -> store BufferSlot
    -> return BufferHandle
```

Addressable buffers must include:

```text
VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
```

If address query returns zero, creation should fail.

## 9. Texture implementation

Creation flow:

```text
public TextureDesc
    -> validate dimensions and format
    -> translate usage to vk::ImageUsageFlags
    -> allocator.try_create_image
    -> create default image view
    -> set initial layout
    -> store TextureSlot
    -> return TextureHandle
```

The backend tracks image layout per texture. For complex subresource layout tracking, begin with whole-image layout tracking and add subresource tracking only when required.

## 10. Descriptor heap implementation

### Descriptor indexing path

Default under `DescriptorHeapMode.AUTO`.

Uses:

```text
large descriptor arrays
runtime descriptor array support
partially bound descriptors
update-after-bind where needed
```

The indexing layout contains `T` sampled images, `T` storage images, and `S`
samplers, all visible to every stage. Before object creation the backend checks
the exact resource total `2T` against
`maxPerStageUpdateAfterBindResources`; plain `SAMPLER` descriptors do not count
toward that limit. Its single update-after-bind pool contains `2T + S`
descriptors and is checked against `maxUpdateAfterBindDescriptorsInAllPools`.
Requests that exceed either aggregate or any per-type limit fail with
`INVALID_ARGUMENT`; capacities are never clamped.

### Descriptor buffer path

Opt-in via `DescriptorHeapMode.DESCRIPTOR_BUFFER`. `AUTO` never selects it: lavapipe (Mesa 25.0.7) miscompiles descriptor-buffer image access.

Backend owns descriptor buffers for:

```text
sampled images
storage images
samplers
```

Public indices map to descriptor entries. Neither path changes the public API or shader material records.

### Descriptor slot policy

```text
DescriptorSlot
    uint index
    ushort generation
    DescriptorKind kind
    bool used
    TextureHandle owner_texture
```

Initial policy should validate descriptor use in debug builds and report leaked descriptors at device destruction.

## 11. Shader and pipeline implementation

### Shader modules

The backend consumes SPIR-V bytes.

Responsibilities:

```text
create vk::ShaderModule
store stage and entry point
run reflection validation
set debug names
```

Reflection validation checks:

```text
entry point exists and matches the declared stage
push constant blocks fit the stage's root-push range
unexpected descriptor sets are rejected
descriptor heap bindings match convention
```

### Compute pipeline

Compute pipeline creation:

Before shader lookup, reject a push-constant size below `RootPush::size`, not
divisible by four, or above `DeviceCaps.max_push_constant_size`. These public
input faults return `INVALID_ARGUMENT` before any Vulkan call.

```text
shader module
pipeline layout with root pointer push constant
global descriptor heap layout or descriptor buffer binding convention
vk::Pipeline
```

### Graphics pipeline

Graphics pipeline creation:

```text
vertex shader
fragment shader
pipeline layout with vertex/fragment root push constants
color/depth formats for dynamic rendering
raster/depth/blend state
viewport/scissor dynamic state
pipeline cache lookup
vk::Pipeline
```

### Pipeline cache

Two layers. A descriptor-keyed dedup cache (`PipelineKey` over immutable state,
with refcounted aliases) sits in front of a driver `vk::PipelineCache`. The
driver cache is created with `DeviceDesc.pipeline_cache_data` as initial data
and exported through `get_pipeline_cache_size` / `get_pipeline_cache_data`.

Compute pipeline layouts are shared per push-constant size in a packed
device-owned cache. Host storage uses pipeline capacity as an initial hint and
grows to the device's finite valid-size count.

## 12. Command buffers

Command pool policy:

```text
one command pool per frame per queue family
reset pools when frame retires
```

Recording-context pool sets are constructed transactionally: ownership is
published only after every per-frame graphics, optional distinct-compute, and
transfer pool exists. Any create fault destroys all earlier pools and releases
the host arrays. Shared-family compute aliases graphics only after success.

Public command token:

```text
CommandList
    Device* device
    CommandListHandle handle
```

The handle resolves through a fixed 4096-entry device table to a backend
`CommandRecord` containing the `vk::CommandBuffer`, recording context, queue,
frame-slot index, lifecycle state, last-bound pipeline cache, and a growable
array of pending texture-layout transitions. The public token stays within two
machine words; copying it creates an alias, not an independent recorder.

Begin/end validate state transitions. Submit preflights a whole batch under the
command-table mutex, rejects duplicate or foreign-owner tokens, and claims every
record as `SUBMITTING` before constructing the Vulkan submission. A fault before
`vkQueueSubmit2` succeeds restores claimed records to `EXECUTABLE`. Success commits layout
transitions in submission order and frees the records, invalidating all aliases.
Frame-slot pool reset reclaims any unsubmitted records before resetting their
Vulkan command pool. A recording context cannot be destroyed while one of its
records remains live.

## 13. Synchronization

Use synchronization2 for barriers.

Translation helpers:

```text
stage_to_vk
hazard_to_access
texture_layout_to_vk
barrier_to_vk_dependency_info
```

Barrier commands:

```text
cmd_buffer_barrier -> vk::BufferMemoryBarrier2
cmd_texture_barrier -> vk::ImageMemoryBarrier2
cmd_global_barrier -> vk::MemoryBarrier2
```

The backend must not insert hidden barriers for user-visible resource transitions except for unavoidable swapchain acquire/present transitions inside WSI helpers.

## 14. Timeline semaphores

Use timeline semaphores for:

```text
frame retirement
queue submission order
arena reset safety
deferred destruction safety
```

Public submit descriptor should support waits and signals:

```text
SubmitDesc
    CommandList[] command_lists
    SemaphoreWait[] waits
    SemaphoreSignal[] signals
```

## 15. Render pass implementation

Use dynamic rendering.

Render pass begin:

```text
validate color/depth targets
transition only if caller explicitly requested via barrier before begin
build rendering attachment infos
vkCmdBeginRendering
```

Render pass end:

```text
vkCmdEndRendering
```

Do not transition attachments to shader-read automatically.

## 16. Swapchain implementation

Swapchain module owns:

```text
vk::SurfaceKHR
vk::SwapchainKHR
vk::Image[]
vk::ImageView[]
Format format
uint width
uint height
present mode
```

Surface creation is platform-specific. SDL3 samples should create windows and provide native handles or use a sample helper that calls backend WSI functions.

Swapchain operations:

```text
create
acquire
present
query present-mode support
resize on out-of-date/suboptimal; recreate the surface on surface-lost
```

`vk_get_present_mode_support` queries the retained `vk::SurfaceKHR` for fifo/immediate/mailbox availability. At creation, `select_present_mode` falls back to FIFO silently when the requested mode is unavailable.

WSI result mapping is explicit and pure-tested:

| Vulkan result | Public outcome | State/recovery |
|---|---|---|
| acquire `TIMEOUT` / `NOT_READY` | `WAIT_TIMEOUT` | pending-acquire state stays unchanged; retry |
| `ERROR_OUT_OF_DATE_KHR` | `SWAPCHAIN_OUT_OF_DATE` | resize the swapchain |
| `ERROR_SURFACE_LOST_KHR` | `SURFACE_LOST` | replace the platform surface and swapchain |
| acquire `SUBOPTIMAL_KHR` | valid image with `suboptimal = true` | finish the frame; resize when convenient |
| present `SUBOPTIMAL_KHR` | success | current public API exposes no soft present result |

## 17. Debug implementation

Debug features:

```text
Vulkan object names
VMA allocation names
live slot reports
leaked descriptor reports
allocation stats
validation message routing
optional command labels
```

Object naming should happen immediately after successful backend object creation.

## 18. Deferred destruction

Resources may be destroyed publicly before the GPU has finished using them.

Backend policy:

```text
remove public handle immediately
push backend object to deferred destruction list with retire timeline
free backend object after timeline completed
```

For VMA-backed resources:

```text
buffer: allocator.destroy_buffer(buffer, allocation)
image: allocator.destroy_image(image, allocation)
```

## 19. Translation helpers

All enum/flag conversion should live in `vk/helpers.c3`.

Helpers:

```text
format_to_vk
buffer_usage_to_vk
texture_usage_to_vk
memory_kind_to_vma
stage_to_vk
hazard_to_vk_access
layout_to_vk
filter_to_vk
address_mode_to_vk
compare_op_to_vk
blend_factor_to_vk
blend_op_to_vk
topology_to_vk
```

Do not duplicate translation switches in command or resource files.

## 20. Backend acceptance criteria

The Vulkan backend is acceptable when:

```text
device creation is validation-clean
all buffers/images are VMA-backed
addressable buffers produce valid GPU addresses
root-pointer compute works
texture heap works through TextureIndex
barriers use synchronization2
offscreen dynamic rendering works
SDL3 swapchain sample presents and resizes
live resource leaks are reported
no vk:: or vma:: type appears in public API signatures
```
