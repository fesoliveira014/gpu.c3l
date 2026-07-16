# gpu.c3l Vulkan Backend

## 1. Purpose

The Vulkan backend implements the `gpu` public API on Vulkan 1.3. It lives under:

```c3
module gpu::vk @private;
```

Backend declarations are private by default. Only the public dispatch layer and
white-box tests import them with a visibility override.

It imports:

```c3
import gpu;
import vk;
import vma;
import spvreflect;
```

`spvreflect` is used by `gpu/vk/shader.c3` for SPIR-V reflection.

No `vk::` or `vma::` type should appear in public `gpu` API signatures.

## 2. Backend files

```text
gpu/vk/backend.c3              instance/device dispatch loading, loader/VMA link probes
gpu/vk/runtime.c3              per-runtime instance, diagnostics, adapter ownership
gpu/vk/surface.c3              per-runtime WSI dispatch and VkSurfaceKHR operations
gpu/vk/adapter.c3              semantic adapter metadata and diagnostic snapshots
gpu/vk/instance.c3             shared instance construction
gpu/vk/device.c3               physical device selection, logical device, feature chain
gpu/vk/queue.c3                queue family selection, queue handles, submit
gpu/vk/allocator.c3            vma::Allocator creation/destruction, stats
gpu/vk/memory.c3               memory kind policy, arenas, virtual allocator
gpu/vk/buffer.c3               VkBuffer + VMA allocation path
gpu/vk/texture.c3              VkImage + VMA allocation, views, layout tracking
gpu/vk/descriptor_heap.c3      descriptor buffer or descriptor indexing implementation
gpu/vk/shader.c3               SPIR-V modules and reflection validation
gpu/vk/pipeline_cache.c3       pipeline dedup cache and driver cache
gpu/vk/pipeline_compute.c3     compute pipeline creation
gpu/vk/pipeline_graphics.c3    graphics pipeline creation
gpu/vk/command.c3              recording contexts and command encoding
gpu/vk/command_state.c3        command-list state and handle tracking
gpu/vk/transfer.c3             upload/readback helpers and staging arenas
gpu/vk/sync.c3                 barriers, timeline semaphores
gpu/vk/render_pass.c3          dynamic rendering
gpu/vk/swapchain.c3            swapchain lifecycle and presentation
gpu/vk/deferred.c3             retired backend-object destruction
gpu/vk/debug.c3                debug names, leak reports
gpu/vk/helpers.c3              enum and flag translation helpers
gpu/vk/validate.c3             descriptor and command validation helpers
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
AUTO: prefer descriptor indexing, then descriptor buffer
forced: descriptor indexing or descriptor buffer
```

Indexing is preferred because lavapipe (Mesa 25.0.7) miscompiles
descriptor-buffer image access.

Device creation should fail with `UNSUPPORTED_FEATURE` if required features are missing.

## 4. Runtime and instance creation

Each `VkRuntimeState` owns one instance, its instance dispatch, one optional
debug messenger, and a stable adapter cache. Runtime creation publishes its
public slot only after instance creation and adapter enumeration succeed.

Canonical `create_device(Adapter*, DeviceRequest*)` uses the exact cached physical device and borrows the runtime-owned instance. Device destruction never destroys that borrowed instance; the retained runtime remains unavailable for destruction until the device is gone. The transitional `create_device_from_desc(DeviceDesc*)` path owns a separate instance and performs its own adapter selection.

Instance creation responsibilities:

```text
select Vulkan API version
collect required instance extensions
collect validation layer names when enabled
create vk::Instance
load extension entry points
install a persistent debug-utils messenger for validation or callback routing
```

Runtime creation enables available platform surface extensions and owns the
surface dispatch. A device retains surface procedures only for a presentation
request and loads only the selected device dispatch groups after logical-device
creation. Headless devices create no presentation state. Missing platform
support faults when that platform constructor is called. The direct-device path
is headless.

`VK_EXT_debug_utils` is requested when validation, Vulkan debug names, or a
structured callback needs it. `enable_debug_names` remains independent of
`enable_validation`; missing debug-utils support makes native naming a
best-effort no-op without discarding slot-owned public debug names.

When validation or a structured callback requests Vulkan routing and the
optional extension is available, the backend installs the persistent
messenger. When validation is enabled, messenger create info is also chained
through instance creation so bootstrap messages use the same route. The
backend stores the callback and userdata before instance creation and
retains them through messenger and instance destruction. Vulkan severity/type
flags, message text,
validation ID name/number, and the first useful named object are translated
into `DebugMessage`; the native callback always returns `VK_FALSE`.

Messages are forwarded synchronously from Vulkan and may be concurrent on
arbitrary driver/application threads. Payloads are borrowed, no diagnostic
allocation or queue is introduced, and callback reentry into gpu.c3l is
prohibited. Native Vulkan object handles/types never cross the public boundary.
When no callback is configured, validation output retains the stderr fallback.
Device teardown runs the public-resource leak scan when validation is enabled
or a structured callback is active. Callback delivery uses
`WARNING`/`resource_lifetime` messages with `destroy_device` operation and
resource identity/name where available; validation without a callback keeps
the stderr report. Device diagnostics remain device-owned. Runtime diagnostics use the same callback contract but have independent instance and messenger lifetimes.

## 5. Adapter enumeration and device selection

Runtime creation enumerates every physical adapter once and caches semantic memory totals, queue counts, general limits, strict support, and separate backend diagnostics. Public enumeration and queries allocate nothing. Cached strings remain valid until runtime destruction.

The direct device path still applies these selection criteria:

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
must satisfy the default one-graphics, one-compute, one-transfer queue request;
roles may alias or use separate families
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

A presentation request names a runtime-owned surface. Device creation requires
at least one requested graphics queue, enables `VK_KHR_swapchain`, and selects a
presentation-capable queue for that surface. It prefers the representative
graphics queue, then compute or transfer representatives, and otherwise uses a
private queue. Split graphics/presentation families use concurrent sharing.
`supports_presentation` reports surface capability.
`supports_device_request` additionally preflights the requested queue counts,
distinct-role constraints, graphics minimum, and presentation topology without
enabling state. Surface formats and present modes remain swapchain-creation
concerns.

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
```

`maintenance4` is always enabled. The strict request adds its heap features:

```text
runtimeDescriptorArray
shaderSampledImageArrayNonUniformIndexing
shaderStorageImageArrayNonUniformIndexing
shaderStorageImageReadWithoutFormat
shaderStorageImageWriteWithoutFormat
```

Optional base features are queried on the selected device and enabled only
when advertised:

```text
fillModeNonSolid -> DeviceCaps.line_polygon_mode
```

`PolygonMode.LINE` faults `UNSUPPORTED_FEATURE` before shader or cache lookup
when this cap is false. The feature is not a physical-device selection requirement.

The selected strict heap path adds either descriptor-buffer support or the
indexing path's partially-bound and update-after-bind features. Unrequested
strict state adds no heap feature chain, extension, dispatch, descriptors, or
pipeline-shared state.

Logical-device queue families form an ordered set. The backend visits the
representative graphics, compute, and transfer queues, then every selected
identity in the same role order, and finally the presentation queue when
present. It appends only the first occurrence of each family. Vulkan
receives one `DeviceQueueCreateInfo` per resulting family. Its `queueCount` is
the highest selected or presentation queue index in that family plus one, with
one priority value per allocated index.

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
    -> validate size, usage, and semantic access
    -> translate BufferUsage to vk::BufferUsageFlags
    -> derive the admitted native queue families
    -> translate MemoryKind to vma::AllocationCreateInfo
    -> allocator.try_create_buffer
    -> query mapped pointer and optional device address
    -> store BufferSlot, including access roles
    -> return BufferHandle
```

Addressable buffers must include:

```text
VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
```

If address query returns zero, creation should fail.
Queue-family lists are derived from the descriptor's semantic `QueueRoles`
access set. The backend visits admitted roles in graphics, compute, transfer
order and includes every selected identity for each role. Private presentation
queues are excluded. One unique family keeps `EXCLUSIVE` sharing with no family
list; two or more use `CONCURRENT` with the exact ordered, deduplicated list.

Frame arenas admit selected graphics and compute roles, or transfer on a
transfer-only device. Persistent-arena backing admits every selected role;
each returned span retains its narrower allocation access set. Concurrent
sharing does not replace barriers, submission ordering, completion waits, or
retirement. Every arena remains scoped to its owning `Device`.

## 9. Texture implementation

Creation flow:

```text
public TextureDesc
    -> validate shape, format, usage, and semantic access
    -> translate usage to vk::ImageUsageFlags
    -> derive the admitted native queue families
    -> allocator.try_create_image
    -> create default image view and set initial layout
    -> store TextureSlot, including access roles
    -> return TextureHandle
```

Textures use the same one-family `EXCLUSIVE` or multi-family `CONCURRENT`
rule as buffers, based only on `TextureDesc.access`.

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
The internal descriptor buffer uses the frame-arena family list: every selected
graphics and compute identity, or every transfer identity on a transfer-only
device. One family remains `EXCLUSIVE`; two or more use `CONCURRENT` with the
exact ordered list. Transfer is otherwise excluded because transfer commands
never bind or consume the descriptor heap.

This internal policy does not widen public resource access. A buffer or texture
consumed by graphics and compute declares both roles in its `access` set. The
descriptor-indexing path remains unchanged. Descriptor tokens and the
internal heap remain scoped to their owning `Device`.

Public indices map to descriptor entries. Neither path changes the public API
or shader material records.

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

Batch creation uses a prepare/commit transaction under the resource lock.
Preparation validates every item and resolves its view without consuming
descriptor slots, changing generations, draining ready retires, or writing
outputs. It records only cache misses created by the transaction. If a later
item faults, those Vulkan image views are destroyed exactly once and their full
prior cache cells are restored; default, pre-existing, and duplicate views are
untouched. After complete preparation, commit drains the ready retire prefix,
allocates every descriptor, publishes outputs, and performs the existing heap
writes.

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

Dynamic rendering begins with fixed-count `vkCmdSetViewport` and
`vkCmdSetScissor` calls covering the full pass. Public overrides use those
same Vulkan 1.3 core commands after library validation: finite viewport
values, nonnegative/positive extents as appropriate, representable scissor
endpoints, depth endpoints in `[0, 1]`, and rectangles bounded by the active
pass. The command list carries the active pass extent only while
`RECORDING_RENDER_PASS`.

Viewport/scissor are deliberately absent from `PipelineKey`, `PipelineSlot`,
and graphics dynamic-state replay. Pipeline binds replay raster/depth
snapshots only, so an explicit rectangle remains active across a bind or
cache-alias handle switch. Multi-viewport arrays, negative-height viewport
flips, and off-pass overscan are not part of the portable public contract.

### Pipeline cache

Two layers. A descriptor-keyed dedup cache (`PipelineKey` over immutable state,
with refcounted aliases) sits in front of a driver `vk::PipelineCache`. The
driver cache is created with `DeviceDesc.pipeline_cache_data` as initial data
and exported through `get_pipeline_cache_size` / `get_pipeline_cache_data`.

Compute pipeline layouts are shared per push-constant size in a packed
device-owned cache. Host storage uses pipeline capacity as an initial hint and
grows to the device's finite valid-size count.

### Result mapping

The context-free Vulkan result mapper handles success, host/device allocation
failures, explicit device loss, and missing features, extensions, or layers.
Operations with additional result semantics use dedicated mappers: backend
bootstrap, surface and swapchain work, texture and shader creation, pipeline
creation, descriptor allocation, virtual-arena allocation, and enumeration.
Unclassified native failures are logged and surface as `BACKEND_ERROR`; they
must never be inferred as device loss.

Queue submission and `wait_queue_idle` use the state-aware result path.
With a callback, failures preserve the mapped public fault and emit one backend
diagnostic with the exact operation (`submit` or `wait_queue_idle`) and native
result text. With a null callback, mapped results return their public faults
silently; only unmapped results retain the stderr fallback. Surface creation and
query, swapchain creation and enumeration, acquire, present, resize/destroy idle
waits, and present-mode queries use the same operation-aware rule while
preserving their specialized WSI fault mappings and the raw signed VkResult. A
swapchain identity is attached only after its handle resolves. Expected WSI
recovery outcomes are silent even with a callback: acquire `TIMEOUT`/`NOT_READY`
returns `WAIT_TIMEOUT`, and dormant acquire returns `SWAPCHAIN_OUT_OF_DATE`,
without diagnostic delivery.

A rejected warm pipeline-cache blob may be retried with an empty cache. Host or
device allocation failure and explicit device loss propagate without retry.

## 12. Command buffers

Command pools follow the representative queue selected for each public
`QueueKind`, not every allocated identity. Each recording context owns:

```text
one graphics pool per frame when graphics is requested
one compute pool per frame when graphics is absent or uses another family
one transfer pool per frame when transfer is requested
reset pools when frame retires
```

Recording-context pool sets are constructed transactionally: ownership is
published only after every representative pool exists. Any create fault
destroys all earlier pools and releases the host arrays. Same-family compute
aliases the graphics pool only after success. Additional selected identities do
not own command pools because command recording and submission take
`QueueKind` and route through its representative native queue.

Public command token:

```text
CommandList
    Device device
    CommandListHandle handle
```

The handle resolves through a fixed 4096-entry device table to a backend
`CommandRecord` containing the `vk::CommandBuffer`, recording context, queue,
frame-slot index, lifecycle state, last-bound pipeline cache, and a growable
array of pending texture-layout transitions. The public token stays within two
machine words; copying it creates an alias, not an independent recorder.

Begin/end validate state transitions. Submit preflights a whole batch under the
command-table mutex, rejects duplicate tokens, mixed queue kinds, and stale
owners, then claims every record as `SUBMITTING` before constructing the Vulkan
submission. A fault before `vkQueueSubmit2` succeeds restores claimed records to
`EXECUTABLE`. Success commits layout transitions in submission order and frees
the records, invalidating all aliases.
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

`Stage.NONE` translates exactly to `VK_PIPELINE_STAGE_2_NONE`. The
presentation preset uses the color-attachment-output stage with an empty
access scope on its WSI-facing side, while explicit `Stage.PRESENT` and
`Hazard.PRESENT_READ` retain their
compatibility mappings to `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` and
`VK_ACCESS_2_MEMORY_READ_BIT`.

Presentation transitions use these exact synchronization2 scopes:

| Transition | Source scope | Destination scope |
|---|---|---|
| `PRESENT -> COLOR_ATTACHMENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_NONE` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT \| VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` |
| `COLOR_ATTACHMENT -> PRESENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT \| VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_NONE` |

The presentation-facing access scope is empty because the presentation engine
is external to the Vulkan pipeline. Its stage remains color-attachment output
so the acquire transition chains after the image-available semaphore wait and
the release transition chains before the render-finished semaphore signal;
both semaphore operations use that stage. `Stage.NONE` is legal for semaphore
operations but carries no pipeline work; use it only deliberately.

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

User signal values must exceed the semaphore counter when they execute. The
caller orders pending signals across queues and observes
`DeviceCaps.max_timeline_semaphore_value_difference`.

## 15. Render pass implementation

Use dynamic rendering.

Render pass begin:

```text
reject color counts above the library or selected-device limit
validate every color handle, usage, mip/layer range, selected-mip extent, and layout
validate the depth handle, usage, mip-zero extent, and layout
transition only if caller explicitly requested via barrier before begin
resolve views and build attachment infos only after all targets validate
vkCmdBeginRendering
```

Render pass end:

```text
vkCmdEndRendering
```

Do not transition attachments to shader-read automatically.

## 16. Swapchain implementation

The runtime-owned `gpu::Surface` backend object owns its `vk::SurfaceKHR` and
retains the Vulkan instance through the runtime. A swapchain borrows that handle
and retains the public surface token until swapchain destruction.

Swapchain module owns:

```text
vk::SwapchainKHR
vk::Image[]
vk::ImageView[]
SwapchainInfo info
uint image_count
```

Consumers create surfaces through the typed
`gpu::surface::{win32,wayland,x11}::create_surface` modules. SDL3 helpers may
translate window properties into those native handle types; they do not call
`gpu::vk` or Vulkan WSI functions.

Swapchain operations:

```text
create
acquire
present
query runtime info
query present-mode support
resize on out-of-date/suboptimal; recreate the surface on surface-lost
```

`vk_get_present_mode_support` queries the borrowed, runtime-owned
`vk::SurfaceKHR` for fifo/immediate/mailbox availability. At creation,
`select_present_mode` falls back to FIFO when the requested mode is unavailable.

WSI result mapping is explicit and pure-tested:

| Vulkan result | Public outcome | State/recovery |
|---|---|---|
| acquire `TIMEOUT` / `NOT_READY` | `WAIT_TIMEOUT` | pending-acquire state stays unchanged; retry |
| `ERROR_OUT_OF_DATE_KHR` | `SWAPCHAIN_OUT_OF_DATE` | resize the swapchain |
| `ERROR_SURFACE_LOST_KHR` | `SURFACE_LOST` | replace the platform surface and swapchain |
| acquire `SUBOPTIMAL_KHR` | valid image with `suboptimal = true` | finish the frame; resize when convenient |
| present `SUBOPTIMAL_KHR` | success | current public API exposes no soft present result |

Swapchain construction keeps selected format, clamped extent, actual image
count returned by the driver, and selected present mode local until every
image is wrapped. A completed build publishes one `SwapchainInfo` snapshot;
a zero extent or failed rebuild publishes the dormant sentinel
(`UNDEFINED`, zero extent/count, FIFO, dormant). Tier-E external
synchronization prevents observation during publication.

Acquire reads `AcquiredImage.prior_layout` from the wrapped texture's
committed tracked layout. Resize creates new wrapped slots at `UNDEFINED`;
submitted barriers commit later states through the normal texture-layout
path, so swapchain code owns no parallel seen-state table.

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

All enum/flag conversion should live in `gpu/vk/helpers.c3`.

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
