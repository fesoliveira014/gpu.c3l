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
gpu/vk/allocation.c3           generic-buffer and raw texture-memory allocations
gpu/vk/buffer.c3               VkBuffer + VMA allocation path
gpu/vk/texture.c3              owned/placed images, views, presentation-use state
gpu/vk/descriptor_heap.c3      descriptor buffer or descriptor indexing implementation
gpu/vk/shader.c3               temporary SPIR-V modules and reflection validation
gpu/vk/pipeline_cache.c3       pipeline dedup cache and driver cache
gpu/vk/pipeline_compute.c3     compute pipeline creation
gpu/vk/pipeline_graphics.c3    graphics pipeline creation
gpu/vk/command.c3              private recording pools and command encoding
gpu/vk/command_state.c3        command-list state and handle tracking
gpu/vk/sync.c3                 barriers, timeline semaphores
gpu/vk/render_pass.c3          dynamic rendering
gpu/vk/swapchain.c3            swapchain lifecycle and presentation
gpu/vk/lifetime.c3             validation-only command resource tracking
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

Shader heaps are selected automatically. Descriptor indexing is preferred when
its features and limits satisfy the requested semantic capacities. Descriptor
buffers provide the fallback only after an exact temporary layout preflight
proves the same capacities fit the candidate's driver-reported limits. Indexing
is preferred in part because lavapipe (Mesa 25.0.7) miscompiles
descriptor-buffer image access.

Device creation should fail with `UNSUPPORTED_FEATURE` if required features are missing.

## 4. Runtime and instance creation

Each `VkRuntimeState` owns one instance, its instance dispatch, one optional
debug messenger, and a stable adapter cache. Runtime creation publishes its
public slot only after instance creation and adapter enumeration succeed.

`create_device(Adapter*, DeviceRequest*)` uses the exact cached physical device and borrows the runtime-owned instance. Device destruction never destroys that instance; the retained runtime remains unavailable for destruction until the device is gone. Runtime device defaults are copied before instance creation and inherited by each created device.

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
support faults when that platform constructor is called.

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
Accepted teardown scans backend state when validation or a structured callback
is active. Normal live children are rejected by the public device registry;
the scan covers internal, partial-initialization, and device-loss leftovers.
Callback messages use `WARNING`/`resource_lifetime` with operation
`destroy_device`; validation without a callback uses stderr. Runtime diagnostics
use the same callback contract with independent instance and messenger lifetimes.

## 5. Adapter enumeration and device selection

Runtime creation enumerates every physical adapter once and caches semantic memory totals, queue counts, general limits, strict support, and separate backend diagnostics. Public enumeration and queries allocate nothing. Cached strings remain valid until runtime destruction.

The cached `AdapterInfo.strict_supported` flag describes support for the
runtime's configured strict profile, including its exact semantic heap
capacities. Applications select an exact adapter from that cache;
`supports_device_request` then evaluates the requested queue and presentation
semantics without enabling state. Device creation uses that same runtime
configuration for the selected adapter.

The strict profile requires:

```text
must support Vulkan 1.3
must support buffer device address
must support synchronization2
must support dynamic rendering
must support timeline semaphores
must support shaderInt64
must support multiDrawIndirect
must support shaderDrawParameters
must support one exact shader-heap implementation
must satisfy the default one-graphics, one-compute, one-transfer queue profile;
roles may alias or use separate families
```

A presentation request names a runtime-owned surface. Device creation requires
at least one requested graphics queue, instance extensions
`VK_KHR_get_surface_capabilities2` and `VK_EXT_surface_maintenance1`, device
extensions `VK_KHR_swapchain` and
`VK_EXT_swapchain_maintenance1`, and the maintenance feature enabled, then
selects a presentation-capable queue for that surface. The maintenance
extension supplies
non-blocking proof that presentation no longer uses private WSI objects. The
backend prefers the representative graphics queue, then compute or transfer
representatives, and otherwise uses a private queue. Split
graphics/presentation families use concurrent sharing. `supports_presentation`
reports the complete surface and strict-presentation capability.
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

Strict devices probe `VK_EXT_device_generated_commands` and
`VK_KHR_maintenance5` as the optional implementation of generated work. The
backend enables them only when both features are advertised and the generated
command properties support vertex, fragment, and compute stages, two-token
layouts, token offset 16, record stride 40, and a nonzero work-count limit.
`DeviceCaps.generated_work` and `max_generated_work_count` report the result
semantically; unsupported devices report false and zero. These extensions are
not physical-device selection requirements.

The backend owns one indirect-command layout for each draw shape and one per
cached compute pipeline layout. Each compute pipeline cache entry and live
pipeline slot carries the generated-dispatch layout paired with its ordinary
pipeline layout. Recording reads that stable slot value directly; it never
searches layout-cache storage that pipeline creation may grow. The device-owned
compute-layout cache remains the sole owner and destroys both handles at device
teardown. Generated recording uses implicit preprocessing with buffers reserved
explicitly by `reserve_generated_scratch` on the calling thread's device
recording context. The queue argument selects and validates the device. For
each pipeline and generated-work kind, reservation queries
`vkGetGeneratedCommandsMemoryRequirementsEXT` with the exact layout and maximum
sequence count, then allocates the requested number of addressable VMA buffers
using the returned size, alignment, and memory-type mask. Warm generated calls
borrow a matching buffer, retain it through command completion, and return
`GENERATED_SCRATCH_EXHAUSTED` without allocating when the count or available
compatible-buffer bound is exhausted. Discard and completion return each
buffer to its owning context. `release_generated_scratch` removes one quiescent
pipeline/kind reservation.
A barrier with
`hazards.draw_arguments` includes both indirect-command and generated
command-preprocess reads when this capability is enabled.

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
allocator.create_buffer_with_alignment
allocator.try_create_image
allocator.try_map
allocator.try_flush
allocator.try_invalidate
allocator.heap_budgets
allocator.stats_string
```

### Independent allocations

`AllocationDesc` is translated without exposing native policy publicly:

```text
validate size, class, alignment, and semantic access
generic classes:
    build a private addressable buffer with the generic usage superset
    select CPU_WRITE, GPU_PRIVATE, or CPU_READ policy
    require mapping for CPU_WRITE and CPU_READ
    require a nonzero device address
texture class:
    intersect queried compatibility values
    allocate raw device-local image memory without a buffer
copy debug names
publish the AllocationTable slot last
```

`CPU_WRITE` uses mapped sequential host access, `GPU_PRIVATE` prefers device
memory without a public mapping, and `CPU_READ` uses mapped random host access.
These choices stay in `gpu::vk`; public code sees only `MemoryClass` and
`AllocationInfo`.

Each slot stores immutable size, class, access, alignment, capabilities, native
ownership, and generation. Texture-memory slots reject span resolution.
Creation rollback releases native ownership without publishing a token.

`free_allocation` invalidates the generation, then destroys the generic buffer or
frees raw texture memory. Live placements return `RESOURCE_IN_USE`. Normal
device destruction is blocked
by live public allocations; accepted loss/partial teardown releases remaining
table entries before destroying the allocator.

## 8. Private buffer implementation

`gpu::vk::BufferHandle`, `BufferDesc`, and `BufferUsage` implement
generic allocation backing. They never cross backend dispatch.

Creation validates size and semantic access, derives the exact native queue
families, translates private usage and memory policy, creates the VMA-backed
buffer, and publishes the private handle only after mapping state and any
required nonzero device address are known. Addressable backing includes
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.

One unique family uses `EXCLUSIVE` sharing; multiple admitted families use
`CONCURRENT` with the exact ordered, deduplicated list. Presentation queues
are excluded. Independent allocations derive sharing only from their immutable
access roles. Sharing does not replace barriers, submission ordering,
completion waits, or lifetime rules.

## 9. Texture implementation

Creation flow:

```text
public TextureDesc
    -> validate shape, format, usage, and semantic access
    -> translate usage to vk::ImageUsageFlags
    -> derive the admitted native queue families
    -> allocator.try_create_image
    -> create default image view
    -> store TextureSlot, including access roles
    -> return TextureHandle
```

Placed creation uses queried image requirements, raw alias-capable VMA memory,
and `create_aliasing_image2`. Dedicated creation is one transaction:

```text
create image
allocate dedicated VMA memory for that image
bind memory
create default view
publish TextureHandle and GpuAllocation under one lock
```

Any fault before publication destroys the temporary view, image, and allocation.

Textures use the same one-family `EXCLUSIVE` or multi-family `CONCURRENT`
rule as buffers, based only on `TextureDesc.access`.

Texture transitions map the caller-declared semantic uses and subresource
range directly to one native image barrier. The backend does not infer or track
general texture layouts. Swapchain slots retain only whether an image completed
a presentation cycle so acquisition can report `prior_use`.

## 10. Descriptor heap implementation

### Descriptor indexing path

Preferred automatically when its features and limits satisfy the requested capacities.

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
`UNSUPPORTED_FEATURE`; capacities are never clamped.

### Descriptor buffer path

Selected automatically when descriptor indexing cannot satisfy the requested
capacities and descriptor-buffer support is available. Candidate selection
creates the exact layout temporarily and checks its driver-reported size
against every descriptor-buffer range and address-space limit before ranking
the adapter. Mesa 25.0.7 lavapipe miscompiles descriptor-buffer image access,
so that path remains gated in native shader tests.

Backend owns descriptor buffers for:

```text
sampled images
storage images
samplers
```
The internal descriptor buffer admits every selected
graphics and compute identity, or every transfer identity on a transfer-only
device. One family remains `EXCLUSIVE`; two or more use `CONCURRENT` with the
exact ordered list. Transfer is otherwise excluded because transfer commands
never bind or consume the descriptor heap.

This internal policy does not widen public resource access. Explicit command
resources are checked against `CommandRecord.queue` before Vulkan commands,
command recording, pipeline binding, or transfer allocation. Span metadata must
remain a non-empty subset of its backing buffer. Descriptor tokens and the
internal heap remain scoped to their owning `Device`.

Public indices map to descriptor entries. Neither path changes the public API
or shader material records.

Sampler identity is not owned by either descriptor path. Every device has an
append-only sampler table keyed by normalized semantic state. Interning occurs
under the resource mutex, creates at most one native sampler for an equal key,
and retains it until device teardown. A strict-enabled device may publish that
identity once into the selected sampler heap; repeated publication returns the
same index, while a capacity fault leaves the identity unpublished and valid.

### Descriptor slot policy

```text
DescriptorSlot
    uint index
    uint generation
    DescriptorKind kind
    bool used
    TextureHandle owner_texture
```

CPU cells retain generation and texture ownership. The public `TextureView`
token also carries the device-owner identity and current generation, while its
`TextureIndex` field is only the shader value: zero is invalid and a live value
encodes the zero-based physical slot plus one. Destruction validates owner and
generation before recycling the slot. Device teardown reports leaked texture
views. Sampler entries are device-owned and are not individually releasable.

Batch creation uses a prepare/commit transaction under the resource lock.
Preparation validates every item and resolves its native view without consuming
heap slots, changing generations, or writing outputs. It records only cache
misses created by the transaction. If a later item faults, those Vulkan image
views are destroyed exactly once and their full prior cache cells are restored;
default, pre-existing, and duplicate views are untouched. After complete
preparation, commit allocates every slot, publishes owner-bearing views, and
performs the selected heap writes.

## 11. Shader and pipeline implementation

### Shader preparation and modules

`ShaderCode` is borrowed CPU-side IR with library-computed identity, not a
backend object. Pipeline creation checks the in-memory cache before compiling.
On a miss, the backend creates a temporary `vk::ShaderModule`, validates
reflection against the requested pipeline ABI, compiles the pipeline, and
destroys the module before returning. Batch creation reuses one temporary
module for each exact shader value and rolls back every created handle and cache
entry before returning a fault. No shader-module handle or table crosses the
public boundary.

Reflection validation checks:

```text
entry point exists and matches the declared stage
push constant blocks fit the stage's root-push range
unexpected descriptor sets are rejected
descriptor heap bindings match convention
```

### Compute pipeline

Compute pipeline creation:

Before shader-code lookup, reject a push-constant size below
`RootPush::size`, not divisible by four, or above
`DeviceCaps.max_push_constant_size`. These public
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
immutable raster and blend state
separate dynamic depth state
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

Viewport, scissor, and depth state are absent from `PipelineKey` and
`PipelineSlot`. Explicit pipeline binding emits the native pipeline and heap
binds when the active cache entry changes; rebinding the same entry or an alias
emits neither. `cmd_set_depth_state` emits the Vulkan 1.3 dynamic depth commands and
marks depth state valid for the active pass. Draw and dispatch only validate
active state, push roots, and execute; they never create a native pipeline.
Viewport and scissor survive pipeline switches, while pass begin resets them to
the full pass and requires depth state again. Multi-viewport arrays,
negative-height viewport flips, and off-pass overscan are outside the portable
contract.

### Pipeline cache

Two layers. A descriptor-keyed dedup cache (`PipelineKey` over immutable state,
with refcounted aliases) sits in front of a driver `vk::PipelineCache`. Shader
digests participate in the key, while cache matches also compare stage, entry
point, length, and exact SPIR-V bytes. Cache entries clone those borrowed
identity inputs with the device host allocator, so digest collisions stay
distinct and a prepared value can be reused across devices. The driver cache is
created with `RuntimeDesc.pipeline_cache_data` as initial data and exported
through `get_pipeline_cache_size` / `get_pipeline_cache_data`.

Compute pipeline layouts are shared per push-constant size in a packed
device-owned cache. Host storage uses pipeline capacity as an initial hint and
grows to the device's finite valid-size count. Lookup returns the ordinary and
generated-dispatch layouts as one value, and publication copies that pair into
the canonical pipeline cache entry and every resulting pipeline slot. Those
copies are non-owning; only the compute-layout cache destroys the native
layouts.

### Result mapping

The context-free Vulkan result mapper handles success, host/device allocation
failures, explicit device loss, and missing features, extensions, or layers.
Operations with additional result semantics use dedicated mappers: backend
bootstrap, surface and swapchain work, presentation-fence polling/reset, texture
creation, shader-module creation,
pipeline creation, descriptor allocation, and enumeration.
Unclassified native failures are logged and surface as `BACKEND_ERROR`; they
must never be inferred as device loss.

Queue submission and completion waits use the state-aware result path.
With a callback, failures preserve the mapped public fault and emit one backend
diagnostic with the exact operation (`submit` or `wait_completion`) and native
result text. With a null callback, mapped results return their public faults
silently; only unmapped results retain the stderr fallback. Surface creation and
query, swapchain creation and enumeration, acquire, present, non-blocking
presentation-fence queries, and present-mode queries use the same
operation-aware rule while preserving their specialized WSI fault mappings and
the raw signed VkResult. A swapchain identity is attached only after its handle
resolves. Expected WSI recovery outcomes are silent even with a callback:
acquire `TIMEOUT`/`NOT_READY` and a busy present fence return `WAIT_TIMEOUT`, and
dormant acquire returns `SWAPCHAIN_OUT_OF_DATE`, without diagnostic delivery.

A rejected warm pipeline-cache blob may be retried with an empty cache. Host or
device allocation failure and explicit device loss propagate without retry.

## 12. Command buffers

`begin_commands(queue)` lazily creates a private recording-pool set for the
calling thread and device. Each set contains one pool per selected queue family.
Compute aliases graphics when both use one family. A selected transfer role has
a separate pool even when its family is shared. Pool construction is
transactional and uses the device host allocator.

Public command values have two states:

```text
CommandList                recording
ExecutableCommandList      ended, one-shot
```

Both carry a device token, `CommandListHandle`, and opaque encoder pointer.
Begin resolves the handle once and publishes a stable root encoder containing
the selected command-operation table, `VkDeviceState*`, and fixed
`CommandRecord*`. Warm recording uses those cached pointers; only lifecycle
operations continue through the device vtable. The record owns the native
command buffer and pool, exact public queue, lifecycle state, bound-pipeline
snapshot, and pending texture transitions.
Successful end consumes the recording token and returns the executable token.
`submit` or explicit executable discard consumes the ended token.

`cmd_bind_pipeline` resolves the pipeline slot and cache entry once and stores
the expected slot generation plus native pipeline/layout, kind, render
compatibility, cache identity, and generated-command layout. Execution helpers
validate the stable cell directly and never revisit pipeline table/cache storage.
The checked command-operation table is immutable per device. Policy selection
happens during device or encoder setup and remains outside warm recording.

Submission preflights the batch under the command-table mutex. It rejects stale,
duplicate, non-executable, or wrong-queue tokens before claiming records. A
failure before native acceptance restores every claim; success commits pending
texture state and invalidates each encoder before its command-table index becomes
reusable. No fallible token resolution occurs after native acceptance.

Render passes resolve explicit `AttachmentViewHandle` values from a fixed
device-owned table into fixed-size local arrays. Attachment creation owns any
non-default native `VkImageView`; render-pass begin performs no image-view
creation, texture-view-cache lookup, or host allocation.

Discard retires the native buffer to its recording context. `submit` transfers
it to a completion-tracked batch; completion observation retires the buffer to
the same context. Before its next begin, only the context owner resets a
compatible retired buffer and reuses its reference and generated-scratch arrays.
Allocation of a native command buffer or growth of those arrays occurs only on
the cold path when no reusable capacity exists. Device teardown relies on
command-pool destruction.

## 13. Synchronization

Use synchronization2 for barriers.

Translation helpers map semantic global stage/hazard masks and explicit texture
transition tuples to synchronization2 scopes.

Barrier commands:

```text
cmd_barrier -> vk::MemoryBarrier2
cmd_texture_barrier -> vk::ImageMemoryBarrier2
```

`cmd_barrier` emits one global memory barrier. Normal access scopes follow from
its producer and consumer stages; draw-argument, descriptor-buffer, and
depth/stencil cache paths are enabled only by their hazard flags. Invalid,
contradictory, consumer-incompatible, or queue-unsupported scopes fault before
recording. Cross-queue ordering remains a submission completion-wait concern.

The backend must not insert hidden barriers for user-visible resource
transitions except for unavoidable swapchain acquire/present transitions
inside WSI helpers.

For texture transitions, each semantic use must map to a stage supported by
the recording queue. `TextureUse.UNDEFINED` maps to empty execution and access
scopes. `TextureUse.PRESENT` uses the color-attachment-output stage with an
empty access scope on its WSI-facing side.

Presentation transitions use these exact synchronization2 scopes:

| Transition | Source scope | Destination scope |
|---|---|---|
| `PRESENT -> COLOR_ATTACHMENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_NONE` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT \| VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` |
| `COLOR_ATTACHMENT -> PRESENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT \| VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_NONE` |

The presentation-facing access scope is empty because the presentation engine
is external to the Vulkan pipeline. The texture transitions keep their narrow
color-attachment scopes. The private acquire semaphore uses the exact
`SubmitDesc.readiness_before` destination mask. The private present signal keeps
the backend's full-submission scope.

## 14. Timeline semaphores

Use timeline semaphores for:

```text
queue-owned completion points
queue submission order
completion-based caller lifetime decisions
```

Each selected queue identity owns one private timeline, monotonic sequence, and
submission mutex. Roles that resolve to the same native queue share that state.
The public `CompletionPoint` packs the device, queue identity, and sequence in
two words. Reservation and publication allocate nothing.

Every public `submit(queue, desc)` signals one sequence on the selected queue
and returns its point. Empty batches are valid. Completion waits resolve to the
owning private timeline and use the exact validated `CompletionWait.before`
stage mask; waits owned by the target queue are validated and elided.
The queue mutex covers sequence reservation and `vkQueueSubmit2`, satisfying
Vulkan external synchronization. Near the timeline-value-difference limit, the
backend queries completed progress and returns `DEVICE_BUSY` before reservation
if headroom remains unavailable. Native failure cancels the reservation before
unlocking. Command records, counters, swapchain state, and point publication
commit only after native success.

```text
SubmitDesc
    ExecutableCommandList[] command_lists
    CompletionWait[] completion_waits
    SwapchainReadiness readiness
    StageMask readiness_before

CompletionWait
    CompletionPoint point
    StageMask before
```

Host poll and wait reject unpublished sequences and query the owning timeline
directly. Successful observation advances cached progress when it covers the
latest published sequence. Device destruction performs the same non-blocking
query and returns `DEVICE_BUSY` while work is incomplete.

## 15. Render pass implementation

Use dynamic rendering.

Render pass begin:

```text
reject color counts above the library or selected-device limit
validate every color source handle, usage, mip/layer range, and selected-mip extent
require one sample count across color and depth sources
validate each resolve as distinct, single-sample, same-format color attachment
validate the depth handle, usage, and mip-zero extent
preflight any bound pipeline against color formats, depth format, and samples
transition only if caller explicitly requested via barrier before begin
create views and build attachment infos only after all targets validate
use AVERAGE for normalized/float resolves and SAMPLE_ZERO for integer resolves
vkCmdBeginRendering
publish active compatibility and retained references
```

Render pass end:

```text
vkCmdEndRendering
```

Do not transition attachments to shader-read automatically.

## 16. Swapchain implementation

A swapchain borrows its runtime-owned `vk::SurfaceKHR` and retains the public
surface until destruction. Each live slot owns its `vk::SwapchainKHR`, wrapped
images, a fixed two-slot acquire-semaphore ring with per-slot retirement points,
per-image present semaphores and presentation-retirement fences, runtime
snapshot, and one pending acquisition state.

Acquire first checks identity headroom, then scans the private ring from its next
slot and polls each slot's retirement `CompletionPoint`. If neither slot is
retired, it returns `WAIT_TIMEOUT` without calling `vkAcquireNextImageKHR`.
Otherwise it passes the selected semaphore to the native acquire.
Success advances the ring and publishes a `SwapchainReadiness` packed from
device, swapchain slot and generation, and a non-repeating acquisition sequence.
Successful submission associates the selected slot with the returned render
completion. The native semaphore never enters the public value.

A readiness-consuming graphics submit:

1. validates the exact pending acquisition and the caller's nonempty supported
   destination stage mask;
2. waits its private acquire semaphore at those exact destination stages;
3. signals the image's private present semaphore and the queue timeline;
4. commits readiness consumption and the returned completion point only after
   `vkQueueSubmit2` succeeds.

Present requires that exact completion point and image identity, then attaches a
private `VkSwapchainPresentFenceInfoEXT` fence and waits the private present
semaphore in `vkQueuePresentKHR`. The caller records the explicit transition to
`TextureUse.PRESENT` in the submitted command list. Successful and enqueued WSI
outcomes retire the acquisition. Host or device allocation failure preserves
it for retry. Reuse polls the image's previous fence; `NOT_READY` becomes
`WAIT_TIMEOUT` without calling native present.

| Vulkan result | Public outcome | Recovery |
|---|---|---|
| acquire `TIMEOUT` / `NOT_READY` | `WAIT_TIMEOUT` | retry acquire |
| `ERROR_OUT_OF_DATE_KHR` | `SWAPCHAIN_OUT_OF_DATE` | resize |
| `ERROR_SURFACE_LOST_KHR` | `SURFACE_LOST` | replace surface and swapchain |
| acquire `SUBOPTIMAL_KHR` | valid image with `suboptimal = true` | present, then resize |
| present `SUBOPTIMAL_KHR` | success | resize when convenient |

Creation and resize publish `SwapchainInfo` only after every image is wrapped.
Zero extent or failed rebuild publishes the dormant sentinel. Resize preserves
the acquisition sequence, so stale readiness cannot alias later work.

Destroy and resize poll all pending presentation fences and drain only already
completed command-reference records. They return `INVALID_RESOURCE_STATE` for
a pending acquisition and `RESOURCE_IN_USE` for unfinished presentation or live
command/view references. They never call `vkQueueWaitIdle`, wait a fence, submit
cleanup work, or defer release. Once the guards pass, wrapped texture slots,
views, semaphores, fences, and the native swapchain are released immediately.

## 17. Debug implementation

Debug features:

```text
Vulkan object names
VMA allocation names
live allocation identity/name reports
live slot reports
leaked descriptor reports
allocation stats
validation message routing
optional command labels
```

Object naming should happen immediately after successful backend object creation.

## 18. Immediate resource lifetime

Core destruction releases native objects immediately. The backend performs no
wait and owns no deferred-release queue. Swapchain destruction and resize use
private presentation fences to prove WSI retirement without hidden waits.

Validation-only command references cover explicitly named spans, textures, and
pipelines across recording, executable, and incomplete submitted work.
Destruction drains only already-completed reference records with non-blocking
timeline queries; a remaining reference returns `RESOURCE_IN_USE`. GPU
addresses and shader indices remain caller-managed because they cannot be
enumerated from a command stream.

Placed and dedicated textures retain their allocation until texture destruction.
Releasing an allocation with a live placement returns `RESOURCE_IN_USE`.

## 19. Translation helpers

Centralize enum and flag conversion in `gpu/vk/helpers.c3` or the backend file
that owns the complete semantic operation. Texture-use conversion stays in
`gpu/vk/sync.c3` beside barrier construction because layouts are private to
that operation.

Shared helpers include format, usage, sampler, blend, topology, and global
barrier conversion. Do not duplicate translation switches in command or
resource files.

## 20. Backend acceptance criteria

The Vulkan backend is acceptable when:

```text
device creation is validation-clean
all native buffer/image allocations are VMA-backed
independent allocation creation publishes only complete native state
addressable spans produce valid GPU addresses
root-pointer compute works
texture heap decodes generation-free TextureIndex values
barriers use synchronization2
offscreen dynamic rendering works
SDL3 swapchain sample presents and resizes
live resource leaks are reported
no vk:: or vma:: type appears in public API signatures
```
