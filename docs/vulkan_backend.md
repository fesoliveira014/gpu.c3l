# gpu.c3l Vulkan Backend

## 1. Purpose

The Vulkan backend implements the `gpu` public API on Vulkan 1.3. It lives under:

```c3
module gpu::internal::vk @private;
```

Backend declarations are private by default and are nested under the
backend-independent private `gpu::internal` module. Public non-callables remain
in `gpu/gpu.c3i`; `gpu/gpu.c3` owns public dispatch callables. Only those
implementations and white-box tests import backend declarations with a scoped
visibility override.

It imports:

```c3
import gpu;
import vk;
import vma;
import spvreflect;
```

`spvreflect` is used by `gpu/internal/vk/shader.c3` for SPIR-V reflection.

No `vk::` or `vma::` type should appear in public `gpu` API signatures.

## 2. Backend files

```text
gpu/internal/vk/backend.c3              instance/device dispatch loading, loader/VMA link probes
gpu/internal/vk/runtime.c3              per-runtime instance, diagnostics, adapter ownership
gpu/internal/vk/surface.c3              per-runtime WSI dispatch and VkSurfaceKHR operations
gpu/internal/vk/adapter.c3              semantic adapter metadata and diagnostic snapshots
gpu/internal/vk/instance.c3             shared instance construction
gpu/internal/vk/device.c3               physical device selection, logical device, feature chain
gpu/internal/vk/queue.c3                queue family selection, queue handles, submit
gpu/internal/vk/allocator.c3            vma::Allocator creation/destruction, stats
gpu/internal/vk/allocation.c3           generic-buffer and raw texture-memory allocations
gpu/internal/vk/buffer.c3               VkBuffer + VMA allocation path
gpu/internal/vk/texture.c3              owned/placed images, views, presentation-use state
gpu/internal/vk/descriptor_heap.c3      descriptor buffer or descriptor indexing implementation
gpu/internal/vk/shader.c3               temporary SPIR-V modules and reflection validation
gpu/internal/vk/pipeline_cache.c3       pipeline dedup cache and driver cache
gpu/internal/vk/pipeline_compute.c3     compute pipeline creation
gpu/internal/vk/pipeline_graphics.c3    graphics pipeline creation
gpu/internal/vk/command.c3              caller-owned allocator pools and command encoding
gpu/internal/vk/command_state.c3        command-list state and handle tracking
gpu/internal/vk/recording_thread_*.c3   portable and Win32 recording-owner identity
gpu/internal/vk/sync.c3                 barriers, timeline semaphores
gpu/internal/vk/render_pass.c3          dynamic rendering
gpu/internal/vk/swapchain.c3            swapchain lifecycle and presentation
gpu/internal/vk/lifetime.c3             optional command resource lifetime tracking
gpu/internal/vk/debug.c3                debug names, leak reports
gpu/internal/vk/helpers.c3              enum and flag translation helpers
gpu/internal/vk/validate.c3             descriptor and command validation helpers
```

## 3. Required Vulkan features

The minimum supported device profile is intentionally Vulkan 1.3 plus
`VK_EXT_extended_dynamic_state3` and
`dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. The backend requires:

```text
Vulkan 1.3
buffer device address
synchronization2
dynamic rendering
timeline semaphores
shaderInt64
multiDrawIndirect
shaderDrawParameters
independentBlend
depthBiasClamp
dynamicPrimitiveTopologyUnrestricted
VK_EXT_extended_dynamic_state3
```

`independentBlend` and `depthBiasClamp` are core physical-device features.
`dynamicPrimitiveTopologyUnrestricted` is reported by
`VkPhysicalDeviceExtendedDynamicState3PropertiesEXT`. The backend requires it
to be `VK_TRUE` but cannot enable it. Requiring it
allows `cmd_set_raster_state` to switch topology classes without compiling
pipeline variants. The backend separately requires and enables the extension
name. Vulkan 1.3 supplies the promoted topology, cull, front-face, and
depth-bias core commands; no extended-dynamic-state feature structure is
enabled.

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
collect validation layer names only for RuntimeDesc.enable_vulkan_validation
create vk::Instance
load extension entry points
install a persistent debug-utils messenger for layer, name, or callback routing
```

Runtime creation enables available platform surface extensions and owns the
surface dispatch. A device retains surface procedures only for a presentation
request and loads only the selected device dispatch groups after logical-device
creation. Headless devices create no presentation state. Missing platform
support faults when that platform constructor is called.

`VK_EXT_debug_utils` is requested when Vulkan validation, Vulkan debug names,
or a structured callback needs it. `enable_debug_names` remains independent of
`enable_vulkan_validation`; missing debug-utils support makes native naming a
best-effort no-op without discarding slot-owned public debug names. Internally,
`VkInstanceDesc.enable_validation` is the backend planning flag derived only
from `RuntimeDesc.enable_vulkan_validation`; it is not a public combined policy.

When layers, debug names, or a structured callback request Vulkan routing and
the optional extension is available, the backend installs the persistent
messenger. When Vulkan validation is enabled, messenger create info is also
chained through instance creation so bootstrap messages use the same route.
Missing Khronos layers can fault runtime creation only when
`enable_vulkan_validation` is true; `ContractValidation.FULL` remains available
without them. The backend stores the callback and userdata before instance
creation and retains them through messenger and instance destruction. Vulkan severity/type
flags, message text,
validation ID name/number, and the first useful named object are translated
into `DebugMessage`; the native callback always returns `VK_FALSE`.

Messages are forwarded synchronously from Vulkan and may be concurrent on
arbitrary driver/application threads. Payloads are borrowed, no diagnostic
allocation or queue is introduced, and callback reentry into gpu.c3l is
prohibited. Native Vulkan object handles/types never cross the public boundary.
When no callback is configured, Vulkan layer output retains the stderr fallback.
Accepted teardown scans backend state for `OBJECT_BOUNDARIES`/`FULL` or when a
structured callback is active. Normal live children are rejected by the public
device registry; the scan covers internal, partial-initialization, and
device-loss leftovers.
Callback messages use `WARNING`/`resource_lifetime` with operation
`destroy_device`; enabled contract reporting without a callback uses stderr.
Runtime diagnostics use the same callback contract with independent instance
and messenger lifetimes. Callback presence changes delivery only; it does not
enable library checks, lifetime tracking, Vulkan layers, or object naming.

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
independentBlend
depthBiasClamp
```

`VK_EXT_extended_dynamic_state3` is also always enabled. Before device creation,
the backend has already required the independently queried
`dynamicPrimitiveTopologyUnrestricted` property; command dispatch remains
Vulkan 1.3 core.

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

The backend owns one indirect-command layout for each draw shape and one
generated-dispatch layout paired with the device's singleton compute pipeline
layout. Each compute cache entry and live pipeline slot borrows that stable
pair. Recording reads the slot value directly, and the device destroys both
owned singleton handles at teardown. Generated recording uses implicit
preprocessing with buffers reserved explicitly by `reserve_generated_scratch`
on the originating command allocator. The allocator fixes the exact queue,
pool, reservation table, and byte budget. Reservations are keyed by public
pipeline handle and generated-work kind, not by native pipeline identity, so
alias handles require separate reservations. For each key, reservation queries
`vkGetGeneratedCommandsMemoryRequirementsEXT` with the exact layout and maximum
sequence count, then allocates the requested number of addressable VMA buffers
using the returned size, alignment, and memory-type mask. That query defines the
maximum sequence count supported by the reservation, so smaller warm calls do
not query requirements again. Warm generated calls claim a matching reservation
slot, retain it through command completion, and return
`GENERATED_SCRATCH_EXHAUSTED` without allocating when the count or available
slot bound is exhausted. A zero-byte preprocess requirement still claims a slot
for exact simultaneous-use accounting. Fixed reservation-table, byte-budget, or
per-list index exhaustion returns `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` before
native mutation. Discard and completion return each buffer to its owning
allocator. `release_generated_scratch` removes one quiescent allocator's
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
These choices stay in `gpu::internal::vk`; public code sees only `MemoryClass` and
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

`gpu::internal::vk::BufferHandle`, `BufferDesc`, and `BufferUsage` implement
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

Texture transitions map caller-declared compositional states and a normalized
subresource range directly to one native image barrier. The backend does not
infer or track general texture layouts. Swapchain slots retain only whether an
image completed a presentation cycle so acquisition can report `prior_state`.

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

Strict heap binding is command-record state rather than pipeline state. The
indexing path emits one set-0 bind for each used strict bind point. The
descriptor-buffer path emits one global buffer bind per command record and one
offset command for each used strict bind point. Switching compatible strict
pipelines does not repeat heap setup. Command-buffer reuse clears the cache, and
private full/per-bind-point invalidation operations are reserved for future
incompatible descriptor domains.

Publishing texture views or samplers does not invalidate this binding state.
Indexing updates retain their update-after-bind rules; descriptor-buffer writes
and reads retain the caller-authored `HazardFlags.descriptors` synchronization
contract.

Every strict-enabled device has an append-only sampler table keyed by normalized
semantic state. Explicitly zero-initialized canonical keys are byte-hashed into
fixed power-of-two buckets with `+1`-encoded links; candidates with an equal hash
are compared by complete canonical equality. Before backend dispatch, the public
frontend validates that enabled anisotropy is finite and within the inclusive
reported range; an over-limit request is rejected rather than clamped. The
accepted value is copied exactly through the canonical key into
`VkSamplerCreateInfo.maxAnisotropy`. Under the resource mutex, interning creates
at most one native sampler for an equal key and publishes its
descriptor, stable index, cell, and bucket link in the same transaction. A
capacity or native-create fault changes neither the table, index, nor descriptor
high-water state. Device teardown destroys each published native sampler and
releases slot and bucket storage.

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
generation with one cell lookup before recycling the slot. Each texture slot
tracks its exact number of live shader-visible views under the resource mutex,
so texture destruction decides ownership in O(1) without scanning descriptor
cells. Cached native views and tracked command references retain independent
lifetime accounting. Device teardown reports leaked texture views. Sampler
indices are device-owned and are not individually releasable.

Batch creation uses a prepare/commit transaction under the resource lock.
Preparation validates every item and resolves its native view without consuming
heap slots, changing generations, or writing outputs. It records only cache
misses created by the transaction. If a later item faults, those Vulkan image
views are destroyed exactly once and their full prior cache cells are restored;
default, pre-existing, and duplicate views are untouched. After complete
preparation, commit allocates every slot, publishes owner-bearing views, and
increments the matching texture counts before performing the selected heap
writes. Full validation rejects count overflow or underflow before mutating a
descriptor cell, owner count, generation, free list, or output token.

## 11. Shader and pipeline implementation

### Shader preparation and modules

`ShaderCode` is borrowed CPU-side IR with library-computed identity, not a
backend object. Before cache lookup, each device interns digest, stage, entry
point, length, and exact SPIR-V into owned storage. Digest selects a bucket;
exact comparison resolves collisions. Debug names do not participate in
identity. On a miss, the backend reflects the owned entry against the requested
pipeline ABI before creating a temporary `vk::ShaderModule`, compiles the
pipeline, and destroys the module before returning. Batch creation reflects
and creates at most one temporary module per `ShaderId`, and rolls back every
created handle, cache entry, and pending shader reference before returning a
fault. No native shader-module handle crosses the public boundary.

Reflection validation checks:

```text
entry point exists and matches the declared stage
descriptor and push-constant enumeration is scoped to the selected entry point
unexpected descriptor sets are rejected and heap bindings match convention
an absent push-constant block is accepted
a present block is unique, starts at offset zero, and exactly matches the
    generated compute or graphics block/member numeric shape
```

The exact check compares block size, member count and order, byte offsets and
sizes, scalar widths, signedness, and integer/float kind. It rejects vectors,
matrices, arrays, nested structs, booleans, and references; reflected names are
ignored. Reflection failures return `SHADER_INVALID` before native shader
creation, pipeline-cache mutation, or output publication. Caller-side pipeline
role errors remain `INVALID_ARGUMENT`.

### Compute pipeline

Compute pipeline creation uses the device-owned singleton layout. Its push
constant range is exactly `RootPush::size`; no per-pipeline layout dimension is
accepted or cached.

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
per-target immutable blend and write-mask state
immutable polygon mode
separate dynamic raster and depth state
viewport/scissor dynamic state
pipeline cache lookup
vk::Pipeline
```

Minimal dynamic-rendering begin emits only `vkCmdBeginRendering` and leaves
command-buffer graphics state unchanged. The convenience begin follows it with
viewport, scissor, five raster commands, and three depth commands as one fixed
ten-command packet. It does not compare against cached state or skip unchanged
fields. `cmd_set_graphics_state` emits the same complete packet before or
during a pass; the individual setters emit their corresponding subset in
either recording phase.

Under `FULL`, state validation checks finite viewport values, selected-device
viewport dimensions and coordinate bounds, independently bounded depth
endpoints, and representable scissor endpoints before native recording.
Negative viewport height, device-bounded negative coordinates, reversed depth,
off-pass overscan, and empty scissors are accepted. Zero viewport height
remains a deliberate non-degenerate library invariant. Trusted entries retain
only the mandatory safe-lowering and state-machine floor. Accepted trusted and
checked values share one exact native lowering path. Checked regular and
generated draws additionally require one complete packet in the current
recording. The initialization bit survives pass boundaries and is cleared when
the command buffer is reset. Trusted draw paths neither read nor update that
bit. The command list retains active render compatibility only while
`RECORDING_RENDER_PASS`; it stores no pass extent for viewport or scissor
validation.

Topology, cull mode, front face, depth bias, viewport, scissor, and depth state
are absent from `PipelineKey` and `PipelineSlot`. `PipelineKey` stores each
color target's format, blend equation, and write mask, plus depth format,
sample count, polygon mode, and shader identity. Explicit pipeline binding
emits the native pipeline and heap binds when the selected bind point's cache
entry changes; graphics and compute selections are independent, and rebinding
the same entry or an alias emits neither. `cmd_set_raster_state`
emits the promoted Vulkan 1.3 topology,
cull, front-face, depth-bias-enable, and depth-bias commands as one validated
operation on a graphics-capable command list before or during a render pass.
`cmd_set_depth_state` emits the
Vulkan 1.3 dynamic depth commands. Draw and dispatch only validate active
state, push roots, and execute; they never create a native pipeline. Dynamic
state survives pipeline switches and pass boundaries; minimal pass begin does
not replay it. There is no backend-generated default or trusted draw-time
initialization branch.
Multi-viewport arrays are outside the portable contract.

### Pipeline cache

Two layers. A descriptor-keyed dedup cache (`PipelineKey` over immutable state,
with refcounted aliases) sits in front of a driver `vk::PipelineCache`. The key
and cache entry contain compact shader IDs, never borrowed `ShaderCode` or
SPIR-V clones. Collision verification and the one owned clone happen only at
the device interning boundary; pipeline lookup compares IDs and immutable state
in average constant time. Cache entries own retained IDs. Last-alias release
unlinks zero-reference identities and returns their slots to a free list, so
capacity follows live cache ownership rather than historical churn. The driver
cache is created with `RuntimeDesc.pipeline_cache_data` as initial data and
exported through `get_pipeline_cache_size` / `get_pipeline_cache_data`.

All compute pipelines borrow one device-owned pipeline layout with the fixed
`RootPush` range. Generated dispatch likewise borrows one device-owned indirect
command layout. Pipeline cache entries and live slots carry non-owning copies;
device teardown destroys the singleton handles after cached pipelines.

### Result mapping

The context-free Vulkan result mapper handles success, host/device allocation
failures, explicit device loss, and missing features, extensions, or layers.
Operations with additional result semantics use dedicated mappers: backend
bootstrap, surface and swapchain work, presentation-fence polling/reset,
texture creation, shader-module creation, pipeline creation, descriptor
allocation, and enumeration.
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

`create_command_allocator(device, queue, desc)` transactionally creates one
private pool for the exact selected queue family, allocates the complete fixed
native command-buffer set in one call, and wires stable per-buffer reference and
generated-index slices plus a recycling stack. The backend publishes the
generational allocator slot only after every host/native allocation succeeds.
Tracking-off devices allocate no reference slab. The table has 256 recyclable
slots; destroyed slots advance generation and re-enter its free list rather than
accumulating historical worker state.

Public command values have two states:

```text
CommandList                recording
ExecutableCommandList      ended, one-shot
```

Each value carries a device token plus `CommandListHandle`. Every use validates
the bounded owner, slot, generation, device, allocator, and authoritative phase
before resolving the record.

Begin claims one stable `CommandRecord` from the device's fixed command table
and one native buffer/scratch unit from the originating allocator's fixed
storage. The record is the sole lifecycle authority. It owns the selected
immutable command-operation table, backend state and backend-command pointers,
retained device ownership, bounded identity, and the current
recording/submission state. The linked `VkCommandRecord` identifies the
originating allocator and fixed buffer/scratch index and holds Vulkan-specific
pipeline snapshots and pending texture transitions. Warm recording resolves the
bounded identity once and dispatches through the record's preselected operation
entry. Trusted backend entries do not repeat capability null checks. Only
lifecycle operations continue through the device vtable and report device loss.
Successful end consumes the recording token and returns the executable token.
`submit` or explicit executable discard consumes the ended token.

`VkRuntimeState`, `VkDeviceConfig`, and `VkDeviceState` carry one named policy:
contract depth, lifetime tracking, and Vulkan-layer selection. Device creation
stores it before policy-dependent subsystems initialize and selects one of four
immutable command tables: trusted/no-tracking, trusted/tracking,
checked/no-tracking, or checked/tracking. `OBJECT_BOUNDARIES` shares trusted
recording entries because its additional checks occur at public boundaries.
The authoritative record stores the selected table during begin; warm recording
performs no policy lookup or branch. Every table retains host
pointer/slice/range safety, overflow protection, command-state and
internal-table integrity, public device ownership, Vulkan result/device-loss
handling, and transactional rollback.

`cmd_bind_pipeline` generation-checks the pipeline slot, resolves its cache
entry, optionally retains tracked ownership, and stores native pipeline/layout, kind,
render compatibility, cache identity, generated-command layout, and public
diagnostic identity. Execution helpers consume only that snapshot; they do not
retain or reread a pipeline cell and never revisit pipeline table/cache storage.
Without lifetime tracking, the caller-owned lifetime contract requires the
pipeline to remain live through command completion. `FULL` controls detailed
semantic preparation independently of that retention choice.

Submission resolves the complete batch before mutation. Bounded resolution
validates the command table and rejects stale, duplicate, non-executable, or
wrong-queue tokens before claiming records. A nonempty attempt allocates one
nonzero device-local visit epoch and stamps each resolved record as it is
visited, so duplicate detection performs one record visit per input until
rejection or completion. When the epoch space is exhausted, the next attempt
clears live stamps across the allocated command cells once before restarting at
epoch one. Empty submissions allocate no epoch. Each record then claims
`EXECUTABLE -> SUBMITTING`; a failure before native acceptance restores every
claim, while success commits pending texture state and publishes `SUBMITTED`.
No fallible token resolution occurs after native acceptance.

Render passes resolve explicit `AttachmentViewHandle` values from a fixed
device-owned table into fixed-size local arrays. Attachment creation owns any
non-default native `VkImageView`; render-pass begin performs no image-view
creation, texture-view-cache lookup, or host allocation.

Discard invalidates the authoritative record before returning its native buffer
and scratch index to the allocator. `submit` retains stable record pointers in
the completion-tracked batch, so one same-queue batch may mix allocators.
Completion observation retires each record exactly once: it first transitions
the record to `INACTIVE`, then releases tracked references and generated
reservations, returns the exact allocator unit, releases retained device
ownership, and finally generation-advances/frees the bounded command cell when
present. Submitted records and allocator units cannot be reused before that
retirement completes.
The next begin resets the preallocated buffer and clears its fixed scratch
counts. If no index is available, begin returns `DEVICE_BUSY`; native/host
growth is never a fallback.

Allocator recording uses its own mutex, and `FULL` rejects a second recording
thread while another recording remains live. Different allocators do not share
a recording lock. The owner clears after the last recording ends or is
discarded, and executable tokens may cross to a synchronized submit thread.
Destroy checks allocator counters and available indices, returns
`RESOURCE_IN_USE` without any queue wait, device wait, completion query, or
poll, and frees the pool, buffers, reservations, slabs, mutex, and table cell
only after quiescence.

Core create/begin/record/end/discard/submit/retire and generated-reservation
paths use explicit allocator-owned or bounded stack storage. They do not access
an ambient per-thread recording cache or require a C3 temporary pool. Creation
and generated reservation remain the intentional cold allocation points.

## 13. Synchronization

Use synchronization2 for barriers.

Translation helpers map semantic global stage/hazard masks and compositional
texture layout/stage/access states to synchronization2 scopes.

Barrier commands:

```text
cmd_barrier -> vk::MemoryBarrier2
cmd_texture_barrier -> vk::ImageMemoryBarrier2
```

`cmd_barrier` emits one global memory barrier. Under `FULL`, normal access scopes follow from
its producer and consumer stages; draw-argument, descriptor-buffer, and
depth/stencil cache paths are enabled only by their hazard flags. Invalid,
contradictory, consumer-incompatible, or queue-unsupported scopes fault before
recording. Trusted entries retain safe lowering and command-state checks but
treat detailed stage/hazard/queue misuse as a caller contract. Cross-queue
ordering remains a submission completion-wait concern.

The backend must not insert hidden barriers for user-visible resource
transitions except for unavoidable swapchain acquire/present transitions
inside WSI helpers.

Full texture-state validation uses one layout-specific matrix:

| Layout | Accepted public stages | Accepted access | Native layout/access |
|---|---|---|---|
| `UNDEFINED` | empty | empty; source only | `UNDEFINED`, none |
| `TRANSFER_SOURCE` | transfer or exclusive `all` | read | transfer source, transfer read |
| `TRANSFER_DESTINATION` | transfer or exclusive `all` | write | transfer destination, transfer write |
| `SAMPLED` | nonempty vertex/fragment/compute combination or exclusive `all` | read | color or depth/stencil read-only, sampled read |
| `STORAGE` | nonempty vertex/fragment/compute combination or exclusive `all` | read, write, or both | general, selected storage access |
| `COLOR_ATTACHMENT` | color output or exclusive `all` | read, write, or both | color attachment, selected color access |
| `DEPTH_ATTACHMENT` | depth output or exclusive `all` | read, write, or both | depth/stencil attachment, selected depth access |
| `PRESENT` | empty | empty | present source; side-aware native stage, no access |

The layout also enforces immutable texture usage, format class, WSI ownership,
and a compatible recording queue. `host` and `present` stage bits are invalid
inside `TextureState`; unknown layout, stage, or access bits fault before
recording. View format must be undefined or exactly match the texture. Zero
mip/layer counts mean the remaining range and are normalized once.

Checked texture barriers perform one handle resolution, one range
normalization, two state validations/lowerings, one native assembly, and one
native emission. Tracking variants retain the explicit texture/allocation once
and roll newly retained references back if later checked preparation faults.
Trusted variants share safe range/lowering and emission without the semantic
matrix; non-tracking variants perform no reference work. No path adds a second
semantic pass or shared layout-history update after successful lowering.

Presentation transitions use these synchronization2 scopes in the shown
write-only color examples:

| Transition | Source scope | Destination scope |
|---|---|---|
| `PRESENT -> COLOR_ATTACHMENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_NONE` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` |
| `PRESENT -> TRANSFER_SOURCE` | `VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT`, `VK_ACCESS_2_NONE` | `VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT`, `VK_ACCESS_2_TRANSFER_READ_BIT` |
| `COLOR_ATTACHMENT -> PRESENT` | `VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT`, `VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT` | `VK_PIPELINE_STAGE_2_NONE`, `VK_ACCESS_2_NONE` |
| `TRANSFER_SOURCE -> PRESENT` | `VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT`, `VK_ACCESS_2_TRANSFER_READ_BIT` | `VK_PIPELINE_STAGE_2_NONE`, `VK_ACCESS_2_NONE` |

The public `PRESENT` state has empty stages and access because the presentation
engine is external to the Vulkan pipeline. A transition to `PRESENT` therefore
uses destination `NONE`/`NONE`; its source remains the caller-declared last
queue use. A transition from `PRESENT` uses no source access and derives its
source-stage anchor from the paired first queue-side state. Non-presentation
access masks remain the validated caller-selected masks. The private acquire
semaphore wait uses the exact `SubmitDesc.readiness_before` mask, which must
cover those same first consumer stages. The private present signal retains the
full-submission scope, and presentation waits it. Split graphics/present
families remain concurrent-shared, and texture barriers keep ignored
queue-family indices.

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
owning private timeline and use the exact validated union of
`CompletionWait.before` and its semantic consumers. `draw_arguments` lowers to
`VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT`. When generated work is enabled, the
same semantic also includes `VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV`
because implicit preprocessing reads generated records before indirect execution.
Waits owned by the target queue are validated and elided.
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
    CompletionConsumerFlags consumers
```

Host poll and wait reject unpublished sequences. Each selected queue owns an
atomic retired prefix whose release-store follows tracked-reference release
when selected, generated-preprocess recycling, and command-buffer retirement for all
published submissions through that sequence. An acquire load satisfying the
requested point is the complete fast path: no Vulkan call, submission mutex, or
resource mutex.

On a cache miss, poll queries the owning timeline once and caps native progress
to the acquire-loaded contiguous published prefix before retiring metadata.
This cap keeps a submission that is natively accepted but not yet publicly
published invisible. Wait calls `vkWaitSemaphores` once and, on success,
retires exactly the requested sequence; timeout advances nothing. Timeline
headroom and submitted-command drains publish through the same retired prefix.
Device-destruction counter queries remain non-retiring readiness checks because
accepted teardown releases all remaining CPU metadata directly.

## 15. Render pass implementation

Use dynamic rendering.

Render pass begin:

```text
reject a host-unsafe pass pointer before backend dispatch
reject color counts above the library or selected-device limit
validate every color source handle, usage, mip/layer range, and selected-mip extent
require one sample count across color and depth sources
validate each resolve as distinct, single-sample, same-format color attachment
validate the depth handle, usage, and selected mip/layer extent
preflight any bound pipeline against color formats, depth format, and samples
transition only if caller explicitly requested via barrier before begin
create views and build attachment infos only after all targets validate
use AVERAGE for normalized/float resolves and SAMPLE_ZERO for integer resolves
track attachment references only after pass preparation succeeds
vkCmdBeginRendering
publish active compatibility and retained references
```

The convenience begin also rejects a host-unsafe state pointer and validates
and lowers the complete packet before tracking or native mutation. After
`vkCmdBeginRendering`, it emits the fixed ten-command packet and publishes
checked graphics-state initialization. Any preparation failure leaves
command-list state, tracked references, and the native command buffer unchanged.

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
The caller's exact nanosecond timeout is passed with it; zero is nonblocking.
`ulong::max` requests an infinite wait and is used only when the surface
platform guarantees presentation forward progress.

Preparation computes the next acquisition and selects a semaphore without
mutating the slot. `TIMEOUT`, `NOT_READY`, and every mapped error preserve the
pending-acquisition fields, ring cursor, render completion, and retirement
points. Native success first validates the image index and wrapped resources,
then commits the acquisition as one logical transaction. Success advances the
ring and publishes a `SwapchainReadiness` packed from device, swapchain slot and
generation, and a non-repeating acquisition sequence.
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
the fixed empty `PRESENT` state in the submitted command list. Successful and
enqueued WSI outcomes retire the acquisition. Host or device allocation failure preserves
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
Shader-view ownership is checked directly on each wrapped texture, so the guard
is O(image count) and independent of global descriptor-heap high-water state;
texture and attachment command-reference checks remain unchanged.

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

When `track_resource_lifetimes` is true, command references cover explicitly
named spans, textures, attachment views, allocations, and pipelines across
recording, executable, and incomplete submitted work. Destruction drains only
already-completed reference records with non-blocking timeline queries; a
remaining reference returns `RESOURCE_IN_USE`. When false, recording allocates
no reference storage and discard, retirement, device loss, and teardown perform
no reference-release work. The backend adds no implicit wait: callers retain
owners until the covering completion is observed. GPU addresses and shader
indices remain caller-managed because they cannot be enumerated from a command
stream.

Placed and dedicated textures retain their allocation until texture destruction.
Releasing an allocation with a live placement returns `RESOURCE_IN_USE`.

## 19. Translation helpers

Centralize enum and flag conversion in `gpu/internal/vk/helpers.c3` or the backend file
that owns the complete semantic operation. Texture-state validation and
lowering stay in one path in `gpu/internal/vk/sync.c3` beside barrier construction
because the layout-specific access mapping belongs to that operation.

Shared helpers include format, usage, sampler, blend, topology, and global
barrier conversion. Texture-state validation, lowering, native barrier
assembly, and emission form one path owned by `gpu/internal/vk/sync.c3`; command and
resource files do not duplicate that translation.

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
