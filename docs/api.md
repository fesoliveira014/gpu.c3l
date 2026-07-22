# gpu.c3l Public API

This is a curated guide to the public API and its idioms. The generated
`api-reference` CI artifact covers every public symbol.
Source doc comments define the contract.

## 1. Public module

All public API lives in:

```c3
module gpu;
```

Backend and dependency modules are not re-exported as public handle types.

Application code should import:

```c3
import gpu;
```

Windowed applications import one platform surface module:

```c3
import gpu::surface::win32;
// or gpu::surface::wayland
// or gpu::surface::x11
```

SDL3 may supply the native handles in a consumer, but core declarations do not
mention `sdl::Window`, `vk::Device`, or `vma::Allocation`.

## 2. Naming rules

Use:

```text
create_device
allocate_memory
free_allocation
begin_commands
cmd_dispatch
cmd_barrier
```

Do not use project-owned OO-style constructors such as:

```text
Device.create
GpuAllocation.create
Texture.destroy
```

Methods are acceptable only for operations that clearly operate on an existing `&self` receiver and where the style guide allows them. Public lifecycle should prefer free functions.

## 3. Type taxonomy

### Backend, runtime, adapters, and device

```text
BackendKind
    VULKAN

ContractValidation
    TRUSTED
    OBJECT_BOUNDARIES
    FULL

RuntimeDesc
    BackendKind backend
    ContractValidation contract_validation
    bool track_resource_lifetimes
    bool enable_vulkan_validation
    bool enable_debug_names
    uint texture_heap_capacity      (0 = default; docs/limitations.md)
    uint sampler_heap_capacity
    uint texture_capacity
    uint pipeline_capacity
    char[] pipeline_cache_data      (copied warm-start blob; section 8)
    ZString application_name
    DebugMessageCallback debug_callback
    void* debug_user_data

Runtime                          (opaque generation token)
Adapter                          (borrowed runtime-owned token)
AdapterList
    Runtime runtime
    uint count
    get(uint index)              -> Adapter?

AdapterMemoryInfo
    ulong device_local_bytes
    ulong host_visible_bytes

AdapterQueueInfo
    uint graphics_count
    uint compute_count
    uint transfer_count

AdapterLimits
    uint max_texture_dimension_2d
    uint max_texture_array_layers
    uint max_color_attachments
    uint max_push_constant_size
    Vec3u max_compute_work_group_count
    uint max_draw_indirect_count

AdapterInfo
    String name
    uint vendor_id
    uint device_id
    AdapterClass device_class
    AdapterMemoryInfo memory
    AdapterQueueInfo queues
    bool strict_supported
    AdapterLimits limits

BackendVersion
    uint major
    uint minor
    uint patch

AdapterDiagnostics
    String backend_name
    BackendVersion backend_version
    String driver_name
    String driver_info
    uint driver_version             (backend-defined)

create_runtime(RuntimeDesc*)      -> Runtime?
enumerate_adapters(Runtime*)      -> AdapterList?
get_adapter_info(Adapter*)        -> AdapterInfo?
get_adapter_diagnostics(Adapter*) -> AdapterDiagnostics?
destroy_runtime(Runtime*)         -> void?
full_validation_runtime_desc()    -> RuntimeDesc
```

Validation controls are independent:

| Control | Zero/default | Effect |
|---|---|---|
| `contract_validation` | `TRUSTED` | Selects library contract diagnostics. Every level rejects stale/foreign identities at public resolve/destroy boundaries as mandatory safety. `OBJECT_BOUNDARIES` adds structured stale/foreign diagnostics at covered public create, query, destroy, command-lifecycle, and submit boundaries. `FULL` also enables detailed command argument, usage, layout, queue, pipeline-kind, render-compatibility, and state diagnostics. |
| `track_resource_lifetimes` | `false` | When true, command records retain explicitly named resources and early destruction returns `RESOURCE_IN_USE`. When false, recording allocates and updates no reference storage; the caller proves completion before destruction. |
| `enable_vulkan_validation` | `false` | Requests `VK_LAYER_KHRONOS_validation`. It does not select library checks or lifetime tracking. |
| `enable_debug_names` | `false` | Requests best-effort native object naming through debug utils. It enables no checks, tracking, or layers. |
| `debug_callback` | null | Selects structured delivery for diagnostics already produced by the contract policy or Vulkan. It does not enable checks, tracking, layers, or names. |

Every contract level retains the mandatory safety floor: host pointer and slice
validity before reads, integer-overflow and backing-range protection required
for safe lowering, command state transitions and internal table safety, public
device ownership, Vulkan result/device-loss handling, and transactional
creation rollback. Under `TRUSTED`, misuse outside that floor is a caller
contract violation and detailed command diagnostics are not promised.

Use the helper for the former all-enabled development configuration:

```c3
gpu::RuntimeDesc runtime_desc = gpu::full_validation_runtime_desc();
runtime_desc.application_name = "my_app";
```

`FULL` does not require Vulkan SDK layers. Set `contract_validation = FULL` and
leave `enable_vulkan_validation = false` when detailed library diagnostics are
needed on a machine without the Khronos layer. For migration only, a former
`enable_validation = true` initializer maps to `FULL`, lifetime tracking on,
and Vulkan validation on (or the helper); `false` or omission maps to
`TRUSTED`, tracking off, and Vulkan validation off. The retired field is not
part of `RuntimeDesc` and intentionally fails to compile.

Creating a runtime is the first operation that may initialize backend discovery. Enumeration returns an allocation-free view:

```c3
gpu::RuntimeDesc runtime_desc = { .backend = gpu::BackendKind.VULKAN };
gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
for (uint i = 0; i < adapters.count; i++) {
    gpu::Adapter adapter = adapters.get(i)!!;
    gpu::AdapterInfo info = gpu::get_adapter_info(&adapter)!!;
}
gpu::destroy_runtime(&runtime)!!;
```

Adapters and query-result strings are borrowed until runtime destruction. Borrowed strings are read-only and must not be modified. Backend and driver diagnostics are for logging; feature selection uses semantic adapter fields. Runtime destruction returns `RESOURCE_IN_USE` while a dependent surface or device is live.

### Runtime-owned surfaces

`Surface` is an opaque token owned by one runtime. Each platform module uses
distinct native handle types:

```text
gpu::surface::win32::create_surface(Runtime*, InstanceHandle, WindowHandle) -> Surface?
gpu::surface::wayland::create_surface(Runtime*, DisplayHandle, SurfaceHandle) -> Surface?
gpu::surface::x11::create_surface(Runtime*, DisplayHandle, WindowHandle) -> Surface?

supports_presentation(Adapter*, Surface*) -> bool?
destroy_surface(Surface*)                 -> void?
```

The adapter and surface must belong to the same runtime. A presentation device
accepts swapchains only for the exact surface named in its request and reports
that capability through `DeviceCaps.presentation_enabled`. Presentation may
use a private queue distinct from graphics. Destroying a surface with a live
swapchain returns `RESOURCE_IN_USE`; destroy surfaces before their runtime.
Destroying a requested surface with no live swapchain succeeds. Its presentation
device remains bound to that stale token, so future `create_swapchain` calls
return `INVALID_HANDLE`.

### Device requests and creation

Presentation and queue requirements are explicit additions to the immutable
strict request.

Strict device creation takes one exact borrowed adapter plus an immutable
semantic request. Support detection is read-only and enables nothing;
successful creation records strict and presentation enablement separately in
`DeviceCaps`. The unmet-requirement label is borrowed static text and names
GPU semantics rather than backend features.

```text
DeviceRequest                    (opaque immutable value)
DeviceRequestSupport
    bool supported
    String unmet_requirement     (borrowed static semantic label)

QueueRequirements
    QueueCounts counts
    QueueRoles distinct

strict_device_request()          -> DeviceRequest
request_presentation(DeviceRequest, Surface*) -> DeviceRequest?
request_queues(DeviceRequest, QueueRequirements) -> DeviceRequest?
supports_device_request(Adapter*, DeviceRequest*) -> DeviceRequestSupport?
create_device(Adapter*, DeviceRequest*) -> Device?
```

The strict request defaults to one graphics, compute, and transfer queue; one
native queue may satisfy several roles. Presentation requires at least one
graphics queue in both the default and an explicit queue group.
`request_queues` replaces that implicit default with one explicit group. Each
role count is 0..255 and requests distinct identities within that role. At
least one count must be nonzero. A role marked
`distinct` must have a nonzero count and may not alias another requested role.
Invalid or duplicate queue groups return `INVALID_ARGUMENT`. Support queries
report unavailable counts or topology without enabling device state.

A live adapter-created device retains its runtime and reuses the runtime-owned
backend instance. Device defaults are copied by `create_runtime` and inherited
by every device created from that runtime. Destroy each device before its runtime.

```text
DeviceCaps
    bool strict_enabled
    bool presentation_enabled
    bool buffer_device_address
    bool synchronization2
    bool dynamic_rendering
    bool shader_int64
    bool draw_indirect_count
    bool generated_work
    bool async_compute
    QueueCounts queues
    bool line_polygon_mode
    uint texture_heap_capacity
    uint sampler_heap_capacity
    uint max_color_attachments
    uint max_push_constant_size
    Vec3u max_compute_work_group_count
    uint max_draw_indirect_count
    uint max_generated_work_count
    usz min_uniform_alignment
    usz min_storage_alignment
    usz min_texel_buffer_alignment
    float max_sampler_lod_bias
    float max_sampler_anisotropy

Device                           (slot | generation | reserved)
get_device_backend(Device*)      -> BackendKind?
get_device_caps(Device*)         -> DeviceCaps?
```

`strict_enabled` reports whether strict semantics were requested and enabled.
The minimum supported device profile is intentionally Vulkan 1.3 plus
`VK_EXT_extended_dynamic_state3` and
`dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. The strict profile also
requires independent per-target blending and depth-bias clamp. Query request
support before creation; creation returns `UNSUPPORTED_FEATURE` when an adapter
cannot provide every requirement. `generated_work` is true only when the
created strict device enables GPU-written root and work records for graphics
and compute. A supported device reports a nonzero `max_generated_work_count`;
an unsupported device reports false and zero. Heap and generated-work
implementation mechanisms remain private.

Runtime heap capacities are exact semantic device defaults, not clampable upper
bounds. Device creation fails rather than clamping when the selected adapter
cannot satisfy them. On success,
`DeviceCaps.texture_heap_capacity` and `DeviceCaps.sampler_heap_capacity`
report the exact capacities of the created shader-visible heaps. A runtime capacity
above the library hard ceiling is malformed and faults `INVALID_ARGUMENT`; a
valid capacity unavailable on the selected adapter faults `UNSUPPORTED_FEATURE`.

Creation:

```text
strict_device_request() -> DeviceRequest
supports_device_request(Adapter*, DeviceRequest*) -> DeviceRequestSupport?
request_presentation(DeviceRequest, Surface*) -> DeviceRequest?
request_queues(DeviceRequest, QueueRequirements) -> DeviceRequest?
create_device(Adapter*, DeviceRequest*) -> Device?
destroy_device(Device*) -> void?
```

Malformed or empty requests fault before adapter/backend work. A valid
unsupported request returns `supported = false` with the first unmet semantic
label; passing it to `create_device` faults `UNSUPPORTED_FEATURE` without
selecting another adapter. Duplicate capability contribution is rejected
transactionally by private composition helpers.

Multiple live devices may coexist. `Device` is a compact slot and generation
token. Most public operations take a short-lived atomic pin before reading
backend state. `begin_commands` transfers its pin to the returned command token;
recording calls validate the token's stable encoder and acquire no additional
pin or device-registry operation.

`destroy_device` never waits. Live resources, command lists,
swapchains and descriptors return `RESOURCE_IN_USE`.
Active operations, incomplete queue work, or
a closing slot return retryable `DEVICE_BUSY`. Every failed attempt preserves
the token, generation, and backend state. Success increments the generation
and invalidates the passed token. A lost device bypasses child and progress
checks after operation pins retire. Lost command tokens remain discardable so
their lifetime pins cannot strand the device.

Queue tokens, command tokens, resource handles, descriptor
indices, GPU addresses/spans, and completion points are scoped to their
owning device. Backend table resolution rejects foreign handle owners before
resource mutation. Shader-visible indices and GPU addresses remain
caller-lifetime values.

### Handles

`Device` is a compact slot-and-generation token. Device-owned table handles
contain an opaque device-and-kind owner identity plus a local slot and generation.

```text
GpuAllocation
TextureHandle
TextureView
PipelineHandle
SwapchainHandle
```

Owning tokens have zero-valued invalid constants such as
`GPU_ALLOCATION_INVALID`, `TEXTURE_HANDLE_INVALID`, `TEXTURE_VIEW_INVALID`,
and their peers.
`token.is_valid()` checks the owner and generation; operations also validate
the local slot generation. Public code should not inspect or construct the
representation.

Handles, `TextureView`, `Queue`, `GpuSpan`, command tokens, and
synchronization values are runtime-only owner-bearing tokens scoped to one
device. `TextureIndex`, `SamplerIndex`, and `GpuAddress` are raw device-local
shader values without owner or generation metadata. Do not persist, serialize,
reconstruct, or pass either category across device or process lifetimes. Compare
`Queue` values as wholes and use `get_queue_info` for inspection; do not
construct or mutate queue fields. `CommandList` embeds a copy of its owning
`Device` token and does not borrow caller variable storage.

### Allocations, spans, and GPU addresses

`GpuAllocation` is an owning, generation-checked token. `GpuSpan` borrows a
range and contains no pointer, GPU address, memory class, access mask, backing
buffer, or ownership token.

```text
GpuAddress = ulong        (typedef — a distinct type, cast explicitly)

GpuSpan
    ulong owner
    uint index
    uint generation
    usz offset
    usz size
```

The identity fields are opaque. Consumers may copy a complete span only while
its owning allocation remains live. Do not construct or mutate the identity.
Mapping, address, access, bounds, and native backing remain device-owned and are
recovered when a public operation resolves the span. A zero `GpuAddress` is a
valid root value. Whether a shader may dereference it is application-defined
and depends on the device's robustness behavior.

`GpuSpan` slicing is explicit:

```text
span.checked_subspan(offset, size)   -> GpuSpan?
span.unchecked_subspan(offset, size) -> GpuSpan
```

`checked_subspan` rejects zero size, a range outside its immediate parent, and
offset overflow. It preserves the identity and changes only `offset` and
`size`. `unchecked_subspan` performs no bounds or overflow checks; use it
only after proving the range.

## 4. Faults

Public operations use C3 optionals/faults. `faultdef` declares a flat list of globally-unique fault values (there is no braced/named fault group in C3 0.8.0); these live in `module gpu` and are referenced as `gpu::INVALID_HANDLE`, raised with the `~` suffix. `gpu/faults.c3` documents each fault at its definition; the table below maps them to the operations that raise them.

Descriptor, configuration, barrier, viewport, scissor, and label pointers must
be non-null unless the API explicitly documents null as a value (such as
`TextureViewDesc* desc`). Null required input faults `INVALID_ARGUMENT` before a
backend call. Null or stale owner-token pointers fault `INVALID_HANDLE`.

| Fault | Fired by | Typical cause |
|---|---|---|
| `UNSUPPORTED_BACKEND` | `create_runtime` | the selected backend is unavailable |
| `UNSUPPORTED_FEATURE` | device creation, `create_runtime`, `create_texture`, `create_dedicated_texture`, `create_texture_view`, `create_texture_views`, `create_swapchain`, `create_graphics_pipeline`, `intern_sampler` | validation layers not installed; presentation was not requested or is unsupported for the adapter and surface; missing optional or required device feature; the selected adapter cannot provide the runtime's semantic heap capacities; unsupported image format or usage; adapter rejects a valid texture descriptor |
| `INVALID_ARGUMENT` | runtime adapter indexing; `request_queues`; any create/export; `allocate_memory`; `GpuSpan.checked_subspan`; `get_span_mapping`; `get_span_address`; `flush_mapped_span`; `invalidate_mapped_span`; `get_queue`; `submit`; `present`; `cmd_copy_buffer`/`cmd_fill_buffer`/buffer↔texture copies; draw/dispatch and barrier commands; `cmd_set_raster_state`/`cmd_set_depth_state`/`cmd_set_viewport`/`cmd_set_scissor`; `prepare_shader_code`; pipeline creates; `texture_transition`/`texture_view_transition`; `create_texture_views`; `intern_sampler` | null or malformed input, heap capacity above the library hard ceiling, zero allocation/span size, non-power-of-two alignment, unavailable mapping/address capability, range outside its immediate parent, offset overflow, `out_views.len != descs.len`, invalid queue access, missing resource usage, malformed command state data, or an out-of-range value |
| `INVALID_HANDLE` | runtime and adapter queries; destruction; device/queue/completion queries; allocation info/span/mapping/address/visibility operations; any resource-handle-taking call; `cmd_*`; command lifecycle; `submit` | zero, destroyed, stale, or foreign runtime, adapter, device, queue, completion point, allocation, span, resource, or command token |
| `INVALID_RESOURCE_STATE` | swapchain lifecycle; `destroy_attachment_view`; `release_generated_scratch` | an acquired swapchain image is pending during resize or destruction, a readiness/acquisition state transition is invalid, a borrowed swapchain attachment view was passed for destruction, or the requested generated-scratch key is not reserved |
| `OUT_OF_HOST_MEMORY` | creates; mapped visibility | driver or backend cache host-allocation failure |
| `OUT_OF_DEVICE_MEMORY` | allocation and texture creates; mapped visibility | backend device-memory exhaustion |
| `DEVICE_LOST` | any Vulkan-backed operation | Vulkan returned `VK_ERROR_DEVICE_LOST`; the affected device rejects later operations while peer devices remain usable |
| `DEVICE_BUSY` | public device operations; `submit`; `destroy_device` | the operation observed a closing device or exhausted bounded pin acquisition, submission reached the native timeline-value-difference limit, or destruction found an active operation or incomplete queue work; retry with the unchanged token |
| `RESOURCE_IN_USE` | resource or swapchain destruction/resize, `free_allocation`, generated-scratch reservation/release, `destroy_device`, `destroy_runtime`, `destroy_surface` | recording, executable, incomplete submitted work, a texture view, generated-scratch reservation, or unfinished presentation still references the resource; a placed or dedicated texture depends on an allocation; a device has a live child; a runtime has a live surface or device; or a surface has a live swapchain |
| `SLOT_TABLE_FULL` | runtime, device, allocation, and resource creates; `intern_sampler`; `begin_commands`; `acquire_next_image`; queue submission | a registry or handle table is at capacity, or a queue completion or swapchain acquisition sequence is exhausted |
| `GENERATED_SCRATCH_EXHAUSTED` | generated draw/dispatch recording | the calling thread has no compatible reserved preprocess buffer for the pipeline, generated-work kind, command count, or concurrent retained-list demand |
| `DESCRIPTOR_HEAP_FULL` | descriptor pool creation/allocation, `create_texture_view`, `create_texture_views`, `intern_sampler` | Vulkan descriptor-pool exhaustion or fragmentation, or capacity below the live descriptor count; overflowing texture batches and sampler interning leave existing entries untouched |
| `PIPELINE_CREATE_FAILED` | pipeline creates | driver rejected the state combination, shader, or compilation |
| `SHADER_INVALID` | `prepare_shader_code`, pipeline creates | malformed SPIR-V structure or backend reflection/module rejection |
| `SURFACE_LOST` | surface creation/query/enumeration, swapchain create/resize, acquire, present | native window or surface was destroyed or became unavailable; destroy the swapchain and create a new one from fresh native handles |
| `SWAPCHAIN_OUT_OF_DATE` | `create_swapchain`, `resize_swapchain`, `acquire_next_image`, `present` | swapchain no longer matches the surface; `resize_swapchain` and retry |
| `COMMAND_RECORDING_ERROR` | `cmd_*`, `end_commands`, `discard_commands`, `discard_executable_commands`, `submit` | call outside its required recording state, execution without a bound pipeline, draw without required per-pass depth state, duplicate command token in one submit batch, or token that is already being submitted |
| `WAIT_TIMEOUT` | `wait_completion`, `acquire_next_image`, `present` | bounded wait, transient image unavailability, or a private present fence not yet reusable; retry with the unchanged value |
| `BACKEND_ERROR` | any Vulkan-backed operation | unclassified or internal native failure; inspect backend diagnostics; does not imply device loss |

Backend-local Vulkan/VMA faults should not leak unless they carry useful public meaning. Map them to public faults and report backend details through the configured diagnostic path. `DEVICE_LOST` is reserved for an explicit native device-loss result; an unmapped native result becomes `BACKEND_ERROR`.

## 5. Memory API

### Independent allocations

```text
MemoryClass.CPU_WRITE
MemoryClass.GPU_PRIVATE
MemoryClass.CPU_READ
MemoryClass.TEXTURE

AllocationDesc
    usz size
    usz alignment
    MemoryClass memory_class
    QueueRoles access
    TextureRequirements[] texture_requirements
    ZString debug_name

AllocationInfo
    usz size
    usz alignment
    MemoryClass memory_class
    QueueRoles access
    bool mapped
    bool coherent
    bool addressable

allocate_memory(Device*, AllocationDesc*) -> GpuAllocation?
free_allocation(Device*, GpuAllocation*) -> void?
get_allocation_info(Device*, GpuAllocation) -> AllocationInfo?
get_allocation_span(Device*, GpuAllocation) -> GpuSpan?
get_span_mapping(Device*, GpuSpan) -> char[]?
get_span_address(Device*, GpuSpan) -> GpuAddress?
flush_mapped_span(Device*, GpuSpan) -> void?
invalidate_mapped_span(Device*, GpuSpan) -> void?
```

`CPU_WRITE` is mapped for host writes, `GPU_PRIVATE` is addressable GPU
data, and `CPU_READ` is mapped for host reads. `TEXTURE` is unmapped,
non-addressable placement storage. These are behavioral classes, not backend
heap or property selectors.

`AllocationDesc.size` must be nonzero. Alignment zero selects 16 bytes; an
explicit alignment must be a power of two and is raised to at least 16.
`access` must be a non-empty subset of the device's selected queue roles.
`AllocationInfo` reports immutable properties and actual mapping, coherence,
and address capabilities.

`get_allocation_span` borrows the complete allocation. Mapping and address
queries resolve a live span; mapping returns its exact byte range and the
address points to its first byte. A mapping query on an unmapped span and an
address query on a non-addressable span fault
`INVALID_ARGUMENT`. Returned mappings and addresses remain valid only while
the allocation remains live.

`flush_mapped_span` publishes CPU writes before GPU use.
`invalidate_mapped_span` publishes completed GPU writes before CPU reads; wait
or poll the relevant `CompletionPoint` first. Neither call waits. Both accept
only live, mapped independent-allocation spans. Coherent ranges return success
without native work; non-coherent alignment stays backend-private.

`free_allocation` releases storage immediately, so all GPU use must be
quiescent. Success invalidates the token and every borrowed span; faults
preserve it. Validation returns `RESOURCE_IN_USE` for detected explicit command
references and for live placed or dedicated textures. Uses reachable only by a
raw GPU address remain the caller's precondition. Any live allocation prevents
normal device destruction.

### Host transfers

The strict core exposes primitives, not transfer policy. For long-lived or
one-shot CPU-written data, allocate `CPU_WRITE` memory, borrow its span,
mapping, and address as needed, write, flush the span, record and submit the
work, wait for or poll its covering completion point, then free the owning
`GpuAllocation`. Do not assume the mapping is coherent;
`flush_mapped_span` is required before GPU use.

For GPU-to-CPU data, allocate `CPU_READ` memory, borrow its span and mapping,
record the copy and a global barrier with `before.transfer` and `after.host`,
submit, wait for or poll completion, invalidate the span, then read its mapping.
Free or reuse the owning allocation only after completion.

Applications choose whether to reuse allocations, suballocate rings, or create
one-shot storage. `GpuSpan.checked_subspan` defines partial transfers.

## 6. Texture API

### Formats

The public `Format` enum should contain only formats supported by the library:

```text
UNDEFINED
R8_UNORM
R8_UINT
RG8_UNORM
RGBA8_UNORM
RGBA8_SRGB
BGRA8_UNORM
BGRA8_SRGB
R16_UINT
R16_FLOAT
RG16_FLOAT
RGBA16_FLOAT
R32_UINT
R32_FLOAT
RG32_FLOAT
RGBA32_FLOAT
D32_FLOAT
```

### Textures and shader-visible views

`TextureUsage` is likewise a bitstruct of bool flags.

```text
bitstruct TextureUsage : uint
    bool sampled      : 0
    bool storage      : 1
    bool color_attach : 2
    bool depth_attach : 3
    bool transfer_src : 4
    bool transfer_dst : 5

TextureFormatFeatures
    bool sampled
    bool storage
    bool color_attach
    bool depth_attach
    bool transfer_src
    bool transfer_dst
    bool linear_filter

TextureSampleCountSupport
    bool one
    bool two
    bool four
    bool eight
    bool sixteen
    bool thirty_two
    bool sixty_four

TextureFormatSupport
    TextureFormatFeatures features
    TextureSampleCountSupport sample_counts

TextureCompatibility
    opaque device-owned value

TextureRequirements
    usz size
    usz alignment
    TextureCompatibility compatibility
    bool dedicated_only

DedicatedTexture
    TextureHandle texture
    GpuAllocation allocation

SampleCount
    ONE
    TWO
    FOUR
    EIGHT
    SIXTEEN
    THIRTY_TWO
    SIXTY_FOUR

TextureDesc
    uint width
    uint height
    uint mip_levels
    uint array_layers
    Format format
    TextureUsage usage
    QueueRoles access
    SampleCount sample_count
    ZString debug_name

TextureViewDesc
    uint base_mip
    uint mip_count
    uint base_layer
    uint layer_count

TextureIndex : uint
    uint value : 0..31

TextureView
    ulong owner
    TextureIndex index
    uint generation

TextureViewCreateDesc
    TextureHandle texture
    TextureViewDesc view
```

### Texture functions

```text
get_texture_format_support(Device* device, Format format) -> TextureFormatSupport?
supports_texture_desc(Device* device, TextureDesc* desc) -> bool?
get_texture_requirements(Device* device, TextureDesc* desc) -> TextureRequirements?
create_texture(Device* device, TextureDesc* desc) -> TextureHandle?
create_placed_texture(Device* device, TextureDesc* desc, GpuAllocation allocation, usz offset) -> TextureHandle?
create_dedicated_texture(Device* device, TextureDesc* desc, AllocationDesc* allocation_desc) -> DedicatedTexture?
destroy_texture(Device* device, TextureHandle texture) -> void?
create_texture_view(Device* device, TextureHandle texture, TextureViewDesc* desc) -> TextureView?
destroy_texture_view(Device* device, TextureView view) -> void?
create_texture_views(Device* device, TextureViewCreateDesc[] descs, TextureView[] out_views) -> void?
```

`get_texture_format_support` reports library-creatable support, not every raw
Vulkan capability. Each usage bit comes from the same exact 2D optimal-tiling
query used by creation, but the bits are independent; use
`supports_texture_desc` for a usage combination. The backend profile masks every
dimension except 2D. Higher sample-count bits report exact color-attachment or
depth-attachment descriptors, as appropriate for the format. Per-format usages,
sample counts, and linear filtering remain adapter-dependent. Depth support is
D32-only.
`supports_texture_desc` returns false for an empty, unknown, or unavailable
access set.

`supports_texture_desc` checks the exact optimal-tiling format, combined usage,
normalized extent, mip and layer counts, access roles, sample count, and image
properties without allocating. Multisample textures require one mip and
attachment-only usage. A false result caused by malformed input corresponds to
`INVALID_ARGUMENT` at creation; a structurally valid descriptor rejected by the
adapter corresponds to `UNSUPPORTED_FEATURE`. Memory exhaustion can still make
creation fail after a true capability result.

`get_texture_requirements` returns size, alignment, a device-owned opaque
compatibility value, and whether dedicated storage is required. Pass every
requirement an allocation must support in
`AllocationDesc.texture_requirements`; incompatible groups are rejected.

`create_placed_texture` requires `MemoryClass.TEXTURE`, compatible memory,
an aligned, in-bounds, non-overlapping range, and allocation access covering
the texture access. Destroying the texture does not release its allocation.
Dedicated-only requirements return `UNSUPPORTED_FEATURE`.

`create_dedicated_texture` creates the image, allocates compatible dedicated
memory, binds it, and publishes both ownership tokens atomically. The allocation
size must equal the queried requirement; its alignment and access must cover the
texture. Destroy the texture before releasing the allocation. A premature
release returns `RESOURCE_IN_USE` without consuming the allocation.

`TextureHandle` owns the image. `create_texture_view` returns a `TextureView`, an
owner- and generation-checked CPU lifetime token whose `index` field is the raw
32-bit value stored in shader data. Zero is invalid. Shader indices carry no
generation bits; destroying a view immediately makes its index reusable. First
discard or complete every use and remove the index from GPU-visible data.
Passing a stale or foreign view to `destroy_texture_view` faults before heap
mutation. Texture-view publication requires strict capability and returns
`UNSUPPORTED_FEATURE` before backend work otherwise. Distinct subresource views
are governed by the device-wide heap capacity, with no smaller fixed per-texture
publication limit. Sampler indices remain stable until device
destruction.

`create_texture_views` batch-publishes N views under one lock hold and ends in
one accumulated descriptor-set update in indexing mode. Descriptor-buffer mode
writes each mapped entry directly. `out_views.len` must equal `descs.len`
(`INVALID_ARGUMENT` otherwise); an empty input is a no-op success. A
zero-initialized `TextureViewCreateDesc.view` selects the default view, matching
a null `desc` passed to `create_texture_view`.

The batch is all-or-nothing: a fault leaves heap cells and generations,
allocator/free-list state, cached native views, Vulkan image-view ownership, and
`out_views` unchanged. Only a successful batch returns owner-bearing views;
release each with `destroy_texture_view`.

## 7. Sampler API

```text
Filter
    NEAREST
    LINEAR

AddressMode
    REPEAT
    MIRRORED_REPEAT
    CLAMP_TO_EDGE
    CLAMP_TO_BORDER

SamplerDesc
    Filter min_filter
    Filter mag_filter
    Filter mip_filter
    AddressMode address_u
    AddressMode address_v
    AddressMode address_w
    float mip_lod_bias
    float min_lod
    float max_lod
    bool anisotropy_enable
    float max_anisotropy
    bool compare_enable         (depth-compare sampler; shadow maps)
    CompareOp compare
    ZString debug_name

intern_sampler(Device* device, SamplerDesc* desc) -> SamplerIndex?
```

`intern_sampler` returns one stable shader-visible index. Descriptions with
the same effective filtering, addressing, LOD, anisotropy, and comparison state
return the same `SamplerIndex`; `debug_name` is not part of identity. LOD values must
be finite, the absolute `mip_lod_bias` must not exceed
`DeviceCaps.max_sampler_lod_bias`, and `min_lod` must not exceed `max_lod`.
Undefined filter, address, or enabled comparison enum values fault
`INVALID_ARGUMENT`. Anisotropy requires the reported device capability. Sampler
indices and their native objects live until device destruction and have no
individual destroy operation. Repeated interning is idempotent.
`DESCRIPTOR_HEAP_FULL` consumes no table or heap entry; a device without strict
capability returns `UNSUPPORTED_FEATURE` before backend mutation.

### Breaking migration

Texture shape is implicitly 2D, view format is always the texture's format,
and sampler interning now returns the shader index in one call:

```c3
gpu::TextureDesc texture_desc = {
    .width  = 1024,
    .height = 1024,
    .format = gpu::Format.RGBA8_UNORM,
    .usage  = { .sampled },
    .access = { .graphics },
};
gpu::SamplerIndex index = gpu::intern_sampler(device, &sampler_desc)!!;
```

No placeholder texture-shape or view-format fields are required.

## 8. Shader and pipeline API

### Shader code

```text
ShaderStage
    COMPUTE
    VERTEX
    FRAGMENT

ShaderDesc
    ShaderStage stage
    char[] spirv
    ZString entry_point
    ZString debug_name

ShaderCode
    ShaderStage stage
    char[] spirv
    ZString entry_point
    ZString debug_name
    ulong digest

prepare_shader_code(ShaderDesc* desc) -> ShaderCode?
```

Shader compilation can be handled by tools or samples. The core library consumes
borrowed SPIR-V bytes. `prepare_shader_code` validates their basic structure,
normalizes a null entry point to `main`, and computes the library-owned identity.
The caller keeps the bytes and strings immutable and alive whenever the value is
used. The digest is opaque and process-local: do not inspect, modify, serialize,
or persist it. Stage, entry point, length, and exact bytes participate in shader
identity; `debug_name` does not. One prepared value may be reused across
pipelines and devices. There is no public shader-module handle.

### Compute pipelines

```text
ComputePipelineDesc
    ShaderCode shader
    ZString debug_name

create_compute_pipeline(Device* device, ComputePipelineDesc* desc) -> PipelineHandle?
create_compute_pipelines(
    Device* device,
    ComputePipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) -> void?
```

Every compute pipeline uses the device's singleton layout with the fixed 8-byte
`RootPush` range. Compute layout size is not pipeline identity or caller policy.

### Graphics pipelines

```text
PrimitiveTopology
    TRIANGLES
    LINES
    POINTS

DepthState
    bool test_enable
    bool write_enable
    CompareOp compare

BlendState
    bool enable
    BlendFactor src_color
    BlendFactor dst_color
    BlendOp color_op
    BlendFactor src_alpha
    BlendFactor dst_alpha
    BlendOp alpha_op

ColorWriteMask
    bool red
    bool green
    bool blue
    bool alpha

ColorTargetState
    Format format
    BlendState blend
    ColorWriteMask write_mask

DynamicRasterState
    PrimitiveTopology topology
    CullMode cull_mode
    FrontFace front_face
    bool depth_bias_enable
    float depth_bias_constant
    float depth_bias_slope
    float depth_bias_clamp

GraphicsPipelineDesc
    ShaderCode vertex_shader
    ShaderCode fragment_shader
    ColorTargetState[] colors
    Format depth_format
    SampleCount sample_count
    PolygonMode polygon_mode
    ZString debug_name

create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) -> PipelineHandle?
create_graphics_pipelines(
    Device* device,
    GraphicsPipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) -> void?
destroy_pipeline(Device* device, PipelineHandle pipeline) -> void?
```
`PolygonMode.LINE` is optional. Query `DeviceCaps.line_polygon_mode` before
using it; unsupported LINE creation returns `UNSUPPORTED_FEATURE`.
`PrimitiveTopology.LINES` remains available independently with FILL mode and is
selected with `cmd_set_raster_state`.

`colors` carries at most `MAX_COLOR_ATTACHMENTS` (8) entries. Format, blend,
and write mask are specified independently for every target. The zero write
mask disables all writes; use `COLOR_WRITE_ALL` for the conventional RGBA
mask. In particular, a zero-initialized `ColorTargetState` that sets only
`.format` creates a valid target that renders no color. Enabled blending is
invalid for integer color formats. A disabled blend equation, or any blend
equation paired with a zero write mask, is normalized out of pipeline identity.

### Pipeline deduplication

Pipeline creation deduplicates through a descriptor-keyed cache. Every create
returns a fresh handle, but descriptors identical in immutable state (exact
shader code identity, polygon mode, per-target format/blend/write masks, depth
format, and sample count) alias one backend pipeline underneath. Compute
identity is shader identity because its `RootPush` layout is fixed. Digest collisions are
resolved by stage, entry point, length, and exact SPIR-V bytes. Topology,
cull/front-face, depth bias, depth test/write/compare, viewport, and scissor are
separate dynamic command state.

Batch output length must equal descriptor count; an empty batch succeeds.
Shared shader code creates one temporary native module per exact stage, entry
point, length, and byte sequence in the batch. Duplicate pipeline identities
compile once. A fault leaves all output handles unchanged and publishes no
pipeline from the batch.

Each successful create must be balanced by exactly one `destroy_pipeline`; the
backend pipeline is destroyed when its last alias is released. Destroying a
handle twice faults `INVALID_HANDLE` and never affects other aliases. Handles
must not be compared to decide whether two pipelines are "the same object" —
distinct handles may or may not share backend state.

### Pipeline cache

Identical immutable-state descriptors alias one backend pipeline (in-memory
dedup, per device). The driver cache is additionally serializable:

```text
get_pipeline_cache_size(Device* device) -> usz?
get_pipeline_cache_data(Device* device, char[] out) -> usz?   (bytes written)
RuntimeDesc.pipeline_cache_data                                 (warm-start blob)
```

Export must not race pipeline creation; blob usefulness is driver-dependent
(docs/limitations.md). `pipeline_cache_timing` demonstrates the round-trip.

## 9. Command API

### Threading

Command tokens are thread-confined. Different threads may call
`begin_commands` for the same device concurrently; the backend owns and caches
recording storage per worker. A recording or executable token and its aliases
must not be used concurrently. See `docs/threading.md`.

### Command lifecycle

```text
get_queue_counts(Device* device) -> QueueCounts?
get_queue(Device* device, QueueKind kind, uint index = 0) -> Queue?
get_queue_info(Device* device, Queue queue) -> QueueInfo?
begin_commands(Queue queue) -> CommandList?
end_commands(CommandList* commands) -> ExecutableCommandList?
discard_commands(CommandList* commands) -> void?
discard_executable_commands(ExecutableCommandList* commands) -> void?
submit(Queue queue, SubmitDesc* desc) -> CompletionPoint?
poll_completion(CompletionPoint point) -> bool?
wait_completion(CompletionPoint point, ulong timeout_ns = ulong::max) -> void?

CompletionPoint
    Device device
    ulong payload

CompletionWait
    CompletionPoint point
    StageMask before

Queue
    Device device
    uint id
    QueueRoles roles

QueueCounts
    uint graphics
    uint compute
    uint transfer

QueueRoles
    bool graphics
    bool compute
    bool transfer

QueueInfo
    uint id
    QueueRoles roles

SubmitDesc
    ExecutableCommandList[] command_lists
    CompletionWait[] completion_waits
    SwapchainReadiness readiness
    StageMask readiness_before
```

`CompletionPoint` is a reusable queue-progress value. It fits in two machine
words and allocates no public synchronization object. `poll_completion` returns
false while work is incomplete. `wait_completion` returns `WAIT_TIMEOUT` when
the bound expires; either operation leaves the point unchanged. Zero, stale,
foreign-device, malformed, and unpublished points fault `INVALID_HANDLE`.

`CommandList` is a recording token. Successful `end_commands` consumes it and
returns a one-shot `ExecutableCommandList`; failure leaves the recording token
unchanged. Copies alias the same backend record. Recording and executable tokens
retain the device until consumed.

Both command token types contain an opaque encoder pointer in addition to the
owning device and handle. The token layout, including that pointer, is part of
the public ABI. Begin publishes the encoder only after backend and retained-pin
setup succeeds; end transfers the same encoder to the executable token. A failed
end, discard, or submit leaves the token and encoder phase retryable.

`discard_commands` consumes unfinished recording. Use
`discard_executable_commands` for an ended token that will not be submitted.
Both remain available after device loss.

Submission targets an explicit `Queue`. Every executable token in the batch
must have been recorded for that queue. Success consumes all tokens and returns
one queue-owned `CompletionPoint`; failure publishes no point and preserves the
tokens for retry or discard. An empty batch signals the selected queue.
Duplicate or non-executable tokens fault `COMMAND_RECORDING_ERROR`; a token for
another queue faults `INVALID_ARGUMENT`. Discard consumes an unsubmitted token.
Completed native command buffers return to their recording context. Only that
context owner resets and reuses them after completion or discard; native
allocation is a cold-path fallback only when no reusable buffer is available.

`SubmitDesc.completion_waits` accepts published points from the same device.
Each `CompletionWait.before` names the first destination stages that consume
the dependency. It must be nonempty, supported by the destination queue, and
must not contain `host` or `present`; `all` is allowed only by itself. Unknown
or unsupported masks fault `INVALID_ARGUMENT`. Cross-queue points become waits
on their queue-owned timelines with the exact requested stage mask. Published
points from the target queue are validated and then elided because queue order
is inherent. Stale, unpublished, malformed, and foreign-device points fault
`INVALID_HANDLE` before native submission and preserve every command token.
Because the public wait mask has no draw-argument or command-preprocess bit,
dependencies consumed as indirect or generated command arguments require
`.all`; shader-stage bits begin too late to order those argument reads.
If outstanding queue progress reaches the device's timeline-value-difference
limit, submission faults retryable `DEVICE_BUSY` before reserving a sequence.

`Queue` is a small device-owned identity for a selected queue. Its `device`
field is a copied `Device` token used for exact ownership validation; it does not
borrow caller storage. `get_queue_counts` reports selected counts by semantic
role. Each role index names a distinct selected identity; identities may satisfy
several roles unless the request marks a role `distinct`. `get_queue` faults
`INVALID_HANDLE` for a non-live device and `INVALID_ARGUMENT` for an
unavailable role index.
`get_queue_info` faults `INVALID_HANDLE` for zero, stale, foreign-device, or
malformed tokens and returns the stable device-local ID and supported roles.
Backend family indices and native handles remain private. Resource descriptions
require a non-empty subset of the selected roles. A queue may use a resource when
at least one of its roles appears in the resource's `access` set. A `GpuSpan` also rejects empty,
unknown, or wider-than-backing access metadata. These checks run before command
or command mutation.

Allocations reached only through root GPU pointers remain a caller contract:
every reachable allocation must admit the recording role because nested pointers
are opaque. The backend keeps one admitted family exclusive and uses private
concurrent sharing only for distinct admitted families.

Transfer/render helper descriptors (`BufferCopyDesc`, `BufferTextureCopyDesc`,
`TextureBufferCopyDesc`, `ClearColor`,
`ClearDepth`) are documented in the generated reference.

### Pipeline command state and dispatch

```text
cmd_bind_pipeline(CommandList* commands, PipelineHandle pipeline) -> void?
cmd_set_raster_state(CommandList* commands, DynamicRasterState* raster) -> void?
cmd_set_depth_state(CommandList* commands, DepthState* depth) -> void?
cmd_dispatch(
    CommandList* commands,
    GpuAddress root,
    Vec3u groups,
) -> void?
```

Pipeline binding selects the active compute or graphics pipeline without
compiling or synthesizing a variant. A failed bind preserves the previous active
pipeline. A successful bind generation-checks the public handle, retains the
pipeline when `track_resource_lifetimes` is enabled, and caches its native
pipeline/layout, kind, render compatibility, cache-entry identity,
generated-work layout, and public diagnostic identity. Later draws and
dispatches use that snapshot without reading a pipeline cell, table, or cache.
Tracked ownership prevents destruction until discard or command completion;
without tracking, the caller must keep the pipeline live through completion.
Dispatch requires an active compute pipeline. Graphics draws require an active
graphics pipeline and explicit depth state for the current render pass. Root
addresses, including zero, are pushed unchanged.
`cmd_set_raster_state` and `cmd_set_depth_state` require an active render pass;
outside one they fault `COMMAND_RECORDING_ERROR`. Raster validation is atomic:
invalid enum values or non-finite enabled depth-bias factors return
`INVALID_ARGUMENT` without emitting any native state.
Pass begin emits a zero `DynamicRasterState` and resets the explicit depth-state
requirement, while viewport and scissor receive their full-pass defaults.
Execution with no required state returns
`COMMAND_RECORDING_ERROR`; a wrong active pipeline kind returns
`INVALID_ARGUMENT`.

Each group count may be zero and must not exceed the corresponding component
of `DeviceCaps.max_compute_work_group_count`. An over-limit call faults
`INVALID_ARGUMENT` before a backend command is recorded.

Warm recording dispatches through the immutable operation table selected for
the encoder. There is currently one checked policy. Policy selection happens at
device or encoder setup; a warm `cmd_*` call never branches on policy or reloads
lifecycle dispatch.

Test builds and builds with `COMMAND_RESOLUTION_STATS` expose
`CommandResolutionStats`, `reset_command_resolution_stats`, and
`command_resolution_stats`. The process-wide relaxed counters measure
live-encoder entry-point attempts, every emitted Vulkan command, and forbidden
resolution paths. Reset and compare them only across an externally synchronized
recording interval; they are absent from ordinary production builds.

### Render pass

```text
ClearColor
    float[4] rgba
    uint[4] uint_rgba

AttachmentViewDesc
    TextureHandle texture
    uint mip_level
    uint array_layer

create_attachment_view(Device* device, AttachmentViewDesc* desc) -> AttachmentViewHandle?
destroy_attachment_view(Device* device, AttachmentViewHandle view) -> void?

ColorTargetDesc
    AttachmentViewHandle view
    AttachmentViewHandle resolve_view
    LoadOp load_op
    StoreOp store_op
    ClearColor clear

DepthTargetDesc
    AttachmentViewHandle view
    LoadOp load_op
    StoreOp store_op
    ClearDepth clear

RenderPassDesc
    ColorTargetDesc[] colors
    DepthTargetDesc* depth
    uint width
    uint height

cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) -> void?
cmd_end_render_pass(CommandList* commands) -> void?
```

An `AttachmentViewHandle` is an immutable device child for one texture mip and
array layer. Create every color, resolve, and depth view before recording and
destroy it only after every recorded or submitted reference has completed.
Creation validates texture ownership and subresource bounds and creates any
required native image view on the cold path. Pass begin resolves only the fixed
attachment-view table and records from fixed-size stack storage; it neither
creates image views nor grows a cache.

Each device has a fixed table of 4096 attachment views. User-created views and
the borrowed view for every live swapchain image share this table. View or
swapchain creation returns `SLOT_TABLE_FULL` when all slots are live or
permanently retired by generation exhaustion.

Attachment views are distinct from shader-visible `TextureView` values.
User-created views retain their texture and must be destroyed by the caller.
Borrowed swapchain views do not retain their texture because the swapchain owns
and invalidates both handles together; callers cannot destroy those views. No
attachment view is published in a descriptor heap or can be destroyed while a
command list holds an explicit reference. Zero, stale, foreign-device, and
mismatched view handles fault before native recording.

Render-pass begin initializes one viewport and one scissor to the full pass
extent and initializes raster state to triangles, no culling,
counter-clockwise front faces, and disabled depth bias. Callers may override
these states for subsequent draws:

```text
Viewport
    float x
    float y
    float width
    float height
    float min_depth
    float max_depth

ScissorRect
    int x
    int y
    int width
    int height

cmd_set_viewport(CommandList* commands, Viewport* viewport) -> void?
cmd_set_scissor(CommandList* commands, ScissorRect* scissor) -> void?
cmd_set_raster_state(CommandList* commands, DynamicRasterState* raster) -> void?
```

A conventional full-depth viewport must set `max_depth` explicitly because
C3 aggregate literals zero omitted fields:

```c3
gpu::Viewport viewport = {
    .x         = 0.0f,
    .y         = 0.0f,
    .width     = 640.0f,
    .height    = 480.0f,
    .min_depth = 0.0f,
    .max_depth = 1.0f,
};
gpu::cmd_set_viewport(&commands, &viewport)!!;
```

All three setters are valid only inside a render pass. Viewports require
finite, nonnegative origins, positive extents, pass-local endpoints, and depth
endpoints in `[0, 1]`; reversed depth ranges are valid. Scissors use signed
inputs so negative origins/extents fault, while zero extent is a valid empty
clip. Scissor endpoints must not overflow and both rectangles stay within the
active pass. Invalid values return `INVALID_ARGUMENT` before changing dynamic
state; calls outside a pass return `COMMAND_RECORDING_ERROR`.

Explicit viewport/scissor/raster state survives graphics pipeline and
cache-alias handle switches. The next render-pass begin restores the full-pass
and zero-raster defaults.
The current API intentionally exposes one rectangle only and does not support
negative-height viewport flips or off-pass overscan.

Use `ClearColor.rgba` for normalized and floating-point attachments, and
`ClearColor.uint_rgba` for unsigned-integer attachments. The inactive union
member is ignored.

A pass names at least one color target or a depth target; depth-only passes
(the shadow-map shape) are valid. A depth target's texture needs `depth_attach`
usage and an explicit transition to a `DEPTH_ATTACHMENT` state. `D32_FLOAT` is
the only supported depth format; pipelines name it in
`GraphicsPipelineDesc.depth_format`. Every selected color and depth view must
cover the pass dimensions; smaller compatible render
areas are valid. All color and depth sources use the same sample count. The
color count must not exceed `DeviceCaps.max_color_attachments`, which is the
lesser of the library ceiling and the selected device's Vulkan limit.

A color target may resolve a multisample source into a distinct, single-sample
view with the same format and sufficient selected-mip extent. Both textures
need `color_attach` usage and explicit transitions to
a `COLOR_ATTACHMENT` state. Normalized and floating-point formats average
samples; integer formats select sample zero. Depth resolve is not exposed.

A bound graphics pipeline must exactly match the pass color count and formats,
depth format, and sample count. Compatibility is checked before render-pass
begin or pipeline bind changes native recording state or retained references.
A graphics binding is scoped to its render pass: `cmd_end_render_pass`
releases it, so a multi-pass command list binds a pipeline inside (or
immediately before) each pass. Compute bindings persist across passes.
Render-pass boundaries and resolves add no synchronization; callers declare
all attachment transitions and later shader/transfer visibility explicitly.

Depth clear values are explicit: a zero-initialized `ClearDepth`
clears depth to **0.0**, which fails every LESS-compare draw. The standard
far-plane clear is an explicit `{ .depth = 1.0 }`; reverse-Z setups clear to
0.0 deliberately.

### Draw

```text
cmd_draw(
    CommandList* commands,

    GpuAddress vertex_root,
    GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) -> void?

cmd_draw_indexed(
    CommandList* commands,

    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan index_span,
    uint index_count,
    uint instance_count,
    IndexType index_type = IndexType.U32,   (enum: U32, U16)
) -> void?
```

### Indirect execution

Argument layouts match Vulkan byte-for-byte and are shader-writable (GLSL
twins generated into `include/shaders/generated/shader_abi.glsl` by the ABI
generator — see `docs/shader_abi.md` §12):

```text
DrawIndirectCommand        { vertex_count, instance_count, first_vertex, first_instance }
DrawIndexedIndirectCommand { index_count, instance_count, first_index, vertex_offset, first_instance }
DispatchIndirectCommand    { x, y, z }

GeneratedDrawRecord        { vertex_root_gpu, fragment_root_gpu, arguments }
GeneratedDrawIndexedRecord { vertex_root_gpu, fragment_root_gpu, arguments, _pad0 }
GeneratedDispatchRecord    { root_gpu, arguments, _pad0 }

GeneratedWorkKind          { DRAW, DRAW_INDEXED, DISPATCH }

GeneratedScratchDesc
    PipelineHandle pipeline
    GeneratedWorkKind kind
    uint max_commands_per_list
    uint preprocess_buffer_count

reserve_generated_scratch(Queue queue, GeneratedScratchDesc* desc) -> void?
release_generated_scratch(
    Queue queue,
    PipelineHandle pipeline,
    GeneratedWorkKind kind,
) -> void?

cmd_draw_indirect(commands, vertex_root, fragment_root, args, draw_count) -> void?
cmd_draw_indexed_indirect(commands, vertex_root, fragment_root, args, draw_count, index_span, index_type) -> void?
cmd_draw_indexed_indirect_count(commands, vertex_root, fragment_root, args, count_span, max_draw_count, index_span, index_type) -> void?
cmd_dispatch_indirect(commands, root, args) -> void?
cmd_draw_generated(commands, records, count_span, max_draw_count) -> void?
cmd_draw_indexed_generated(commands, records, count_span, max_draw_count, index_span, index_type) -> void?
cmd_dispatch_generated(commands, records, count_span, max_dispatch_count) -> void?
```

The generated records have fixed std430 strides of 32, 40, and 24 bytes. They
pair the roots and arguments written by a GPU producer without a parallel
`gl_DrawID` lookup table. `DeviceCaps.generated_work` reports whether the
created device supports all three commands. Unsupported devices fault
`UNSUPPORTED_FEATURE`; the shared-root indirect commands remain the portable
execution path and the library never emulates generated work with a CPU loop.

Generated commands require an explicit cold reservation on the calling
thread's device recording context. The queue selects and validates the device;
the reservation can be consumed by compatible selected queues on that device.
Each reservation is keyed by the exact public `PipelineHandle` and
`GeneratedWorkKind`, not by the shared native pipeline. Two alias handles for
one native pipeline therefore require separate reservations.
`max_commands_per_list` bounds the maximum generated count accepted by one
call, and `preprocess_buffer_count` bounds simultaneously retained calls for
that key across incomplete command lists. The backend asks the driver for the
exact size, alignment, and memory-type requirements for the pipeline, layout,
and count before allocating. Reserving the same key replaces it; other keys in
the context remain live. A live reservation retains its pipeline; release every
key before destroying the pipeline.

Reservation replacement or `release_generated_scratch` returns
`RESOURCE_IN_USE` while the calling context has recording, executable, or
submitted work. Descriptors must set every field and remain within
`DeviceCaps.max_generated_work_count`, or reservation returns
`INVALID_ARGUMENT`. Releasing a key that is not reserved returns
`INVALID_RESOURCE_STATE`. Generated recording returns deterministic
`GENERATED_SCRATCH_EXHAUSTED` when the count or compatible-buffer supply is
insufficient. A failed recording call preserves the command list for retry or
discard.

Generated record spans are 8-byte aligned and must hold the declared maximum
count. The count span is a 4-byte-aligned GPU-readable `uint`. Both spans must
come from live allocations admitted to the recording queue. The GPU-written
count may be zero and must not exceed either the command maximum or
`DeviceCaps.max_generated_work_count`. A zero command maximum records no
native work. Generated indexed draws accept only `IndexType.U16` or
`IndexType.U32`; GPU-written index bounds remain subject to device robustness.
The active pipeline supplies execution state and is not an API argument.
Root-reachable allocations and resources are caller-owned, are not tracked from
GPU-written addresses, and must remain live until the covering completion point
finishes.

Argument spans must support indirect-command reads, be 4-byte aligned, and
contain `draw_count` (or `max_draw_count`) times the tight argument size. One
vertex/fragment root pair applies to every draw in a
multi-draw; per-draw variation indexes a table through `gl_DrawID` (see
`docs/shader_abi.md`). Direct draw counts and GPU-written count values may be
zero and must not exceed `DeviceCaps.max_draw_indirect_count`. In the count
variant, `max_draw_count` may exceed that limit, but the argument span must
hold `max_draw_count` entries; execution uses the smaller of `max_draw_count`
and the GPU-written count. Argument byte calculations are checked for
overflow; violations fault `INVALID_ARGUMENT` before recording.

Each GPU-written `DispatchIndirectCommand` component must not exceed the
corresponding `DeviceCaps.max_compute_work_group_count` component. Ordering
between argument writes and indirect consumption is the caller's barrier with
`hazards.draw_arguments` set.

The count variant requires `DeviceCaps.draw_indirect_count` and faults
`UNSUPPORTED_FEATURE` without it.

Index bounds in indirect indexed draws are **not validated**: `index_count`
lives in GPU-written memory, so out-of-bounds indices follow device
robustness behavior.

### Transfer

```text
BufferCopyDesc
    GpuSpan src
    GpuSpan dst

cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) -> void?
cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) -> void?
cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) -> void?
cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) -> void?
```

Copy spans must be nonzero, equal in size, and non-overlapping.
`cmd_fill_buffer` fills the exact destination span; its byte offset and
size must be 4-byte aligned. There is no zero-size shorthand. Callers provide
the surrounding barriers.

### Direct readback

Use a caller-owned `CPU_READ` allocation as the copy destination:

```text
allocate_memory(CPU_READ)
get_allocation_span
cmd_copy_buffer or cmd_copy_texture_to_buffer
cmd_barrier(TRANSFER -> HOST)
submit
poll_completion or wait_completion
invalidate_mapped_span
get_span_mapping
```

The caller owns the allocation, barriers, completion point, and mapped data.
No application work boundary or readback-specific token is required.

### Barriers

```text
StageMask
    all
    host
    transfer
    compute
    vertex_shader
    fragment_shader
    color_output
    depth_output
    present

HazardFlags
    draw_arguments
    descriptors
    depth_stencil

Barrier
    StageMask before
    StageMask after
    HazardFlags hazards

cmd_barrier(CommandList* commands, Barrier* barrier) -> void?
```

`Barrier` is a global execution and memory dependency. It has no resource
handle, address, range, layout, or queue-family field. Each stage mask must be
nonempty; `all` and `present` are each exclusive. Unknown mask bits, special
hazards combined with presentation, and stages or hazards unsupported by the
recording queue fault `INVALID_ARGUMENT`.

Normal host, transfer, shader, color-output, and depth-output access scopes are
derived from the stage masks. `draw_arguments`, `descriptors`, and
`depth_stencil` opt into special consumer data paths; `descriptors` requires a
shader or `all` consumer stage. The library does not infer barriers. Cross-queue
dependencies use `SubmitDesc.completion_waits`, not `cmd_barrier`.

Texture transitions remain explicit and semantic:

```text
TextureLayout
    UNDEFINED
    TRANSFER_SOURCE
    TRANSFER_DESTINATION
    SAMPLED
    STORAGE
    COLOR_ATTACHMENT
    DEPTH_ATTACHMENT
    PRESENT

TextureAccess
    read
    write

TextureState
    TextureLayout layout
    StageMask stages
    TextureAccess access

TextureBarrier
    TextureHandle texture
    TextureViewDesc view
    TextureState before
    TextureState after

sampled_at(StageMask stages) -> TextureState
storage_at(StageMask stages, TextureAccess access) -> TextureState
texture_transition(TextureHandle texture, TextureState before,
    TextureState after)
    -> TextureBarrier?
texture_view_transition(TextureHandle texture, TextureViewDesc view,
    TextureState before, TextureState after) -> TextureBarrier?
cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) -> void?
```

`before` asserts the caller-established prior state; `after` declares the next
state. Layout, execution stages, and read/write access are independent.
`sampled_at` selects `SAMPLED` with read access, while `storage_at` selects
`STORAGE` and preserves the caller's access. Both constructors only compose a
value and insert no synchronization.

The semantic matrix is exact:

| Layout | Public stages | Access | Required texture/queue |
|---|---|---|---|
| `UNDEFINED` | empty | empty | source only |
| `TRANSFER_SOURCE` | transfer, or exclusive `all` | read | `transfer_src`; transfer-capable queue |
| `TRANSFER_DESTINATION` | transfer, or exclusive `all` | write | `transfer_dst`; transfer-capable queue |
| `SAMPLED` | nonempty vertex/fragment/compute combination, or exclusive `all` | read | `sampled`; shader-capable queue |
| `STORAGE` | nonempty vertex/fragment/compute combination, or exclusive `all` | read, write, or both | `storage`; shader-capable queue |
| `COLOR_ATTACHMENT` | color output, or exclusive `all` | read, write, or both | `color_attach`; non-depth format; graphics queue |
| `DEPTH_ATTACHMENT` | depth output, or exclusive `all` | read, write, or both | `depth_attach`; depth format; graphics queue |
| `PRESENT` | empty | empty | swapchain-owned non-depth texture; graphics queue |

Unknown bits and layouts fault `INVALID_ARGUMENT`. Texture states cannot name
the global-only `host` or `present` stage bits. Same-state transitions remain
valid explicit memory dependencies. Sampled depth/stencil textures lower to the
appropriate read-only depth/stencil layout.

A zero `view` selects the full texture. Zero mip or layer counts select the
remaining range from their respective base. Format reinterpretation and
out-of-range subresources fault `INVALID_ARGUMENT`.

Recording resolves the texture handle once, validates recording access,
normalizes the range once, validates and lowers both states once, assembles one
native barrier, and emits it once. Validation covers queue access, semantic
values, stage support, immutable texture usage and format, presentation
ownership, and the selected subresource range. Rejection emits nothing and
rolls back any retained command reference.

The backend does not infer, track, compare, or repair prior state. A wrong
`before` declaration is a caller synchronization error; applications own their
layout history, including history for separate subresource ranges.

At the presentation boundary, `AcquiredImage.prior_state` is directly usable as
the first transition's `before` value. The fixed public `PRESENT` state has
empty stages and access because the presentation engine is external to the
pipeline. The Vulkan backend preserves the validated WSI policy by lowering
the presentation-facing side to color-attachment-output with no access; the
concrete rendering side still comes from the caller's state.

`SubmitDesc.readiness_before` names the destination stages of the first
command that consumes the acquired image. When the first recorded transition
leaves `PRESENT`, the mask must also include `color_output`: the transition's
fixed color-attachment-output source scope is ordered against the acquire wait
only through that stage, so a mask without it leaves the layout change
unordered relative to acquisition.

`UNDEFINED` supplies no source dependency and discards prior contents. Use it
only for first use or after earlier access has been ordered separately.

No command helper should silently insert barriers for a later use.

### Structured debug messages, labels, and leak reporting

`RuntimeDesc.debug_callback` optionally receives `DebugMessage` values produced
by the selected contract policy, backend failures, native Vulkan
validation/performance routing, and eligible resource-lifetime scans during
device teardown. Callback presence is observational: it enables no library
checks, tracking, layers, or debug names and never changes the value, fault,
rollback, native emission, or resource state of the originating operation. The
flat public fault remains authoritative; `has_fault` says whether
`public_fault` accompanies the message.

```text
DebugMessage
    DebugMessageSeverity severity
    DebugMessageCategory category
    ZString operation
    bool has_fault
    fault public_fault
    DebugResourceRef resource       (public index/generation only when known)
    ZString rejected_field
    ZString invariant
    ZString backend_text
    ZString validation_id_name
    int validation_id_number
```

Delivery is synchronous and allocation-free. The message and every referenced
string are borrowed only until the callback returns; copy anything that must
outlive the call. `debug_user_data` configured on a runtime or device must remain
valid from the matching create entry through destroy return, including teardown
messages. No callback occurs afterward.

Public/backend messages run on the calling thread. Vulkan messages may arrive
concurrently on arbitrary application or driver threads. The callback must be
nonblocking and must not call gpu.c3l: delivery can occur while internal locks
are held, and callbacks are neither serialized nor reentrant. See
`docs/threading.md`.

Validation IDs, numeric IDs, backend text, and the first useful object name
are forwarded when Vulkan provides them. Native Vulkan handles and types are
never exposed. For validation-category messages, `validation_id_number` is the
numeric validation ID; for backend-result messages, it carries the raw signed
native result code.

Stored public debug names remain available in every policy;
`enable_debug_names` controls best-effort Vulkan object naming independently.

`TRUSTED` still reports backend/device failures and callback-requested teardown
leaks, but does not promise detailed misuse diagnostics. `OBJECT_BOUNDARIES`
reports covered stale/foreign public identities. `FULL` reports detailed
rejected fields and invariants for command misuse. These library diagnostics do
not require Vulkan validation layers. Backend failures preserve the public
fault and report the public operation name, such as `submit` or
`wait_completion`.

Allocation diagnostics cover invalid classes, sizes, alignments, mapping and
address capability mismatches, visibility failures, stale spans, and detected
resource references. `GpuAllocation` diagnostics carry the public index and
generation.

```text
cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) -> void?
cmd_end_label(CommandList* commands) -> void?
```

Labels group work for capture tools; they are valid while recording,
including inside render passes, and silently succeed when debug-utils is
absent. Balance is the caller's responsibility.

Accepted destruction scans backend state under `OBJECT_BOUNDARIES` or `FULL`,
or whenever a callback is configured. `TRUSTED` without a callback may remain
silent. Normal live children are rejected before this scan; diagnostics are a
safety net for internal, partial-initialization, and device-loss leftovers.
Callback messages use `WARNING`/`resource_lifetime` with operation
`destroy_device`; enabled contract reporting without a callback uses stderr.
Layer and debug-name settings do not control the scan. Debug names are stored
as truncating 63-byte copies.

## 10. Swapchain API

`create_swapchain` borrows a runtime-owned `Surface`; the surface must outlive
the swapchain.

```text
PresentMode: FIFO, IMMEDIATE, MAILBOX

SwapchainDesc
    uint width
    uint height
    Format preferred_format
    PresentMode present_mode
    uint image_count
    bool srgb
    ZString debug_name

SwapchainInfo
    Format format
    uint width
    uint height
    uint image_count
    PresentMode present_mode
    bool dormant

SwapchainReadiness
    opaque one-shot value

AcquiredImage
    TextureHandle texture
    AttachmentViewHandle attachment_view
    SwapchainReadiness readiness
    uint index
    bool suboptimal
    TextureState prior_state

create_swapchain(Device*, Surface*, SwapchainDesc*) -> SwapchainHandle?
destroy_swapchain(Device*, SwapchainHandle) -> void?
resize_swapchain(Device*, SwapchainHandle, uint width, uint height) -> void?
get_swapchain_info(Device*, SwapchainHandle) -> SwapchainInfo?
acquire_next_image(Device*, SwapchainHandle) -> AcquiredImage?
present(Device*, AcquiredImage*, CompletionPoint render_completion) -> void?
get_present_mode_support(Device*, SwapchainHandle) -> PresentModeSupport?
```

The rendering submission passes `acquired.readiness` in `SubmitDesc` and names
its first consuming stages with `readiness_before`. Readiness requires a
graphics queue and a nonempty supported device stage; absent readiness requires
a zero `readiness_before`. Successful submission consumes readiness and returns
the only completion point accepted by `present` for the acquisition. Validation
and retryable native failure preserve readiness; device loss is terminal.
Replays, stale acquisitions, foreign devices, non-graphics queues, and unrelated
completion points fault before native mutation.

```c3
gpu::AcquiredImage acquired = gpu::acquire_next_image(&device, swapchain)!;
// Use acquired.attachment_view as the color target; it is borrowed.
gpu::SubmitDesc submit = {
    .command_lists    = lists[..],
    .readiness        = acquired.readiness,
    .readiness_before = { .color_output },
};
gpu::CompletionPoint rendered = gpu::submit(graphics, &submit)!;
gpu::present(&device, &acquired, rendered)!;
```

A successful or enqueued present consumes `acquired`; validation failure,
private-fence `WAIT_TIMEOUT`, and native host/device allocation failure leave it
retryable. The caller explicitly transitions the acquired texture to `PRESENT`
before submission. Native presentation resource retirement is tracked with
private fences. Presentation requires `VK_KHR_get_surface_capabilities2` and
`VK_EXT_surface_maintenance1` on the instance, plus
`VK_EXT_swapchain_maintenance1` on the device.

Destroy and resize are immediate. They never wait for a queue or submit hidden
work. A pending acquisition returns `INVALID_RESOURCE_STATE`; unfinished
presentation or live command/view references return `RESOURCE_IN_USE`. Every
fault preserves the swapchain for retry. Present an acquired image, discard or
finish explicit references, and poll/retry after presentation can complete.

`WAIT_TIMEOUT` from acquire leaves the swapchain unchanged. It can mean the
native acquire timed out or reported not ready, or that both private acquire
semaphore slots remain retired behind incomplete render completions.
`WAIT_TIMEOUT` from present means the per-image presentation fence is not yet
reusable and preserves the acquired image. Out-of-date requires resize; surface
loss requires a new surface and swapchain. A suboptimal image is valid and may
be presented before resizing. Unsupported requested present modes fall back to
FIFO; query `get_present_mode_support` to choose explicitly.

`get_swapchain_info` returns the selected format, extent, image count, present
mode, and dormant state. Re-query after resize and rebuild format-dependent
pipelines when the format changes. A dormant swapchain reports `UNDEFINED`,
zero extent/count, and FIFO until resize succeeds.

`AcquiredImage.prior_state` is the exact empty `UNDEFINED` state for a newly
wrapped image and the exact empty `PRESENT` state after the normal presentation
cycle. `AcquiredImage.attachment_view` is the
borrowed, swapchain-owned color target for the texture; callers must not destroy
it. Resize stales both borrowed handles. Destroy descriptors that reference
swapchain textures, and discard or complete commands that name either handle,
before resize or destruction.

SDL integration belongs in samples or an optional helper module.

## 11. Example: root-pointer compute

```c3
import gpu;

struct RootArgs {
    gpu::GpuAddress input;
    gpu::GpuAddress output;
    uint            count;
    uint            _pad0;
    uint            _pad1;
    uint            _pad2;
}

fn void? run_compute(gpu::Device* device, gpu::PipelineHandle pipeline) {
    gpu::AllocationDesc storage_desc = {
        .size         = 8192,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .compute },
        .debug_name   = "compute_storage",
    };
    gpu::GpuAllocation storage =
        gpu::allocate_memory(device, &storage_desc)!;
    defer (void)gpu::free_allocation(device, &storage);
    gpu::GpuSpan storage_span =
        gpu::get_allocation_span(device, storage)!;

    gpu::AllocationDesc root_desc = {
        .size         = RootArgs::size,
        .alignment    = RootArgs::alignment,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .compute },
        .debug_name   = "compute_root",
    };
    gpu::GpuAllocation root_allocation =
        gpu::allocate_memory(device, &root_desc)!;
    defer (void)gpu::free_allocation(device, &root_allocation);
    gpu::GpuSpan root_span =
        gpu::get_allocation_span(device, root_allocation)!;
    RootArgs* root =
        (RootArgs*)gpu::get_span_mapping(device, root_span)!.ptr;
    root.input = gpu::get_span_address(
        device,
        storage_span.checked_subspan(0, 4096)!,
    )!;
    root.output = gpu::get_span_address(
        device,
        storage_span.checked_subspan(4096, 4096)!,
    )!;
    root.count = 1024;
    gpu::flush_mapped_span(device, root_span)!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.COMPUTE)!;
    gpu::CommandList commands = gpu::begin_commands(queue)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::cmd_bind_pipeline(&commands, pipeline)!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     gpu::get_span_address(device, root_span)!,
        groups:   { 16, 1, 1 },
    )!;
    gpu::ExecutableCommandList executable =
        gpu::end_commands(&commands)!;
    defer (void)gpu::discard_executable_commands(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;
}
```

The wait covers the last use of both allocations, so their deferred frees are
safe. A non-blocking caller retains the allocations and completion point until
`poll_completion` succeeds.

## 12. API acceptance criteria

The public API is acceptable when:

```text
no public signature exposes vk::, vma::, or sdl:: types
all fallible operations return optionals/faults
individually owned resources have explicit destruction and caller-managed completion lifetimes
generic GPU data uses allocations, spans, and addresses without a public buffer object
root-pointer compute can be written without descriptor-set concepts
texture sampling can be written with TextureIndex and SamplerIndex
barriers are explicit and expressive enough for all samples
submission dependencies use reusable CompletionPoint values
headless tests do not depend on SDL3
windowed samples depend on sdl3 only in sample project files
```
