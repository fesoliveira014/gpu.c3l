# gpu.c3l Public API

This is a curated guide to the public API and its idioms. The generated
`api-reference` CI artifact covers every public symbol.
Source doc comments define the contract.

## 1. Public module

All public API lives in:

```c3
module gpu;
```

The source-of-truth split is explicit: `gpu/gpu.c3i` contains all public
non-callable declarations, while `gpu/gpu.c3` contains all public callable
implementations with their doc contracts, attributes, and default arguments.
Backend-independent implementation details live in private `gpu::internal`;
Vulkan implementation details live in private `gpu::internal::vk`.

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

Each surface module keeps its native handle typedefs in its local `surface.c3i`
and `create_surface` in the adjacent `surface.c3`. These platform modules are
the only public API outside the root `gpu` source pair.

SDL3 may supply the native handles in a consumer, but core declarations do not
mention `sdl::Window`, `vk::Device`, or `vma::Allocation`.

## 2. Naming rules

Use:

```text
create_device
create_command_allocator
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
ContractValidation
    TRUSTED
    FULL

RuntimeDesc
    ContractValidation contract_validation
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
    QueueRoles available

AdapterLimits
    uint max_texture_dimension_2d
    uint max_texture_dimension_3d
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

### Validation contract

Public contracts use four cause categories:

| Category | Meaning |
|---|---|
| **Preconditions** | Facts valid callers guarantee under every policy. |
| **Always checked** | The mandatory public and command-token identity, authoritative-phase, host-safety, safe-lowering, integrity, cold-path, lifecycle, and runtime-result floor. Violations return deterministic documented faults under every policy. |
| **FULL diagnostics** | Detailed semantic misuse diagnosed by the FULL command table only when `contract_validation = FULL`. |
| **Runtime failures** | Capacity, allocation, timeout, device/surface loss, unsupported capability, and backend failures that valid callers may encounter under every policy. |

A precondition can also be always checked: the first category states the
caller's obligation, while the others state library behavior. Callable
comments use these exact headings and omit categories that add no
operation-specific information.

Runtime controls are independent:

| Control | Zero/default | Effect |
|---|---|---|
| `contract_validation` | `TRUSTED` | Selects library contract behavior. `TRUSTED` uses trusted command recording with the always-checked floor and no command reference storage. `FULL` adds detailed argument, usage, layout, queue, pipeline-kind, render-compatibility, and state diagnostics; retains explicitly named command resources; and emits teardown leak diagnostics. |
| `enable_vulkan_validation` | `false` | Requests `VK_LAYER_KHRONOS_validation`. It does not select library checks or lifetime tracking. |
| `enable_debug_names` | `false` | Requests best-effort native object naming through debug utils. It enables no checks, tracking, or layers. |
| `debug_callback` | null | Selects structured delivery for diagnostics already produced by the contract policy or Vulkan. It does not enable checks, tracking, layers, or names. |

A zero-initialized `RuntimeDesc` therefore selects `TRUSTED`, Vulkan validation
layers off, debug names off, and no callback. Every contract mode retains the
always-checked floor: host pointer and slice safety
before reads, integer-overflow and backing-range protection needed for safe
lowering, authoritative command phase and internal table integrity, public
device ownership, Vulkan result/device-loss handling, and transactional
creation rollback. Command recording always compares the direct token
generation and authoritative phase before backend mutation. Semantic misuse
outside that floor remains a caller precondition under `TRUSTED`, which does
not promise `FULL` command diagnostics.

Use the helper for the former all-enabled development configuration:

```c3
gpu::RuntimeDesc runtime_desc = gpu::full_validation_runtime_desc();
runtime_desc.application_name = "my_app";
```

`FULL` does not require Vulkan SDK layers. Set `contract_validation = FULL` and
leave `enable_vulkan_validation = false` when detailed library diagnostics are
needed on a machine without the Khronos layer. For migration only, a former
`enable_validation = true` initializer maps to `FULL` and Vulkan validation on
(or the helper); `false` or omission maps to `TRUSTED` and Vulkan validation
off. FULL always includes command resource lifetime tracking; TRUSTED never
allocates reference storage. The retired `OBJECT_BOUNDARIES` value and
`track_resource_lifetimes` field intentionally fail to compile.

Creating a runtime is the first operation that may initialize Vulkan discovery
behind the backend-neutral API. Enumeration returns an allocation-free view:

```c3
gpu::RuntimeDesc runtime_desc = {};
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
accepts swapchains only for the exact surface named in its descriptor and
reports that capability through `DeviceCaps.presentation_enabled`.
Presentation may use a private queue distinct from graphics. Destroying a
surface with a live swapchain returns `RESOURCE_IN_USE`; destroy surfaces
before their runtime. Destroying a descriptor surface with no live swapchain
succeeds. Its presentation device remains bound to that stale token, so future
`create_swapchain` calls return `INVALID_HANDLE`.

### Device descriptors and creation

Device creation takes one exact borrowed adapter and an optional plain
descriptor. The descriptor contains only per-device semantic requirements:
presentation, queue topology, and the optional sparse-texture capability. The
library's required device baseline is implicit. Support detection is optional
application functionality for adapter selection; it is read-only, enables
nothing, and does not mutate the descriptor.
The unmet-requirement label is borrowed static text and names GPU semantics
rather than backend features.

```text
DeviceDesc
    Surface surface
    QueueRequest queues
    bool enable_sparse_textures

DeviceSupport
    bool supported
    String unmet_requirement     (borrowed static semantic label)

QueueRequest
    QueueRoles required
    QueueRoles distinct

supports_device_desc(Adapter*, DeviceDesc* = null) -> DeviceSupport?
create_device(Adapter*, DeviceDesc* = null)        -> Device?
```

An omitted descriptor, null, and a zero-initialized `DeviceDesc` are
equivalent. They select a non-presenting device with one graphics, compute, and
transfer queue and sparse textures disabled; one native queue may satisfy
several roles. The omitted form is the canonical minimal call:

```c3
gpu::Device device = gpu::create_device(&adapter)!;
```

Store a live same-runtime surface directly to request presentation:

```c3
gpu::DeviceDesc device_desc = {
    .surface = surface,
};
gpu::Device device = gpu::create_device(&adapter, &device_desc)!;
```

`DeviceDesc.queues = {}` selects the default topology. Any nonzero
`QueueRequest` is explicit. At least one role must be required. A role marked
`distinct` must also be required and may not alias another requested role.
Presentation requires graphics. Unknown role bits, invalid required/distinct
combinations, and a presentation descriptor without graphics fault
`INVALID_ARGUMENT`.

Set `enable_sparse_textures` to require core sparse binding, at least one of 2D
or 3D sparse residency, and at least one selected semantic role whose native
family can submit sparse bindings. This opt-in does not add a sparse queue kind
or select a hidden sparse-only family. An unavailable valid request is reported
as unsupported.

Descriptor input is borrowed for the call and copied during normalization.
Zero surface means no presentation. A nonzero surface must resolve live and
belong to the adapter's runtime; stale, malformed, or foreign tokens fault
`INVALID_HANDLE`. A valid surface or queue topology unavailable on the adapter
is unsupported rather than malformed.

A live adapter-created device retains its runtime and reuses the runtime-owned
backend instance. Device defaults are copied by `create_runtime` and inherited
by every device created from that runtime. A presentation device records but
does not retain its surface token; only a live swapchain blocks surface
destruction. Destroy each device before its runtime.

```text
DeviceCaps
    bool presentation_enabled
    bool draw_indirect_count
    bool generated_work
    bool async_compute
    QueueRoles queues
    bool line_polygon_mode
    TimestampCaps timestamps
    SparseTextureCaps sparse_textures
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

TimestampCaps
    QueueRoles queues
    uint graphics_valid_bits
    uint compute_valid_bits
    uint transfer_valid_bits
    float period_ns

SparseTextureCaps
    bool image_2d
    bool image_3d
    bool nonresident_strict
    QueueRoles binding_queues

Device                           (slot | generation | reserved)
get_device_caps(Device*)         -> DeviceCaps?
```

Every published device implements the same mandatory semantic baseline. The
minimum supported device profile is intentionally Vulkan 1.3 plus
`VK_EXT_extended_dynamic_state3` and
`dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. The baseline also
requires the extension's color blend-enable, color blend-equation, and color
write-mask features, together with independent per-target blending and
depth-bias clamp. Creation returns `UNSUPPORTED_FEATURE` when an adapter cannot
provide every requirement. `supports_device_desc` may be used before creation
when an application wants to select among adapters. `generated_work` is true
only when the created device enables
GPU-written root and work records for graphics and compute. A supported device
reports a nonzero `max_generated_work_count`; an unsupported device reports
false and zero. Heap and generated-work implementation mechanisms remain
private.

Runtime heap capacities are exact semantic device defaults, not clampable upper
bounds. Device creation fails rather than clamping when the selected adapter
cannot satisfy them. On success,
`DeviceCaps.texture_heap_capacity` and `DeviceCaps.sampler_heap_capacity`
report the exact capacities of the created shader-visible heaps. A runtime capacity
above the library hard ceiling is malformed and faults `INVALID_ARGUMENT`; a
valid capacity unavailable on the selected adapter faults `UNSUPPORTED_FEATURE`.

`DeviceCaps.timestamps` reports only selected logical roles that can execute the
complete reset/write/resolve workflow. Each role has its own valid-bit width;
unsupported roles have width zero. An aliased transfer role is available when
its native queue supports graphics or compute and reports nonzero timestamp
bits. A dedicated transfer-only queue is excluded even when it reports
timestamp bits. `period_ns` is device-wide and is zero only when no selected
role supports the workflow.

`DeviceCaps.sparse_textures` reports enabled behavior, not ambient hardware
support. It is entirely zero unless sparse textures were requested. On a
successful sparse device, `image_2d` and `image_3d` name the enabled residency
dimensions. When `nonresident_strict` is true, nonresident reads are guaranteed
to return zero and writes are discarded; when false, that behavior is not
guaranteed. `binding_queues` names every selected semantic role backed by a
sparse-binding-capable family. Sparse resources and binding operations are not
part of this capability-only surface.

Creation:

```text
supports_device_desc(Adapter*, DeviceDesc* = null) -> DeviceSupport?
create_device(Adapter*, DeviceDesc* = null)        -> Device?
destroy_device(Device*) -> void?
```

Malformed descriptors fault before backend mutation. A valid unsupported
descriptor returns `supported = false` with the first unmet semantic label;
passing it to `create_device` faults `UNSUPPORTED_FEATURE` without selecting
another adapter. Support queries and creation share descriptor normalization
and validation semantics. Failed creation publishes no device and changes no
runtime or surface dependency count.

Multiple live devices may coexist. `Device` is a compact slot and generation
token. Most public operations take a short-lived atomic pin before reading
the typed private Vulkan state. `begin_commands` transfers its pin to the
returned command token; recording calls reach the token's stable authoritative
record and acquire no additional pin or device-registry operation.

`destroy_device` never waits. Live resources, command allocators, command lists,
swapchains and descriptors return `RESOURCE_IN_USE`.
Active operations, incomplete queue work, or
a closing slot return retryable `DEVICE_BUSY`. Every failed attempt preserves
the token, generation, and typed private state. Success increments the
generation and invalidates the passed token. A lost device bypasses child and
progress checks after operation pins retire. Lost command tokens remain
discardable so their lifetime pins cannot strand the device.

Queue tokens, command tokens, resource handles, descriptor
indices, GPU addresses/spans, and completion points are scoped to their
owning device. Private table resolution rejects foreign handle owners before
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
CommandAllocatorHandle
TimestampPoolHandle
```

Owning tokens have zero-valued invalid constants such as
`GPU_ALLOCATION_INVALID`, `TEXTURE_HANDLE_INVALID`, `TEXTURE_VIEW_INVALID`,
`COMMAND_ALLOCATOR_HANDLE_INVALID`, `TIMESTAMP_POOL_HANDLE_INVALID`, and their
peers.
`token.is_valid()` checks the owner and generation; operations also validate
the local slot generation. Public code should not inspect or construct the
representation.

Handles, `TextureView`, `Queue`, `CommandAllocator`, `GpuSpan`, command tokens, and
synchronization values are runtime-only owner-bearing tokens scoped to one
device. `TextureIndex`, `SamplerIndex`, and `GpuAddress` are raw device-local
shader values without owner or generation metadata. Do not persist, serialize,
reconstruct, or pass either category across device or process lifetimes. Compare
`Queue` values as wholes and use `get_queue_info` for inspection; do not
construct or mutate queue fields. `CommandList` embeds a library-owned typed
pointer to its address-stable command record, a reuse generation, and a packed
static device-slot identity; it does not borrow caller variable storage.
Applications must not inspect, construct, mutate, persist, or serialize this
representation.

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
recovered when a public operation resolves the span. Direct, indirect, and
generated compute and graphics commands forward every `GpuAddress` unchanged,
independent of validation policy. Zero is valid. A shader using zero as an
absent or sentinel root must branch before dereferencing it unless the
application deliberately relies on defined device robustness behavior.

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

Public operations use C3 optionals/faults. `faultdef` declares a flat list of globally-unique fault values (there is no braced/named fault group in C3 0.8.0); these live in `module gpu` and are referenced as `gpu::INVALID_HANDLE`, raised with the `~` suffix. `gpu/gpu.c3i` documents each fault at its definition; the table below maps them to the operations that raise them.

Descriptor, configuration, barrier, graphics-state, viewport, scissor, and
label pointers must be non-null unless the API explicitly documents null as a
value (such as `TextureViewDesc* desc`). Required host-pointer safety is always
checked before a read. Null or stale owner-token pointers fault
`INVALID_HANDLE`.

The category column classifies each documented **cause**, not the whole fault
value. In particular, `INVALID_ARGUMENT` and `COMMAND_RECORDING_ERROR` have
both always-checked and `FULL`-only causes.

| Fault | Cause category | Fired by | Typical cause |
|---|---|---|---|
| `UNSUPPORTED_BACKEND` | Runtime failures | `create_runtime` | the Vulkan loader, driver, or required backend initialization path is unavailable |
| `UNSUPPORTED_FEATURE` | Runtime failures | device creation, `create_runtime`, `create_texture`, `create_dedicated_texture`, `create_texture_view`, `create_texture_views`, `create_timestamp_pool`, `create_swapchain`, `create_graphics_pipeline`, `intern_sampler`, generated draw/dispatch recording, timestamp recording, indexed-indirect-count execution | validation layers not installed; presentation was not requested or is unsupported for the adapter and surface; missing optional or required device feature; an enabled sparse request lacks a residency dimension or selected binding queue; no selected role supports the timestamp workflow; the selected adapter cannot provide the runtime's semantic heap capacities; unsupported image format or usage; adapter rejects a valid texture descriptor |
| `INVALID_ARGUMENT` | Always checked / FULL diagnostics | runtime adapter indexing; device descriptor validation; any create/export, including `create_command_allocator`; `allocate_memory`; `GpuSpan.checked_subspan`; get/mapping/visibility operations; `get_queue`; `submit`; `present`; `cmd_*`; `render_geometry_state`; pipeline creates; transitions; descriptor publication; sampler interning; generated-scratch reservation | always checked for required pointer/slice safety, safe ranges and integer lowering, cold-path configuration, and fixed API limits; `FULL` additionally diagnoses command enum, usage, layout, queue, capability, render-compatibility, and dynamic-state misuse |
| `INVALID_HANDLE` | Always checked | runtime and adapter queries; destruction; device/queue/completion queries; allocation info/span/mapping/address/visibility operations; any resource-handle-taking call; `cmd_*`; command lifecycle; `submit` | zero, destroyed, stale, consumed, malformed, or foreign runtime, adapter, device, queue, completion point, allocation, span, or resource handle; or a zero, stale, consumed, or wrong-phase valid-origin direct command token; submit also rejects a token recorded for another device |
| `INVALID_RESOURCE_STATE` | Always checked lifecycle/safe snapshot / Runtime failures | command execution; surface and swapchain creation/lifecycle; `destroy_attachment_view`; `release_generated_scratch` | an authoritative lifecycle transition is invalid, trusted-table command execution has no usable bound-pipeline snapshot, a borrowed view was passed for destruction, a generated-scratch key is not reserved, or Vulkan reports that the native window is already in use |
| `OUT_OF_HOST_MEMORY` | Runtime failures | creates; mapped visibility | driver or backend cache host-allocation failure |
| `OUT_OF_DEVICE_MEMORY` | Runtime failures | allocator, allocation, and texture creates; mapped visibility | backend device-memory exhaustion |
| `DEVICE_LOST` | Runtime failures | any Vulkan-backed operation | Vulkan returned `VK_ERROR_DEVICE_LOST`; the affected device rejects later operations while peer devices remain usable |
| `DEVICE_BUSY` | Runtime failures / Always checked lifecycle | public device operations; `read_timestamps`; `begin_commands`; `submit`; `destroy_device` | closing state, unavailable timestamp results, bounded pin or allocator-unit contention, timeline headroom, active operations, or incomplete queue work; retry, and ignore timestamp output after a not-ready read |
| `RESOURCE_IN_USE` | Always checked lifecycle | resource or swapchain destruction/resize, `free_allocation`, allocator destruction, generated-scratch reservation/release, `destroy_device`, `destroy_runtime`, `destroy_surface` | a detected live owner, dependent, command reference, acquisition, presentation, child, or non-quiescent allocator prevents immediate mutation or destruction |
| `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` | Runtime failures | generated-scratch reservation; generated command recording; tracked command recording | fixed reference, generated-index, reservation-table, or preprocess-byte storage cannot represent valid requested work |
| `SLOT_TABLE_FULL` | Runtime failures | runtime, device, allocator, allocation, and resource creates; `intern_sampler`; `acquire_next_image`; queue submission | a registry or handle table is at capacity, or a queue completion or swapchain acquisition sequence is exhausted |
| `GENERATED_SCRATCH_EXHAUSTED` | Runtime failures | generated draw/dispatch recording | no matching reserved preprocess slot is free for valid retained work |
| `DESCRIPTOR_HEAP_FULL` | Runtime failures | descriptor pool creation/allocation, `create_texture_view`, `create_texture_views`, `intern_sampler` | descriptor-pool exhaustion or fragmentation, or capacity below the live descriptor count |
| `PIPELINE_CREATE_FAILED` | Runtime failures | pipeline creates | the driver rejected the state combination, shader, or compilation |
| `SHADER_INVALID` | Always checked cold path / Runtime failures | pipeline creates | malformed or ABI-incompatible SPIR-V, a selected-entry role mismatch, or backend shader-module rejection |
| `SURFACE_LOST` | Runtime failures | surface creation/query/enumeration, swapchain create/resize, acquire, present | the native window or surface was destroyed or became unavailable |
| `SWAPCHAIN_OUT_OF_DATE` | Runtime failures | `create_swapchain`, `resize_swapchain`, `acquire_next_image`, `present` | the swapchain no longer matches the surface |
| `COMMAND_RECORDING_ERROR` | Always checked / FULL diagnostics | `cmd_*`, `end_commands`, `discard_commands`, `discard_executable_commands`, `submit` | always checked for authoritative command/pass phase, duplicate claims, an already-submitting token, and prerequisites needed for safe recording; `FULL` additionally diagnoses a missing logical bind, detailed pipeline/pass compatibility, and complete graphics state |
| `WAIT_TIMEOUT` | Runtime failures | `wait_completion`, `acquire_next_image`, `present` | bounded wait, transient image unavailability, or a private present fence not yet reusable |
| `BACKEND_ERROR` | Runtime failures | any Vulkan-backed operation | unclassified or internal native failure; inspect backend diagnostics; does not imply device loss |

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

The core exposes primitives, not transfer policy. For long-lived or
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
    uint depth
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

`TextureDesc.depth == 0` selects a 2D image with native depth one. A positive
value selects a 3D image with that native depth, so `depth = 1` is a genuine 3D
image. 3D textures require one normalized array layer, `SampleCount.ONE`, and
sampled, storage, and/or transfer usage; color and depth attachments are
rejected. Their mip count is bounded by width, height, and depth.
`AdapterLimits.max_texture_dimension_3d` is a quick extent ceiling, while
`supports_texture_desc` is the exact creatability query.

`get_texture_format_support` reports library-creatable 2D support, not every
raw Vulkan capability. Its usage bits are independent; use
`supports_texture_desc` for an exact 3D descriptor or any complete usage
combination. Higher sample-count bits report exact 2D color-attachment or
depth-attachment descriptors, as appropriate for the format. Per-format
usages, sample counts, and linear filtering remain adapter-dependent. Depth
support is D32-only.
`supports_texture_desc` returns false for an empty, unknown, or unavailable
access set.

`supports_texture_desc` checks the exact optimal-tiling format, image type,
three-axis extent, combined usage, normalized mip and layer counts, access
roles, sample count, and image properties without allocating. Multisample
textures are 2D, require one mip, and use attachment-only usage. A false result
caused by malformed input corresponds to `INVALID_ARGUMENT` at creation; a
structurally valid descriptor rejected by the adapter corresponds to
`UNSUPPORTED_FEATURE`. Memory exhaustion can still make creation fail after a
true capability result.

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
mutation. Texture-view publication uses every device's mandatory shader-visible
heap. Distinct subresource views are governed by the device-wide heap capacity,
with no smaller fixed per-texture publication limit. Sampler indices remain
stable until device
destruction.

For 3D textures, a view uses native `TYPE_3D`, requires `base_layer == 0` and
`layer_count` zero or one, and covers the complete depth of every selected mip.
Arbitrary z-slice views and 3D attachment views are not exposed.

`create_texture_views` batch-publishes N views under one lock hold and ends in
one accumulated update to the device-global descriptor set.
`out_views.len` must equal `descs.len` (`INVALID_ARGUMENT` otherwise); an empty
input is a no-op success. A
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
`INVALID_ARGUMENT`. Enabled anisotropy requires a finite `max_anisotropy` in the
inclusive range `[1, DeviceCaps.max_sampler_anisotropy]`. A value above
`DeviceCaps.max_sampler_anisotropy` faults `INVALID_ARGUMENT` and is not implicitly
clamped. A valid anisotropy request on a device that reports zero
support faults `UNSUPPORTED_FEATURE`. When anisotropy or comparison is disabled,
its associated value is ignored and canonicalized to zero; signed zero also has
one identity. Sampler indices and their native objects live until device
destruction and have no individual destroy operation. Repeated interning is idempotent.
`DESCRIPTOR_HEAP_FULL` consumes no table or heap entry.

### Breaking migration

The packed device-request builder has been retired. Migrate its operations as
follows:

| Retired API | Replacement |
| --- | --- |
| `DeviceRequest` and `strict_device_request()` | `DeviceDesc`; omit it or pass null for the default headless description |
| `request_presentation()` | Set `DeviceDesc.surface` |
| `request_queues()` | Set `DeviceDesc.queues` |
| `supports_device_request()` | `supports_device_desc()` |
| `DeviceRequestSupport` | `DeviceSupport` |

Pass the resulting description directly to `create_device`. The pointer-first
baseline is mandatory and implicit, so no strict-capability bit or builder call
is needed.

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

Earlier releases implicitly clamped an enabled `max_anisotropy` to the device
limit. Query `DeviceCaps` and clamp explicitly when that behavior is desired:

```c3
gpu::DeviceCaps caps = gpu::get_device_caps(device)!!;
float requested_anisotropy = 16.0f;
sampler_desc.anisotropy_enable = caps.max_sampler_anisotropy > 0.0f;
sampler_desc.max_anisotropy = requested_anisotropy > caps.max_sampler_anisotropy
    ? caps.max_sampler_anisotropy
    : requested_anisotropy;
```

No placeholder texture-shape or view-format fields are required.

## 8. Shader and pipeline API

### Shader input

```text
ShaderDesc
    char[] spirv
    ZString entry_point
    ZString debug_name
```

Shader compilation can be handled by tools or samples. The core library consumes
borrowed SPIR-V bytes directly through each pipeline descriptor. The enclosing
`shader`, `vertex_shader`, or `fragment_shader` field determines the expected
compute, vertex, or fragment role. A null entry point selects `main`. The
descriptor, bytes, and strings need to remain valid only until the synchronous
pipeline creation call returns; successful creation retains an owned private
identity with no caller pointer. Role, normalized entry point, length, and exact
bytes participate in that identity; `debug_name` does not.

Pipeline creation reflects only the selected entry point.
It permits no push-constant block, or requires one declared block to match the
selected role's generated root ABI exactly. Physical offset, size, complete
member coverage, order, and 64-bit width are fixed. The supported authoring
policy additionally requires flat unsigned integer root members; signed,
aggregate, and physical-reference members remain noncanonical and rejected even
when a particular SPIR-V form is byte-compatible. Reflected names are ignored.
A reflection mismatch returns `SHADER_INVALID` before native shader creation or
pipeline publication, and diagnostic-enabled runtimes identify the incompatible
property. A selected entry whose execution model does not match its enclosing
pipeline field also returns `SHADER_INVALID`. There is no public shader-module
handle or shader-preparation object.

### Compute pipelines

```text
ComputePipelineDesc
    ShaderDesc shader
    ZString debug_name

create_compute_pipeline(Device* device, ComputePipelineDesc* desc) -> PipelineHandle?
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
    BlendState blend
    ColorWriteMask write_mask

ColorState
    ColorTargetState[] targets

DynamicRasterState
    PrimitiveTopology topology
    CullMode cull_mode
    FrontFace front_face
    bool depth_bias_enable
    float depth_bias_constant
    float depth_bias_slope
    float depth_bias_clamp

GraphicsPipelineDesc
    ShaderDesc vertex_shader
    ShaderDesc fragment_shader
    Format[] color_formats
    Format depth_format
    SampleCount sample_count
    PolygonMode polygon_mode
    ZString debug_name

create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) -> PipelineHandle?
destroy_pipeline(Device* device, PipelineHandle pipeline) -> void?
```
`PolygonMode.LINE` is optional. Query `DeviceCaps.line_polygon_mode` before
using it; unsupported LINE creation returns `UNSUPPORTED_FEATURE`.
`PrimitiveTopology.LINES` remains available independently with FILL mode and is
selected through `GraphicsState.raster.topology`.

`color_formats` carries at most `MAX_COLOR_ATTACHMENTS` (8) entries and defines
the pipeline's ordered color-target domain. The matching blend equations and
write masks come from a complete command-time `ColorState` packet. The zero
write mask disables all writes; use `COLOR_WRITE_ALL` for conventional RGBA
writes. Enabled blending is invalid for integer color formats.

Common caller-owned packet constructors are `color_blend_disabled`,
`alpha_blend`, `premultiplied_alpha_blend`, `additive_blend`, and
`uniform_color_state`; the library retains none of their storage.

### Pipeline deduplication

Pipeline creation deduplicates through a descriptor-keyed cache. Every create
returns a fresh handle, but descriptors identical in immutable state (exact
private shader identity, polygon mode, ordered color formats, depth format, and
sample count) alias one backend pipeline underneath. Compute
identity is shader identity because its `RootPush` layout is fixed. Private
hash collisions are resolved by role, normalized entry point, length, and exact
SPIR-V bytes. Topology,
cull/front-face, depth bias, depth test/write/compare, viewport, scissor,
blend equations, and write masks are separate command state.

There is no public batch helper. Callers that need a collection create each
pipeline in a loop. If the collection must be transactional, the caller destroys
the handles created earlier in the loop when a later creation faults. Repeated
identical descriptors still converge on the same backend pipeline through the
device cache.

```c3
fn void? create_graphics_set(
    gpu::Device* device,
    gpu::GraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] pipelines,
) {
    if (pipelines.len != descs.len) return gpu::INVALID_ARGUMENT~;
    usz created;
    foreach (i, &desc : descs) {
        gpu::PipelineHandle? result =
            gpu::create_graphics_pipeline(device, desc);
        if (catch create_fault = result) {
            while (created > 0) {
                created--;
                (void)gpu::destroy_pipeline(device, pipelines[created]);
                pipelines[created] = gpu::PIPELINE_HANDLE_INVALID;
            }
            return create_fault~;
        }
        pipelines[i] = result;
        created++;
    }
}
```

Each successful create must be balanced by exactly one `destroy_pipeline`; the
backend pipeline is destroyed when its last alias is released. Destroying a
handle twice faults `INVALID_HANDLE` and never affects other aliases. Handles
must not be compared to decide whether two pipelines are "the same object" —
distinct handles may or may not share native pipeline state.

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

Command tokens and allocator recording are thread-confined. While an allocator
has any recording list, one thread owns all recording through that allocator.
The owner clears when the last list ends or is discarded, so an
application-synchronized handoff can move the allocator to another worker.
Different allocators may record concurrently. An executable token may be handed
to a submit thread after `end_commands`; a token and its aliases must never be
used concurrently. See `docs/threading.md`.

### Command lifecycle

```text
get_queue(Device* device, QueueKind kind) -> Queue?
get_queue_info(Device* device, Queue queue) -> QueueInfo?
create_command_allocator(
    Device* device,
    Queue queue,
    CommandAllocatorDesc* desc = null,
) -> CommandAllocator?
destroy_command_allocator(CommandAllocator* allocator) -> void?
begin_commands(CommandAllocator* allocator) -> CommandList?
end_commands(CommandList* commands) -> ExecutableCommandList?
discard_commands(CommandList* commands) -> void?
discard_executable_commands(ExecutableCommandList* commands) -> void?
submit(Queue queue, SubmitDesc* desc) -> CompletionPoint?
poll_completion(CompletionPoint point) -> bool?
wait_completion(CompletionPoint point, ulong timeout_ns = TIMEOUT_INFINITE) -> void?

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

QueueRoles
    bool graphics
    bool compute
    bool transfer

QueueInfo
    uint id
    QueueRoles roles

CommandAllocator
    Device device
    Queue queue
    CommandAllocatorHandle handle

CommandAllocatorDesc
    uint command_buffer_capacity
    uint max_resource_references_per_list
    uint max_generated_preprocess_buffers_per_list
    usz generated_preprocess_bytes
    ZString debug_name

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
`TIMEOUT_INFINITE` is `ulong::max` and is shared by explicit waits and
swapchain acquisition.

`CommandList` is a recording token. Successful `end_commands` consumes it and
returns a one-shot `ExecutableCommandList`; failure leaves the recording token
unchanged. Copies alias the same authoritative record. Recording and executable
tokens retain the device while the record is recording or executable.
Successful submission consumes the public executable value but keeps the same
record, allocator unit, device ownership, fixed scratch, and native command
buffer live until ordered completion retirement.

`CommandList.is_valid` and `ExecutableCommandList.is_valid` test only whether a
value differs from its invalid zero/token sentinel. They do not validate live
provenance or phase. Each operation first checks the token's static device-slot
identity, then compares its record generation and checks the authoritative
record phase before native mutation.

`CommandAllocator` is a caller-owned child of one exact selected queue. Create
one before recording and destroy it before its device. Creation allocates one
native command pool for that queue family, every configured command buffer,
and all fixed host bookkeeping. Zero fields select these public defaults:

| Capacity | Default | Maximum | Scaling |
|---|---:|---:|---|
| `command_buffer_capacity` | `DEFAULT_COMMAND_ALLOCATOR_CAPACITY` = 8 | `MAX_COMMAND_ALLOCATOR_CAPACITY` = 4096 | native command buffers, scratch records, and available-index storage |
| `max_resource_references_per_list` | `DEFAULT_COMMAND_REFERENCES_PER_LIST` = 64 | `MAX_COMMAND_REFERENCES_PER_LIST` = 4096 | one fixed sequential reference list per command buffer under FULL; zero storage under TRUSTED; see `docs/performance.md` for the linear-scan cost at each bound |
| `max_generated_preprocess_buffers_per_list` | `DEFAULT_COMMAND_PREPROCESS_PER_LIST` = 4 | `MAX_COMMAND_PREPROCESS_PER_LIST` = 64 | generated-reservation indices per command buffer and reservation-table entries multiplied by command-buffer capacity |

`generated_preprocess_bytes` has no nonzero default: zero disables generated
scratch reservations on that allocator. It is a byte budget for exact
descriptor-driven reservations, not an untyped allocation. Invalid ceilings
or capacity-product overflow return `INVALID_ARGUMENT` before backend work.

Warm `begin_commands`, recording, end, discard, submit, completion retirement,
and reuse allocate no host memory, native command buffer, command pool, VMA
memory, or C3 temporary-pool storage. If every allocator buffer is live,
`begin_commands` returns retryable `DEVICE_BUSY`. Fixed per-list or reservation
capacity exhaustion returns `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`; enlarge a
quiescent allocator rather than waiting.

Tracking-enabled duplicate detection scans the fixed sequential list and uses
full owner/index/generation equality. Each unique entry retains the resource
once and stores its canonical retained counter for direct release. Compound
operations preflight their unique candidates against the remaining capacity
and roll back only entries appended after their transaction checkpoint. No
reference storage grows while recording.

`destroy_command_allocator` never waits, polls completion, or queries a
semaphore. Recording, executable, or incomplete submitted work returns
`RESOURCE_IN_USE` and leaves the value unchanged for retry. After every list is
discarded or retired, destruction releases idle generated reservations and all
native/host allocator storage, consumes the value, and releases its device
child. Even an empty live allocator prevents `destroy_device`.

The command-token representation is a fixed 16-byte public ABI on the supported
linux-x64 and windows-x64 targets. Recording and executable values contain a
library-owned typed pointer to one address-stable authoritative command record,
its reuse generation, and a packed static device-slot identity. A valid
recording call acquire-loads that static slot and verifies its liveness and
generation before dereferencing the record. It then compares the record
generation and validates the authoritative phase. This performs no retained
device-operation borrow or command-table lookup.
Tokens must originate from `begin_commands`; callers must never construct,
mutate, persist, or serialize their fields.

Begin initializes the fixed record while inactive and publishes `RECORDING`
last. End transfers the same record to the executable phase. Failed begin
publishes no token; failed end remains recording; failed submission restores
every claimed record to executable and preserves all input tokens. Discard
invalidates the record before releasing its references, reservations, allocator
unit, and retained ownership.

Command tokens are one-shot. A copied alias must not be used after another
alias is ended, discarded, submitted, or completion-retired. Deterministic
device-slot, record-generation, and phase checks reject that consumed identity
before native mutation, even after device teardown releases the record storage.
Device loss is reported at lifecycle boundaries rather than by every `cmd_*`
call.

`discard_commands` consumes unfinished recording. Use
`discard_executable_commands` for an ended token that will not be submitted.
Both remain available after device loss.

Submission targets an explicit `Queue`. Every executable token in the batch
must have been recorded for that queue. Success consumes all tokens and returns
one queue-owned `CompletionPoint`; each authoritative record stays `SUBMITTED`
and unavailable for reuse until ordered retirement. Retirement first
invalidates the record, then releases tracked references, generated
reservations, the exact allocator unit, and retained ownership before publishing
the retired queue prefix. Failure publishes no point, restores every claimed
record to `EXECUTABLE`, and preserves the tokens for retry or discard. An empty
batch signals the selected queue. Duplicate or non-executable tokens fault
`COMMAND_RECORDING_ERROR`; a token for another queue faults `INVALID_ARGUMENT`.
Discard consumes an unsubmitted token.
One batch may mix executable tokens from different allocators only when every
allocator is bound to the exact target queue. Each completed native command
buffer and scratch index returns to its originating allocator; reset happens on
the next reuse after completion or discard.

`SubmitDesc.completion_waits` accepts published points from the same device.
Each `CompletionWait` names the first destination consumers of the dependency.
`before.indirect` selects indirect draw, indirect dispatch, indirect count, and
implicitly preprocessed generated-command records. The mask must be nonempty;
`host` and `present` are invalid, and `all` is allowed only by itself. Unknown or
unsupported masks fault `INVALID_ARGUMENT`. `indirect` requires a graphics- or
compute-capable destination queue. Cross-queue points become waits on their
queue-owned timelines with the exact requested mask. Published points from the
target queue are validated and then elided because queue order is inherent.
Stale, unpublished, malformed, and foreign-device points fault
`INVALID_HANDLE` before native submission and preserve every command token.

`SubmitDesc.readiness_before` names the first stage that consumes an acquired
swapchain image. It rejects `indirect` under every validation policy; select the
actual image-consuming transfer, shader, color-output, or depth-output stage.
If outstanding queue progress reaches the device's timeline-value-difference
limit, submission faults retryable `DEVICE_BUSY` before reserving a sequence.

`Queue` is a small device-owned identity for a selected queue. Its `device`
field is a copied `Device` token used for exact ownership validation; it does not
borrow caller storage. Each semantic role names at most one selected identity;
identities may satisfy several roles unless the request marks a role `distinct`.
`get_queue` faults `INVALID_HANDLE` for a non-live device and
`INVALID_ARGUMENT` for an unavailable role.
`get_queue_info` faults `INVALID_HANDLE` for zero, stale, foreign-device, or
malformed tokens and returns the stable device-local ID and supported roles.
Backend family indices and native handles remain private. Resource descriptions
require a non-empty subset of the selected roles. A queue may use a resource
when at least one of its roles appears in the resource's `access` set. A valid
`GpuSpan` does not carry empty, unknown, or wider-than-backing access metadata.
Cold-path resource creation validates these facts under every policy; `FULL`
also diagnoses their violation during command recording before native mutation.

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
cmd_set_graphics_state(CommandList* commands, GraphicsState* state) -> void?
cmd_set_viewport(CommandList* commands, Viewport* viewport) -> void?
cmd_set_scissor(CommandList* commands, ScissorRect* scissor) -> void?
cmd_dispatch(
    CommandList* commands,
    GpuAddress root,
    Vec3u groups,
) -> void?
```

Pipeline binding selects the active compute or graphics pipeline without
compiling or synthesizing a variant. A failed bind preserves the previous active
pipeline. The live identity, owner, and bind snapshot needed for safe later
use are always checked. A successful FULL bind retains the pipeline and caches
its native
pipeline/layout, kind, render compatibility, cache-entry identity,
generated-work layout, and public diagnostic identity. Later draws and
dispatches use that snapshot without reading a pipeline cell, table, or cache.
FULL ownership prevents destruction until discard or command completion; under
TRUSTED, the caller must keep the pipeline live through completion.
Execution without a usable bound-pipeline snapshot faults under every policy.
`FULL` additionally diagnoses a compute/graphics pipeline-kind mismatch. Valid
graphics draws also require an active render pass; its phase is always checked.
Direct compute and graphics roots are pushed unchanged; zero is valid under
every policy. A shader must branch before dereferencing a zero root unless the
application deliberately relies on defined robustness behavior.

Graphics state belongs to a graphics-capable command buffer, not to one render
pass. After a compatible graphics pipeline is bound, `cmd_set_graphics_state`
records a complete viewport, scissor, raster, depth, and color packet before
or during a pass. `cmd_set_viewport` and `cmd_set_scissor` are the only narrow
overrides and may also be recorded in either phase. A minimal pass begin leaves
graphics state unchanged, and neither pass begin nor pipeline bind emits or
replays it. Host-safe descriptor access and lowering prerequisites are always
checked. Under `FULL`, complete-packet validation is atomic: invalid values
return `INVALID_ARGUMENT` before any native state is emitted.

Under `ContractValidation.FULL`, regular and generated graphics draws return
`COMMAND_RECORDING_ERROR` until one complete packet has succeeded in the
current command-buffer recording. Viewport/scissor overrides do not establish
that initialization. Compatible pipeline switches and pass boundaries do not
clear it, incompatible color-format domains clear color readiness, and
command-buffer reset clears all readiness.
Trusted command entries retain this requirement as a caller contract without a
warm validation branch. A wrong active pipeline kind returns
`INVALID_ARGUMENT`.

Each group count may be zero and is a caller precondition not to exceed the
corresponding `DeviceCaps.max_compute_work_group_count` component. `FULL`
diagnoses an over-limit call with `INVALID_ARGUMENT` before recording;
`TRUSTED` does not promise that semantic diagnostic.

Warm recording dispatches through the immutable operation table stored in the
authoritative record. There is one FULL policy table. Policy selection
happens at device or record setup; a warm `cmd_*` call never branches on policy
or resolves another device operation. The direct representation retains
`CommandOps` indirection and the fallible public signatures.

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

GraphicsState
    Viewport viewport
    ScissorRect scissor
    DynamicRasterState raster
    DepthState depth
    ColorState color

render_geometry_state(uint width, uint height) -> GraphicsState?
cmd_begin_render_pass(
    CommandList* commands,
    RenderPassDesc* desc,
) -> void?
cmd_set_graphics_state(CommandList* commands, GraphicsState* state) -> void?
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

Minimal render-pass begin always checks the host-safe descriptor shape, bounded
attachment identities/ranges, authoritative phase, and other safe-lowering
requirements. `FULL` additionally validates attachment usage, queue access,
formats, dimensions, and compatibility. An accepted call tracks attachment
references, emits one native begin-rendering command, and publishes the active
pass. It neither requires nor emits a `GraphicsState` packet.
A rejected begin leaves the command outside a pass, retains no new attachment
reference, and emits no native begin.

A complete packet emits viewport, scissor, five raster commands, and three
depth commands in fixed order, followed by three color-array commands when the
selected pipeline has color targets. There is no state diffing or dirty-bit
cache. `cmd_set_graphics_state` safely lowers and emits the same complete
replacement before or during a pass; select a compatible graphics pipeline
first. `FULL` prepares and validates every component before the first native
call and diagnoses invalid values and a non-graphics queue. The only narrow
state commands are viewport and scissor:

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
```

A conventional starting packet can be constructed from the pass dimensions:

```c3
gpu::GraphicsState state = gpu::render_geometry_state(
    pass.width,
    pass.height,
)!!;
gpu::ColorTargetState[1] color_targets = {
    gpu::color_blend_disabled(),
};
state.color = { .targets = color_targets[..] };
gpu::cmd_begin_render_pass(&commands, &pass)!!;
gpu::cmd_bind_pipeline(&commands, pipeline)!!;
gpu::cmd_set_graphics_state(&commands, &state)!!;
```

Keep a fully initialized `GraphicsState` as caller-owned cached state. To
change raster, depth, or color behavior, mutate that packet, bind the graphics
pipeline whose color domain it targets, and record the complete replacement:

One complete raster/depth/color packet keeps the caller's source of truth and
`FULL` validation atomic instead of maintaining parallel partial setters.
Viewport and scissor remain narrow overrides because callers commonly replace
them independently on hot paths; timing remains advisory rather than an API
gate.

```c3
state.raster = next_raster;
state.depth = next_depth;
state.color.targets = next_targets[..];
gpu::cmd_bind_pipeline(&commands, next_pipeline)!;
gpu::cmd_set_graphics_state(&commands, &state)!;
```

All slice backing storage, including `next_targets`, must remain live through
the call. The library does not retain the packet or its slices.

The helper faults `INVALID_ARGUMENT` before signed casts when either dimension
is zero or exceeds `int::max`. It returns a full-area viewport and scissor, the
conventional `[0, 1]` depth range, zero raster/depth state, and an empty color
packet. Zero depth state means disabled depth testing and writing and is valid
for drawing. Depth-only pipelines use the empty color packet unchanged; color
pipelines replace it with a packet matching their ordered color-format domain.

Migration: `full_render_graphics_state` is retired in favor of
`render_geometry_state`. The new name makes the boundary explicit: the helper
constructs geometry-related state but cannot manufacture caller-owned color
target storage. Existing color-pass callers must assign `GraphicsState.color`
before recording. This is a valid-caller precondition under `TRUSTED` and a
diagnosed target-count error under `FULL`.

Every viewport field
and computed axis endpoint must be finite. Width is positive, height is
nonzero, width and absolute height fit the selected device's
`maxViewportDimensions`, and both endpoints of each axis fit its
`viewportBoundsRange`. Height may be negative for a Y flip, coordinates may be
negative within those bounds, and the rectangle may extend beyond the active
render area. Both depth endpoints are independently in `[0, 1]`; reversed
depth ranges are valid. Rejecting zero height is a deliberate non-degenerate
library invariant, not a Vulkan 1.3 validity requirement.

Scissors use signed inputs, so valid callers provide nonnegative origins and
extents. Each widened offset-plus-extent fits `int::max` before native
lowering. The rectangle may extend beyond the render area and zero extent is a
valid empty clip. `FULL` diagnoses invalid viewport, scissor, raster, depth, or
color values with `INVALID_ARGUMENT` before changing command state.
Authoritative recording/pass phase is always checked and faults
`COMMAND_RECORDING_ERROR`.

Explicit command state survives compatible graphics pipeline and cache-alias
handle switches and render-pass boundaries. An incompatible color-format
domain invalidates color readiness while leaving the other command state
intact. Minimal begin and pipeline bind preserve state without hidden
re-emission. A complete packet replaces all graphics state; viewport and
scissor are the only independently replaceable components.
The current API intentionally exposes one viewport and one scissor only.

The complete packet's color target count and order must match the selected
graphics pipeline's ordered color-format domain. `FULL` validates the whole
packet before any native call, including integer-format blend rejection. A
compatible pipeline alias or pass boundary preserves initialization; binding a
different color-format domain clears it. Draw and generated-draw paths reject
an uninitialized domain. Every complete packet emits the deterministic full
sequence again, including exactly three native color-array commands for a
nonempty domain; identical packets are not suppressed.

This is a source-breaking experimental API. Replace
`cmd_begin_render_pass_with_state(commands, desc, state)` with independent
begin and complete-state calls. For a fresh recording, begin the pass, bind the
compatible graphics pipeline, set the complete packet, draw, and end. If an
incompatible pipeline persists from an earlier pass, bind the next compatible
pipeline before begin. A state fault after a successful begin leaves the pass
and its attachment references active, so the caller may correct and retry the
complete packet, end the pass, or discard the recording. Raster, depth, and
color changes migrate by mutating a fully initialized caller-owned packet and
recording it after the compatible pipeline bind.

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

A valid bound graphics pipeline exactly matches the pass color count and
formats, depth format, and sample count. `FULL` diagnoses compatibility misuse
before the affected FULL bind or pass begin emits native work; trusted
command recording treats it as a caller precondition.
Graphics and compute bindings are tracked independently by bind point. Ending
a render pass clears active attachment compatibility but does not release the
logical or native graphics binding; a compatible later pass can reuse it without
another native pipeline bind. Switching between graphics and compute likewise
preserves both native selections, while binding a different pipeline still
updates the selected logical pipeline.
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

reserve_generated_scratch(
    CommandAllocator* allocator,
    GeneratedScratchDesc* desc,
) -> void?
release_generated_scratch(
    CommandAllocator* allocator,
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

Root semantics are uniform across this entire operation family. The root
arguments to direct and indirect commands and every root field loaded from a
generated record are forwarded unchanged. Zero is valid for compute, vertex,
and fragment roots under `TRUSTED` and `FULL`. Shaders
using an optional zero root must branch before dereference unless the
application intentionally relies on defined robustness behavior.

Generated commands require an explicit cold reservation on their originating
command allocator. Reservations cannot be borrowed by another allocator, even
when it is bound to the same queue.
Each reservation is keyed by the exact public `PipelineHandle` and
`GeneratedWorkKind`, not by the shared native pipeline. Two alias handles for
one native pipeline therefore require separate reservations.
`max_commands_per_list` bounds the maximum generated count accepted by one
call, and `preprocess_buffer_count` bounds simultaneously retained calls for
that key across incomplete command lists. The backend asks the driver for the
exact size, alignment, and memory-type requirements for the pipeline, layout,
and declared maximum count before allocating. Vulkan defines that queried count
as the maximum the returned preprocess memory supports, so every recording at
or below it uses the reservation without another requirements query. Reserving
the same key replaces it; other keys in
the allocator remain live. The allocator descriptor's preprocess count
multiplied by command-buffer capacity bounds reservation slots, while
`generated_preprocess_bytes` bounds their total native bytes. A live reservation
retains its pipeline; release every key before destroying the pipeline.

Reservation replacement or `release_generated_scratch` returns
`RESOURCE_IN_USE` while the allocator has recording, executable, or
submitted work. Descriptors must set every field and remain within
`DeviceCaps.max_generated_work_count`, or reservation returns
`INVALID_ARGUMENT`. A reservation-table or byte-budget overflow returns
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`. Releasing a key that is not reserved
returns `INVALID_RESOURCE_STATE`. Generated recording returns deterministic
`GENERATED_SCRATCH_EXHAUSTED` when the count or matching-slot supply is
insufficient, and returns `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` when its fixed
per-list reservation-index slice is insufficient. Either failure preserves the
command list and native state for retry or discard. A driver-reported zero-byte
preprocess requirement still consumes one explicit reservation slot per
retained generated command; zero storage does not bypass caller-selected
capacity.

Generated record spans are caller-preconditioned to be 8-byte aligned and hold
the declared maximum count. The count span is a 4-byte-aligned GPU-readable
`uint`; both spans come from live allocations admitted to the recording queue.
The GPU-written count may be zero and must not exceed either the command maximum
or `DeviceCaps.max_generated_work_count`. A zero command maximum records no
native work. Generated indexed draws use `IndexType.U16` or `IndexType.U32`;
GPU-written index bounds remain subject to device robustness. The always-checked
path retains identity, bounded-range, overflow, reservation-capacity, and other
safe-lowering checks. `FULL` additionally diagnoses command semantic usage,
queue, count, pipeline, and state misuse.
The active pipeline supplies execution state and is not an API argument.
Root-reachable allocations and resources are caller-owned, are not tracked from
GPU-written addresses, and must remain live until the covering completion point
finishes.

Valid argument spans support indirect-command reads, are 4-byte aligned, and
contain `draw_count` (or `max_draw_count`) times the tight argument size. One
vertex/fragment root pair applies to every draw in a
multi-draw; per-draw variation indexes a table through `gl_DrawID` (see
`docs/shader_abi.md`). Direct draw counts and GPU-written count values may be
zero and must not exceed `DeviceCaps.max_draw_indirect_count`. In the count
variant, `max_draw_count` may exceed that limit, but the argument span must
hold `max_draw_count` entries; execution uses the smaller of `max_draw_count`
and the GPU-written count. Identity, backing bounds, and argument-byte overflow
needed for safe lowering are always checked. `FULL` diagnoses indirect usage,
queue, count, and index-type precondition violations.

Each GPU-written `DispatchIndirectCommand` component must not exceed the
corresponding `DeviceCaps.max_compute_work_group_count` component. Ordering
between argument writes and indirect consumption is the caller's barrier with
`after.indirect` set.

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

BufferTextureCopyDesc
    GpuSpan src
    uint row_length_texels
    TextureHandle texture
    uint mip
    uint base_layer
    uint layer_count
    uint x
    uint y
    uint z
    uint width
    uint height
    uint depth

TextureBufferCopyDesc
    TextureHandle texture
    GpuSpan dst
    uint row_length_texels
    uint mip
    uint base_layer
    uint layer_count
    uint x
    uint y
    uint z
    uint width
    uint height
    uint depth

cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) -> void?
cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) -> void?
cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) -> void?
cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) -> void?
```

Valid copy spans are nonzero, equal in size, non-overlapping, and have the
required usage and queue access.
`cmd_fill_buffer` fills the exact destination span; its byte offset and
size are 4-byte aligned. There is no zero-size shorthand.

For buffer-texture copies, zero width and height select the remaining selected
mip extent from x and y; this includes nonzero x/y offsets. On 2D textures,
z must be zero, depth must be zero or one and normalizes to one, and
base/layer count select array layers with zero layer count meaning one. On 3D
textures, base layer must be zero, layer count must be zero or one, z selects
the first slice, and zero depth selects the remaining mip depth from z.
Positive depth selects exactly that many slices. Mip depth reduces alongside
width and height.

Zero `row_length_texels` tightly packs each row; a positive value must be at
least the copied width. The span must contain
`texel_size * normalized_row_texels * copy_height * slice_count`, where slice
count is the 2D array-layer count or 3D copy depth. This deliberately includes
every padded row of every layer or depth slice; no separate slice pitch is
exposed.

Live identities, dimension/subresource/extent bounds, exact backing range,
byte-size overflow, and alignment required for safe lowering are always
checked. `FULL` adds semantic usage and queue-access diagnostics and retains
the span and texture transactionally. Copies use the transfer source or
destination native layout, but callers remain responsible for the surrounding
texture barriers and completion-based lifetimes.

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

### Timestamp queries

```text
TimestampPoolHandle
    ulong owner
    uint index
    uint generation

TimestampPoolDesc
    uint capacity
    ZString debug_name

create_timestamp_pool(
    Device* device,
    TimestampPoolDesc* desc,
) -> TimestampPoolHandle?
destroy_timestamp_pool(
    Device* device,
    TimestampPoolHandle pool,
) -> void?
cmd_reset_timestamps(
    CommandList* commands,
    TimestampPoolHandle pool,
    uint first,
    uint count,
) -> void?
cmd_write_timestamp(
    CommandList* commands,
    TimestampPoolHandle pool,
    uint index,
    StageMask stage,
) -> void?
cmd_resolve_timestamps(
    CommandList* commands,
    TimestampPoolHandle pool,
    uint first,
    uint count,
    GpuSpan dst,
) -> void?
read_timestamps(
    Device* device,
    TimestampPoolHandle pool,
    uint first,
    uint count,
    ulong[] out,
) -> void?
timestamp_delta_ns(
    TimestampCaps* caps,
    QueueKind queue,
    ulong begin,
    ulong end,
) -> double?
```

Pool capacity must be positive, and the device must report at least one role in
`DeviceCaps.timestamps.queues`. Reset and resolve use nonempty in-bounds ranges;
write uses one in-bounds index. Reset and resolve record only outside a render
pass. Write is valid inside or outside a render pass and, under `FULL`, requires
exactly one executable stage supported by the recording queue.

Resolve writes `count` tightly packed `ulong` values at the start of `dst`. The
destination offset is aligned to `ulong::size`, the span covers the complete
result, and under `FULL` it admits transfer-destination use by the recording
queue. Resolve requests availability on the device, so command recording does
not block the host. An ordered resolve of an unwritten query can leave the
device waiting indefinitely.

`read_timestamps` directly requests 64-bit results into `out[0..count]` and
never waits, allocates staging, or establishes submission completion. If any
requested result is unavailable, it returns `DEVICE_BUSY`; the requested output
range is unspecified and must be ignored. Order the read after the relevant
submission completion and retry explicitly.

The caller owns query history in both validation modes: reset before reuse,
write every query before resolve or host read, and keep the pool and destination
alive through execution. `FULL` retains the pool for recorded commands and the
resolve destination allocation, but does not track per-slot reset/write state.
`TRUSTED` performs no command-reference work.

`timestamp_delta_ns` masks and subtracts using the selected role's valid-bit
width, handles one modular counter wrap, and scales by `period_ns`. Compare only
timestamps written on the same native queue. Different logical roles are
comparable only when they alias that queue; distinct native queues are not
calibrated by this API. An unadvertised role returns `UNSUPPORTED_FEATURE`;
invalid capabilities, role values, widths, or periods return `INVALID_ARGUMENT`.

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
    indirect

Barrier
    StageMask before
    StageMask after

cmd_barrier(CommandList* commands, Barrier* barrier) -> void?
```

`Barrier` is a global execution and memory dependency. It has no resource
handle, address, range, layout, or queue-family field. As preconditions, each
stage mask is nonempty; `all` and `present` are each exclusive; bits are known;
and stages are consistent with the recording queue. `FULL` diagnoses those
semantic violations with `INVALID_ARGUMENT`. Under `TRUSTED`, stage shape and
queue compatibility remain caller contracts; null-barrier and authoritative
command-state checks remain active.

Normal host, transfer, shader, color-output, and depth-output access scopes are
derived from the stage masks. `after.indirect` adds indirect-command reads and,
when generated work is enabled, command-preprocess reads. `before.indirect`
adds the matching execution scope without inventing source access.
`depth_output` supplies the depth/stencil execution and read/write scope.
Descriptor-set publication has no public GPU memory-hazard bit. The library
does not infer barriers. Cross-queue dependencies use
`SubmitDesc.completion_waits`, not `cmd_barrier`.

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
state. `TextureState.layout` is an operational requirement, not descriptive
metadata: the caller must record the layout established by earlier explicit
ordering and provide the layout required by the next use. Layout, execution
stages, and read/write access are independent.
`sampled_at` selects `SAMPLED` with read access, while `storage_at` selects
`STORAGE` and preserves the caller's access. Both constructors only compose a
value and insert no synchronization.

The semantic matrix is exact:

| Layout | Native Vulkan layout | Public stages | Access | Required texture/queue |
|---|---|---|---|---|
| `UNDEFINED` | `VK_IMAGE_LAYOUT_UNDEFINED` | empty | empty | source only |
| `TRANSFER_SOURCE` | `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` | transfer, or exclusive `all` | read | `transfer_src`; transfer-capable queue |
| `TRANSFER_DESTINATION` | `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` | transfer, or exclusive `all` | write | `transfer_dst`; transfer-capable queue |
| `SAMPLED` | `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` for color; `VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL` for depth/stencil | nonempty vertex/fragment/compute combination, or exclusive `all` | read | `sampled`; shader-capable queue |
| `STORAGE` | `VK_IMAGE_LAYOUT_GENERAL` | nonempty vertex/fragment/compute combination, or exclusive `all` | read, write, or both | `storage`; shader-capable queue |
| `COLOR_ATTACHMENT` | `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` | color output, or exclusive `all` | read, write, or both | `color_attach`; non-depth format; graphics queue |
| `DEPTH_ATTACHMENT` | `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` | depth output, or exclusive `all` | read, write, or both | `depth_attach`; depth format; graphics queue |
| `PRESENT` | `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` | empty | empty | swapchain-owned non-depth texture; graphics queue |

Known bits, legal layouts, and stage/access consistency are caller
preconditions. Texture states cannot name the global-only `host`, `present`, or
`indirect` stage bits. `FULL` diagnoses those semantic violations with
`INVALID_ARGUMENT`; under `TRUSTED`, layout, stage, access, and queue
compatibility remain caller contracts. Same-state transitions remain valid
explicit memory dependencies.
Sampled depth/stencil textures lower to the appropriate read-only depth/stencil
layout.

A zero `view` selects the full texture. Zero mip or layer counts select the
remaining range from their respective base. Bounded subresource range is always
checked for safe lowering. `FULL` additionally diagnoses invalid format
reinterpretation.

Recording resolves the texture handle once, normalizes the range once, lowers
both states once, assembles one native barrier, and emits it once. Live
identity, bounded subresource range, and safe native lowering remain
always-checked. `FULL` additionally validates queue access, semantic values,
stage support, immutable texture usage and format, and presentation ownership.
Rejection before emission rolls back any retained command reference.

The backend does not infer, globally track, compare, or repair prior state. A
wrong `before` declaration is a caller synchronization error; applications own
their layout history, including separate histories for independently
transitioned subresource ranges.

At the presentation boundary, `AcquiredImage.prior_state` is directly usable as
the first transition's `before` value. The fixed public `PRESENT` state has
empty stages and access because the presentation engine is external to the
pipeline. A transition to `PRESENT` keeps the last queue-side producer scope
and uses destination `NONE`/`NONE`. A transition from `PRESENT` uses no source
access, anchors its source stage to the paired first queue-side consumer, and
keeps that consumer's ordinary destination scope.

`SubmitDesc.readiness_before` names the destination stages of the first
command that consumes the acquired image. When the first recorded transition
leaves `PRESENT`, its paired queue-side state and `readiness_before` must cover
the same first consumer stages. The acquire semaphore wait and the barrier's
source-stage anchor then order the layout transition after image readiness.
Submission still signals the presentation semaphore after all submitted
commands, and presentation waits that semaphore.

`UNDEFINED` supplies no source dependency and discards prior contents. Use it
only for first use or after earlier access has been ordered separately.

There is one texture-layout profile. Every texture state lowers through the
mapping above; device creation does not negotiate an alternate layout policy.
A global `Barrier` cannot establish or change a texture layout because it has
no texture identity or subresource range. Use an explicit `TextureBarrier`
whenever the required layout changes, including initialization and
presentation transitions. No command helper silently inserts a barrier for a
later use.

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
Timestamp pool names are copied at creation, used for native object naming when
enabled, and reported as `DebugResourceKind.TIMESTAMP_POOL` with public identity
during eligible teardown leak scans.

`TRUSTED` still reports backend/device failures, but does not promise detailed
misuse diagnostics, retain command resources, or run teardown leak scans.
`FULL` uses diagnostic command operations, retains command resources, runs teardown
leak scans, and reports detailed rejected fields and invariants for command
misuse.
These library diagnostics do not require Vulkan validation layers. Backend failures preserve the public
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

Accepted destruction scans private Vulkan state under `FULL`. Callback
presence only selects delivery and does not enable this scan under `TRUSTED`.
Normal live children are rejected before this scan; diagnostics
are a safety net for internal, partial-initialization, and device-loss leftovers.
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
acquire_next_image(
    Device*,
    SwapchainHandle,
    ulong timeout_ns = 0,
) -> AcquiredImage?
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

Acquisition passes `timeout_ns` to the backend unchanged. Zero, the default,
is nonblocking. A finite value bounds native image acquisition in nanoseconds.
`TIMEOUT_INFINITE` requests an unbounded native wait and is valid only when the
surface platform guarantees presentation forward progress. Shipped render
loops should normally use zero or a small finite budget so event handling and
shutdown remain responsive.

`WAIT_TIMEOUT`, out-of-date, surface loss, allocation failure, and device loss
do not advance the acquisition sequence or semaphore ring and do not publish a
pending image. The caller may retry a timeout with the same swapchain; the
backend reuses the same eligible semaphore until an acquisition succeeds.

```c3
gpu::AcquiredImage acquired = gpu::acquire_next_image(
    &device,
    swapchain,
    2_000_000,
)!;
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
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(device, queue)!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
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
