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

RuntimeDesc
    BackendKind backend
    bool enable_validation
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
```

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
backend instance. Destroy the device before destroying that runtime.
`create_device_from_desc` is the direct headless convenience path; it performs
its own discovery and owns that discovery state.

```text
DescriptorHeapMode
    AUTO                    (indexing preferred; buffer where proven)
    DESCRIPTOR_BUFFER
    DESCRIPTOR_INDEXING

DeviceDesc
    BackendKind backend
    bool enable_validation
    bool enable_debug_names
    DescriptorHeapMode descriptor_heap_mode
    uint texture_descriptor_capacity      (0 = default; docs/limitations.md)
    uint sampler_descriptor_capacity
    uint texture_capacity
    char[] pipeline_cache_data            (warm-start blob; §8)
    ZString application_name
    DebugMessageCallback debug_callback       (null = no structured delivery)
    void* debug_user_data

DeviceCaps
    bool strict_enabled
    bool presentation_enabled
    bool buffer_device_address
    bool synchronization2
    bool dynamic_rendering
    bool shader_int64
    bool draw_indirect_count
    bool descriptor_buffer
    bool descriptor_indexing
    bool async_compute
    QueueCounts queues
    bool line_polygon_mode
    uint max_texture_descriptors
    uint max_sampler_descriptors
    uint max_color_attachments
    uint max_push_constant_size
    Vec3u max_compute_work_group_count
    uint max_draw_indirect_count
    usz min_uniform_alignment
    usz min_storage_alignment
    usz min_texel_buffer_alignment
    float max_sampler_lod_bias
    float max_sampler_anisotropy

Device                           (slot | generation | reserved)
get_device_backend(Device*)      -> BackendKind?
get_device_caps(Device*)         -> DeviceCaps?
```

`strict_enabled` and the descriptor-path flags report enabled device state,
not adapter support. Query request support before creation.

Descriptor capacities are exact creation requests, not clampable upper bounds.
For descriptor indexing, texture capacity contributes once to the sampled-image
binding and once to the storage-image binding. Per-stage resource usage is
`2 * texture_descriptor_capacity`; plain sampler descriptors do not count toward
that limit. All-pools usage is `2 * texture_descriptor_capacity +
sampler_descriptor_capacity`. `create_device_from_desc` returns `INVALID_ARGUMENT` when a
requested capacity exceeds a per-type or aggregate device limit. On success,
`DeviceCaps.max_texture_descriptors` and
`DeviceCaps.max_sampler_descriptors` report the capacities of the created heap.

Creation:

```text
strict_device_request() -> DeviceRequest
supports_device_request(Adapter*, DeviceRequest*) -> DeviceRequestSupport?
request_presentation(DeviceRequest, Surface*) -> DeviceRequest?
request_queues(DeviceRequest, QueueRequirements) -> DeviceRequest?
create_device(Adapter*, DeviceRequest*) -> Device?
create_device_from_desc(DeviceDesc*) -> Device?
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
recording calls borrow that pin without acquiring another one.

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
Sampler
PipelineHandle
ShaderHandle
SwapchainHandle
```

Owning tokens have zero-valued invalid constants such as
`GPU_ALLOCATION_INVALID`, `TEXTURE_HANDLE_INVALID`, and `SAMPLER_INVALID`.
`token.is_valid()` checks the owner and generation; operations also validate
the local slot generation. Public code should not inspect or construct the
representation.

Handles, `Sampler`, `Queue`, `TextureIndex`, `SamplerIndex`, `GpuAddress`, `GpuSpan`,
command tokens, and synchronization values are runtime-only and scoped to their
owning device. Do not persist, serialize, reconstruct, or pass them across
device or process lifetimes. Compare `Queue` values as wholes and use
`get_queue_info` for inspection; do not construct or mutate queue fields.
`CommandList` embeds a copy of its owning `Device` token and does not borrow
caller variable storage.

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
recovered when a public operation resolves the span. A zero `GpuAddress` is
invalid.

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
`TextureViewDesc* view`). Null required input faults `INVALID_ARGUMENT` before a
backend call. Null or stale owner-token pointers fault `INVALID_HANDLE`.

| Fault | Fired by | Typical cause |
|---|---|---|
| `UNSUPPORTED_BACKEND` | `create_runtime`, `create_device_from_desc` | no Vulkan 1.3 driver / loader found no ICD |
| `UNSUPPORTED_FEATURE` | device creation, `create_runtime`, `create_texture`, `create_dedicated_texture`, `create_swapchain`, `create_graphics_pipeline`, `intern_sampler`, `publish_sampler` | validation layers not installed; presentation was not requested or is unsupported for the adapter and surface; missing optional or required device feature; unsupported image format or usage; adapter rejects a valid texture descriptor |
| `INVALID_ARGUMENT` | runtime adapter indexing; `request_queues`; any create/export; `allocate_memory`; `GpuSpan.checked_subspan`; `get_span_mapping`; `get_span_address`; `flush_mapped_span`; `invalidate_mapped_span`; `get_queue`; `submit`; `present`; `cmd_copy_buffer`/`cmd_fill_buffer`/buffer↔texture copies; draw/dispatch and barrier commands; `cmd_set_viewport`/`cmd_set_scissor`; pipeline/shader creates; `texture_transition`; `create_texture_descriptors`; `intern_sampler` | null or malformed input, zero allocation/span size, non-power-of-two alignment, unavailable mapping/address capability, range outside its immediate parent, offset overflow, `out_indices.len != descs.len`, invalid queue access, missing resource usage, malformed command state data, or an out-of-range value |
| `INVALID_HANDLE` | runtime and adapter queries; destruction; device/queue/completion queries; allocation info/span/mapping/address/visibility operations; any resource-handle-taking call; `cmd_*`; command lifecycle; `submit` | zero, destroyed, stale, or foreign runtime, adapter, device, queue, completion point, allocation, span, resource, or command token |
| `INVALID_RESOURCE_STATE` | swapchain lifecycle, `cmd_texture_barrier` | an acquired swapchain image is pending during resize, or `old_layout` disagrees with the list's effective layout |
| `OUT_OF_HOST_MEMORY` | creates; mapped visibility | driver host-allocation failure |
| `OUT_OF_DEVICE_MEMORY` | allocation and texture creates; mapped visibility | backend device-memory exhaustion |
| `DEVICE_LOST` | any Vulkan-backed operation | Vulkan returned `VK_ERROR_DEVICE_LOST`; the affected device rejects later operations while peer devices remain usable |
| `DEVICE_BUSY` | public device operations; `submit`; `destroy_device` | the operation observed a closing device or exhausted bounded pin acquisition, submission reached the native timeline-value-difference limit, or destruction found an active operation or incomplete queue work; retry with the unchanged token |
| `RESOURCE_IN_USE` | resource destruction, `free_allocation`, `destroy_device`, `destroy_runtime`, `destroy_surface` | recording, executable, or incomplete submitted work explicitly references a resource; active pipeline creation uses a shader; a placed or dedicated texture depends on an allocation; a live descriptor owns a texture; a device has a live child; a runtime has a live surface or device; or a surface has a live swapchain |
| `SLOT_TABLE_FULL` | runtime, device, allocation, and resource creates; `intern_sampler`; `begin_commands`; `acquire_next_image`; queue submission | a registry or handle table is at capacity, or a queue completion or swapchain acquisition sequence is exhausted |
| `DESCRIPTOR_HEAP_FULL` | descriptor pool creation/allocation, `create_texture_descriptor`, `create_texture_descriptors`, `publish_sampler` | Vulkan descriptor-pool exhaustion or fragmentation, or capacity below the live descriptor count; overflowing texture batches and sampler publication leave existing entries untouched |
| `PIPELINE_CREATE_FAILED` | pipeline creates | driver rejected the state combination, shader, or compilation |
| `SHADER_INVALID` | `create_shader` | SPIR-V rejected by the driver |
| `SURFACE_LOST` | surface creation/query/enumeration, swapchain create/resize, acquire, present | native window or surface was destroyed or became unavailable; destroy the swapchain and create a new one from fresh native handles |
| `SWAPCHAIN_OUT_OF_DATE` | `create_swapchain`, `resize_swapchain`, `acquire_next_image`, `present` | swapchain no longer matches the surface; `resize_swapchain` and retry |
| `COMMAND_RECORDING_ERROR` | `cmd_*`, `end_commands`, `discard_commands`, `discard_executable_commands`, `submit` | call outside its required recording state, duplicate command token in one submit batch, or token that is already being submitted |
| `WAIT_TIMEOUT` | `wait_completion`, `acquire_next_image` | bounded wait or transient image unavailability; retry with the unchanged completion point or resource |
| `BACKEND_ERROR` | any Vulkan-backed operation | unclassified or internal native failure; inspect backend diagnostics; does not imply device loss |

Backend-local Vulkan/VMA faults should not leak unless they carry useful public meaning. Map them to public faults and log backend details when validation/debug is enabled. `DEVICE_LOST` is reserved for an explicit native device-loss result; an unmapped native result becomes `BACKEND_ERROR`.

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
record the copy and a `TRANSFER_WRITE` to `HOST_READ` barrier on the
destination, submit, wait for or poll completion, invalidate the span, then
read its mapping. Free or reuse the owning allocation only after completion.

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
D24_UNORM_S8_UINT   (current backend profile reports unsupported)
```

### Texture descriptors

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

TextureDimensionSupport
    bool tex_1d
    bool tex_2d
    bool tex_3d
    bool cube

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
    TextureDimensionSupport dimensions
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

TextureDimension
    TEX_1D   (query false; creation faults INVALID_ARGUMENT)
    TEX_2D   (current backend profile)
    TEX_3D   (query false; creation faults INVALID_ARGUMENT)
    CUBE     (query false; creation faults INVALID_ARGUMENT)

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

TextureViewDesc
    Format format
    uint base_mip
    uint mip_count
    uint base_layer
    uint layer_count

TextureDescriptorDesc
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
create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) -> TextureIndex?
destroy_texture_descriptor(Device* device, TextureIndex index) -> void?
create_texture_descriptors(Device* device, TextureDescriptorDesc[] descs, TextureIndex[] out_indices) -> void?
```

`get_texture_format_support` reports library-creatable support, not every raw Vulkan capability. Each usage bit comes from the same exact 2D optimal-tiling query used by creation, but the bits are independent; use `supports_texture_desc` for a usage combination. The backend profile masks every dimension except 2D and every sample count except one. Per-format usages and linear filtering remain adapter-dependent; D24S8 reports empty support until the rendering path supports it end to end. `supports_texture_desc` returns false for an empty, unknown, or unavailable access set.

`supports_texture_desc` checks the exact optimal-tiling format, combined usage,
normalized extent, mip and layer counts, access roles, and required single-sample
image properties without allocating. A false result caused by malformed input
corresponds to `INVALID_ARGUMENT` at creation; a structurally valid descriptor
rejected by the adapter corresponds to `UNSUPPORTED_FEATURE`. Memory exhaustion
can still make creation fail after a true capability result.

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

`TextureHandle` owns the image. `TextureIndex` is a shader-visible descriptor
heap entry. Destroyed texture indices are immediately reusable. The caller must
first discard or complete every use and remove stale indices from GPU-visible
data. Published sampler indices instead remain stable until device destruction.

`create_texture_descriptors` batch-creates N descriptors under one lock hold, ending in one accumulated descriptor-set update in indexing mode (buffer mode writes per-item, already a mapped-memory store). `out_indices.len` must equal `descs.len` (`INVALID_ARGUMENT` otherwise); an empty `descs` is a no-op success. A zero-initialized `TextureDescriptorDesc.view` collapses to the default view, same as a null `view` to `create_texture_descriptor`.

All-or-nothing: a fault leaves descriptor cells and generations, allocator/free-list state, texture view caches, Vulkan image-view ownership, and `out_indices` unchanged. Only a successful batch returns owned indices; release each with `destroy_texture_descriptor`.

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

Sampler                       (opaque owner | slot | generation)
intern_sampler(Device* device, SamplerDesc* desc) -> Sampler?
publish_sampler(Device* device, Sampler sampler) -> SamplerIndex?
```

`intern_sampler` returns an immutable device-owned identity. Descriptions with
the same effective filtering, addressing, LOD, anisotropy, and comparison state
return the same `Sampler`; `debug_name` is not part of identity. LOD values must
be finite, the absolute `mip_lod_bias` must not exceed
`DeviceCaps.max_sampler_lod_bias`, and `min_lod` must not exceed `max_lod`.
Undefined filter, address, or enabled comparison enum values fault
`INVALID_ARGUMENT`. Anisotropy requires the reported device capability. Sampler
identities and their native objects live
until device destruction and have no individual destroy operation.

`publish_sampler` is separate and requires strict capability. It returns one
stable shader-visible `SamplerIndex` for the identity. Repeated publication is
idempotent. `DESCRIPTOR_HEAP_FULL` leaves the identity valid and consumes no
entry; a device without strict capability returns `UNSUPPORTED_FEATURE` before
backend publication.

## 8. Shader and pipeline API

### Shader modules

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

create_shader(Device* device, ShaderDesc* desc) -> ShaderHandle?
destroy_shader(Device* device, ShaderHandle shader) -> void?
```

Shader compilation can be handled by tools or samples. The core library consumes SPIR-V bytes.

### Compute pipelines

```text
ComputePipelineDesc
    ShaderHandle shader
    uint push_constant_size
    ZString debug_name

create_compute_pipeline(Device* device, ComputePipelineDesc* desc) -> PipelineHandle?
```

The first ABI requires at least the 8-byte `RootPush` size. The requested size
must be a multiple of four and no greater than
`DeviceCaps.max_push_constant_size`, which reports the selected device's
Vulkan `maxPushConstantsSize` limit.

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

RasterState
    CullMode cull_mode
    FrontFace front_face
    PolygonMode polygon_mode
    float depth_bias_constant   (any nonzero bias field enables depth bias)
    float depth_bias_slope
    float depth_bias_clamp

BlendState
    bool enable
    BlendFactor src_color
    BlendFactor dst_color
    BlendOp color_op
    BlendFactor src_alpha
    BlendFactor dst_alpha
    BlendOp alpha_op

GraphicsPipelineDesc
    ShaderHandle vertex_shader
    ShaderHandle fragment_shader
    PrimitiveTopology topology
    DepthState depth
    RasterState raster
    BlendState blend
    Format[] color_formats
    Format depth_format
    ZString debug_name

create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) -> PipelineHandle?
destroy_pipeline(Device* device, PipelineHandle pipeline) -> void?
```
`PolygonMode.LINE` is optional. Query `DeviceCaps.line_polygon_mode` before
using it; unsupported LINE creation returns `UNSUPPORTED_FEATURE`.
`PrimitiveTopology.LINES` remains available independently with FILL mode.

`color_formats` carries at most `MAX_COLOR_ATTACHMENTS` (8) entries.

### Pipeline deduplication

Pipeline creation deduplicates through a descriptor-keyed cache. Every create
returns a fresh handle, but descriptors identical in immutable state (shaders,
topology, polygon mode, blend, formats, and — for compute — push size) alias
one backend pipeline underneath. Raster cull/front-face and depth
test/write/compare state are applied per handle at draw time as dynamic state,
so descriptors differing only there also share a backend pipeline.

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
DeviceDesc.pipeline_cache_data                                 (warm-start blob)
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
    CompletionPoint[] completion_waits
    SwapchainHandle swapchain
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
context owner reclaims them, before its next allocation.

`SubmitDesc.completion_waits` accepts published points from the same device.
Cross-queue points become waits on their queue-owned timelines. Published points
from the target queue are validated and then elided because queue order is
inherent. Stale, unpublished, malformed, and foreign-device waits fault
`INVALID_HANDLE` before native submission and preserve every command token.
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
or tracked-layout mutation.

Allocations reached only through root GPU pointers remain a caller contract:
every reachable allocation must admit the recording role because nested pointers
are opaque. The backend keeps one admitted family exclusive and uses private
concurrent sharing only for distinct admitted families.

Transfer/render helper descriptors (`BufferCopyDesc`, `BufferTextureCopyDesc`,
`TextureBufferCopyDesc`, `ClearColor`,
`ClearDepthStencil`) are documented in the generated reference.

### Dispatch

```text
cmd_dispatch(
    CommandList* commands,
    PipelineHandle pipeline,
    GpuAddress root,
    Vec3u groups,
) -> void?
```

Each group count may be zero and must not exceed the corresponding component
of `DeviceCaps.max_compute_work_group_count`. An over-limit call faults
`INVALID_ARGUMENT` before a backend command is recorded.

### Render pass

```text
ColorTargetDesc
    TextureHandle texture
    uint mip_level
    uint array_layer
    LoadOp load_op
    StoreOp store_op
    ClearColor clear

DepthTargetDesc
    TextureHandle texture
    LoadOp load_op
    StoreOp store_op
    ClearDepthStencil clear

RenderPassDesc
    ColorTargetDesc[] colors
    DepthTargetDesc* depth
    uint width
    uint height

cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) -> void?
cmd_end_render_pass(CommandList* commands) -> void?
```

Render-pass begin initializes one viewport and one scissor to the full pass
extent. Callers may override either state for subsequent draws:

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

Both commands are valid only inside a render pass. Viewports require finite,
nonnegative origins, positive extents, pass-local endpoints, and depth
endpoints in `[0, 1]`; reversed depth ranges are valid. Scissors use signed
inputs so negative origins/extents fault, while zero extent is a valid empty
clip. Scissor endpoints must not overflow and both rectangles stay within the
active pass. Invalid values return `INVALID_ARGUMENT` before changing dynamic
state; calls outside a pass return `COMMAND_RECORDING_ERROR`.

Explicit viewport/scissor state survives graphics pipeline and cache-alias
handle switches. The next render-pass begin restores the full-pass defaults.
The current API intentionally exposes one rectangle only and does not support
negative-height viewport flips or off-pass overscan.

A pass names at least one color target or a depth target; depth-only passes
(the shadow-map shape) are valid. A depth target needs `depth_attach` usage
and the `DEPTH_STENCIL` tracked layout. `D32_FLOAT` is the only supported
depth format; pipelines name it in `GraphicsPipelineDesc.depth_format`.
Every selected color mip and the depth texture's mip zero must cover the pass
dimensions; smaller compatible render areas are valid. The color count must
not exceed `DeviceCaps.max_color_attachments`, which is the lesser of the
library ceiling and the selected device's Vulkan limit.

Depth clear values are explicit: a zero-initialized `ClearDepthStencil`
clears depth to **0.0**, which fails every LESS-compare draw. The standard
far-plane clear is an explicit `{ .depth = 1.0 }`; reverse-Z setups clear to
0.0 deliberately.

### Draw

```text
cmd_draw(
    CommandList* commands,
    PipelineHandle pipeline,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) -> void?

cmd_draw_indexed(
    CommandList* commands,
    PipelineHandle pipeline,
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

cmd_draw_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count) -> void?
cmd_draw_indexed_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count, index_span, index_type) -> void?
cmd_draw_indexed_indirect_count(commands, pipeline, vertex_root, fragment_root, args, count_span, max_draw_count, index_span, index_type) -> void?
cmd_dispatch_indirect(commands, pipeline, root, args) -> void?
```

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
between argument writes and indirect consumption is the caller's barrier
(`INDIRECT_COMMAND` / `INDIRECT_READ`).

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
cmd_buffer_barrier(TRANSFER_WRITE -> HOST_READ) on the destination
submit
poll_completion or wait_completion
invalidate_mapped_span
get_span_mapping
```

The caller owns the allocation, barriers, completion point, and mapped data.
No application work boundary or readback-specific token is required.

### Barriers

```text
Stage
    HOST
    TRANSFER
    COMPUTE_SHADER
    VERTEX_SHADER
    FRAGMENT_SHADER
    COLOR_ATTACHMENT
    DEPTH_STENCIL
    INDIRECT_COMMAND
    PRESENT
    NONE

Hazard
    HOST_WRITE
    HOST_READ
    TRANSFER_READ
    TRANSFER_WRITE
    SHADER_READ
    SHADER_WRITE
    COLOR_READ
    COLOR_WRITE
    DEPTH_READ
    DEPTH_WRITE
    INDIRECT_READ
    PRESENT_READ
    NONE
    SHADER_READ_WRITE
    COLOR_READ_WRITE
    DEPTH_READ_WRITE

BufferBarrier
    GpuSpan span
    Stage before_stage
    Stage after_stage
    Hazard before_hazard
    Hazard after_hazard

TextureBarrier
    TextureHandle texture
    Stage before_stage
    Stage after_stage
    Hazard before_hazard
    Hazard after_hazard
    TextureLayout old_layout
    TextureLayout new_layout

GlobalBarrier
    Stage before_stage
    Stage after_stage
    Hazard before_hazard
    Hazard after_hazard

cmd_buffer_barrier(CommandList* commands, BufferBarrier* barrier) -> void?
cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) -> void?
cmd_global_barrier(CommandList* commands, GlobalBarrier* barrier) -> void?
```

`BufferBarrier` applies to the exact nonzero span. It has no whole-buffer or
zero-size shorthand.

`Stage.NONE` is an empty execution scope. Use it only for a barrier side that
has no pipeline work to wait for. Explicit `Stage.PRESENT` and
`Hazard.PRESENT_READ` are broad raw-barrier spellings that map to all commands
and memory read. The `TextureUse.PRESENT` preset instead uses
`COLOR_ATTACHMENT` with no access scope so its barriers chain with private WSI
wait and signal stages.

`TextureLayout` values: UNDEFINED, GENERAL, COLOR_ATTACHMENT, DEPTH_STENCIL,
SHADER_READ, TRANSFER_SRC, TRANSFER_DST, PRESENT. `old_layout` must match the
list's effective layout — its own pending transitions if it has recorded any
for the texture, else the tracked layout — or the barrier faults
`INVALID_RESOURCE_STATE`. A recorded transition is staged on the list and
only commits onto tracked state when the list submits; a list that never
submits leaves tracked state untouched.

For common whole-texture transitions, `TextureUse` maps resource intent to
an exact synchronization tuple:

| `TextureUse` | `Stage` | `Hazard` | `TextureLayout` |
|---|---|---|---|
| `UNDEFINED` | `HOST` | `NONE` | `UNDEFINED` |
| `TRANSFER_DESTINATION` | `TRANSFER` | `TRANSFER_WRITE` | `TRANSFER_DST` |
| `SAMPLED_COMPUTE` | `COMPUTE_SHADER` | `SHADER_READ` | `SHADER_READ` |
| `SAMPLED_FRAGMENT` | `FRAGMENT_SHADER` | `SHADER_READ` | `SHADER_READ` |
| `STORAGE_COMPUTE` | `COMPUTE_SHADER` | `SHADER_READ_WRITE` | `GENERAL` |
| `COLOR_ATTACHMENT` | `COLOR_ATTACHMENT` | `COLOR_READ_WRITE` | `COLOR_ATTACHMENT` |
| `DEPTH_ATTACHMENT` | `DEPTH_STENCIL` | `DEPTH_READ_WRITE` | `DEPTH_STENCIL` |
| `PRESENT` | `COLOR_ATTACHMENT` | `NONE` | `PRESENT` |

```text
texture_transition(TextureHandle texture, TextureUse before, TextureUse after)
    -> TextureBarrier?
```

The constructor is pure: it does not inspect tracked layout, record a command,
or insert synchronization. `UNDEFINED` is source-only and faults
`INVALID_ARGUMENT` as `after`. Same-use barriers remain valid for explicit
memory ordering. Presets intentionally cover only common cases; construct a
raw `TextureBarrier` for transfer sources, other shader stages, read-only
attachments, subresource-specific work, or any unusual tuple.

At the presentation boundary, both barrier sides use `COLOR_ATTACHMENT` for
the texture transition itself. The private readiness bridge conservatively
covers the complete submission. Access remains asymmetric by composition:
`PRESENT -> COLOR_ATTACHMENT` uses `NONE -> COLOR_READ_WRITE`, while
`COLOR_ATTACHMENT -> PRESENT` uses `COLOR_READ_WRITE -> NONE`.

The `UNDEFINED` preset's `HOST`/`NONE` source scope is only for first use or
discard when no earlier GPU work still accesses the texture. Reinitializing a
texture that an earlier submission may still read requires a raw barrier whose
source stage and hazard cover that access.

No command helper should silently insert barriers for a later use.

### Structured debug messages, labels, and leak reporting

`DeviceDesc.debug_callback` optionally receives `DebugMessage` values for
public-contract failures, backend failures, Vulkan validation/performance
messages, and resource-lifetime warnings during device teardown. A null
callback disables structured delivery and never changes the value, fault,
rollback, or resource state of the originating operation. The flat public
fault remains authoritative; `has_fault` says whether `public_fault`
accompanies the message.

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

Stored public debug names remain available when validation is disabled;
`enable_debug_names` controls best-effort Vulkan object naming independently.

Contract diagnostics cover resource creation, recording, submission, memory,
descriptors, pipelines, queue progress, and WSI failures. Backend failures
preserve the public fault and report the public operation name, such as
`submit` or `wait_completion`.

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

Accepted destruction scans backend state when validation or a debug callback
is enabled. Normal live children are rejected before this scan; diagnostics are
a safety net for internal, partial-initialization, and device-loss leftovers.
Callback messages use `WARNING`/`resource_lifetime` with operation
`destroy_device`; validation without a callback uses stderr. Debug names are
stored as truncating 63-byte copies.

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
    SwapchainReadiness readiness
    uint index
    bool suboptimal
    TextureLayout prior_layout

create_swapchain(Device*, Surface*, SwapchainDesc*) -> SwapchainHandle?
destroy_swapchain(Device*, SwapchainHandle) -> void?
resize_swapchain(Device*, SwapchainHandle, uint width, uint height) -> void?
get_swapchain_info(Device*, SwapchainHandle) -> SwapchainInfo?
acquire_next_image(Device*, SwapchainHandle) -> AcquiredImage?
present(Device*, AcquiredImage*, CompletionPoint render_completion) -> void?
get_present_mode_support(Device*, SwapchainHandle) -> PresentModeSupport?
```

The rendering submission passes `acquired.readiness` in `SubmitDesc`. Successful
submission consumes that readiness and returns the only completion point
accepted by `present` for the acquisition. Validation and retryable native
failure preserve readiness; device loss is terminal. Replays, stale
acquisitions, foreign devices, non-graphics queues, and unrelated completion
points fault before native mutation.

```c3
gpu::AcquiredImage acquired = gpu::acquire_next_image(&device, swapchain)!;
gpu::SubmitDesc submit = {
    .command_lists = lists[..],
    .readiness     = acquired.readiness,
};
gpu::CompletionPoint rendered = gpu::submit(graphics, &submit)!;
gpu::present(&device, &acquired, rendered)!;
```

A successful or enqueued present consumes `acquired`; validation failure and
native host/device allocation failure leave it retryable. `present` requires the
texture's tracked layout to be `PRESENT`. Resize rejects a pending acquisition;
destruction waits submitted render work and invalidates it.

`WAIT_TIMEOUT` from acquire leaves the swapchain unchanged. It can mean the
native acquire timed out or reported not ready, or that both private acquire
semaphore slots remain retired behind incomplete render completions. Out-of-date
requires resize; surface loss requires a new surface and swapchain. A
suboptimal image is valid and may be presented before resizing. Unsupported
requested present modes fall back to FIFO; query `get_present_mode_support` to choose explicitly.

`get_swapchain_info` returns the selected format, extent, image count, present
mode, and dormant state. Re-query after resize and rebuild format-dependent
pipelines when the format changes. A dormant swapchain reports `UNDEFINED`,
zero extent/count, and FIFO until resize succeeds.

`AcquiredImage.prior_layout` is the committed texture layout: `UNDEFINED` for a
newly wrapped image and `PRESENT` after the normal presentation cycle. Resize
stales prior borrowed texture handles. Destroy descriptors that reference
swapchain textures before resize.

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
    gpu::cmd_dispatch(
        commands: &commands,
        pipeline: pipeline,
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
