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
create_buffer
destroy_buffer
begin_commands
cmd_dispatch
cmd_barrier
```

Do not use project-owned OO-style constructors such as:

```text
Device.create
Buffer.create
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

Presentation is an explicit addition to the immutable strict request. It binds
queue selection and swapchain enablement to one surface.

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

strict_device_request()          -> DeviceRequest
request_presentation(DeviceRequest, Surface*) -> DeviceRequest?
supports_device_request(Adapter*, DeviceRequest*) -> DeviceRequestSupport?
create_device(Adapter*, DeviceRequest*) -> Device?
```

A live adapter-created device retains its runtime and reuses the runtime-owned
backend instance. Destroy the device before destroying that runtime.
`create_device_from_desc` remains as a transitional direct-device path for
existing headless consumers; it performs its own discovery and owns that discovery state.

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
    usz staging_arena_size
    usz readback_arena_size
    uint frames_in_flight
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
    bool timeline_semaphore
    bool shader_int64
    bool draw_indirect_count
    bool descriptor_buffer
    bool descriptor_indexing
    bool async_compute
    bool line_polygon_mode
    uint max_texture_descriptors
    uint max_sampler_descriptors
    uint max_color_attachments
    uint max_push_constant_size
    Vec3u max_compute_work_group_count
    uint max_draw_indirect_count
    ulong max_timeline_semaphore_value_difference
    usz min_uniform_alignment
    usz min_storage_alignment
    usz min_texel_buffer_alignment
    float max_sampler_anisotropy

Device                           (slot | generation | reserved)
get_device_backend(Device*)      -> BackendKind?
get_device_caps(Device*)         -> DeviceCaps?
```

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
create_device(Adapter*, DeviceRequest*) -> Device?
create_device_from_desc(DeviceDesc*) -> Device?    (transitional)
destroy_device(Device*) -> void?
```

Malformed or empty requests fault before adapter/backend work. A valid
unsupported request returns `supported = false` with the first unmet semantic
label; passing it to `create_device` faults `UNSUPPORTED_FEATURE` without
selecting another adapter. Duplicate capability contribution is rejected
transactionally by private composition helpers.

Multiple live devices may coexist. `Device` is a compact slot and generation
token; destroying it invalidates stale copies without affecting other devices.
Public device operations other than destruction take a short-lived atomic pin
before reading backend state. An operation that observes a closing device
faults `DEVICE_BUSY`.
`destroy_device` never waits for active operations: it restores the live state,
returns `DEVICE_BUSY`, and preserves the token and generation for retry. Backend
teardown begins only when no operation pins remain.

Frame tokens, command tokens, resource handles, descriptor indices, GPU
addresses/spans, and synchronization values are scoped to their owning device.
Passing one to another device is invalid; table- and index-backed values without
owner metadata may resolve a coincident resource instead of faulting.

### Handles

All handles are `bitstruct : ulong` (index | generation | reserved).

```text
BufferHandle
TextureHandle
PipelineHandle
ShaderHandle
SemaphoreHandle
SwapchainHandle
RecordingContextHandle
```

(Samplers have no handle — they are `SamplerIndex` heap indices.)

Invalid sentinels are module constants (`BUFFER_HANDLE_INVALID`, ...), all
zero-valued. `handle.is_valid()` checks that the token is nonzero; operations
validate current ownership and liveness.

A valid handle packs slot index and generation. Public code should not inspect the packed representation.

Handles, `TextureIndex`, `SamplerIndex`, `GpuAddress`, `GpuSpan`,
command tokens, and synchronization values are runtime-only and scoped to its
owning device. Do not persist, serialize, reconstruct, or pass
them across device or process lifetimes. `FrameToken` and `CommandList` embed a
copy of their owning `Device` token; they do not borrow caller variable storage.

### GPU addresses

```text
GpuAddress = ulong        (typedef — a distinct type, cast explicitly)

GpuSpan
    GpuAddress gpu
    void* cpu
    usz size
    BufferHandle buffer
    usz offset
    MemoryKind kind
```

Rules:

```text
GpuAddress zero is invalid for shader-visible memory.
GpuSpan.cpu may be null.
GpuSpan.gpu must be aligned for the intended shader layout.
GpuSpan.buffer identifies the backing buffer for barriers/copies/debug.
```

`GpuSpan` slicing is explicit:

```text
span.checked_subspan(offset, size)   -> GpuSpan?
span.unchecked_subspan(offset, size) -> GpuSpan
```

`checked_subspan` returns `INVALID_ARGUMENT` for zero size, if the requested
exact-sized range escapes its immediate parent, or if advancing the GPU
address, non-null CPU pointer, or backing-buffer offset would overflow.
`unchecked_subspan` performs only the metadata additions; callers must prove
the range and derived metadata are valid before using it.

## 4. Faults

Public operations use C3 optionals/faults. `faultdef` declares a flat list of globally-unique fault values (there is no braced/named fault group in C3 0.8.0); these live in `module gpu` and are referenced as `gpu::INVALID_HANDLE`, raised with the `~` suffix. `gpu/faults.c3` documents each fault at its definition; the table below maps them to the operations that raise them.

Descriptor, configuration, barrier, viewport, scissor, and label pointers must
be non-null unless the API explicitly documents null as a value (such as
`TextureViewDesc* view`). Null required input faults `INVALID_ARGUMENT` before a
backend call. Null or stale owner-token pointers fault `INVALID_HANDLE`.

| Fault | Fired by | Typical cause |
|---|---|---|
| `UNSUPPORTED_BACKEND` | `create_runtime`, `create_device_from_desc` | no Vulkan 1.3 driver / loader found no ICD |
| `UNSUPPORTED_FEATURE` | device creation, `create_runtime`, `create_texture`, `create_swapchain`, `create_graphics_pipeline`, sampler/aniso paths | validation layers not installed; presentation was not requested or is unsupported for the adapter and surface; missing optional or required device feature; unsupported image format or usage; adapter rejects a valid texture descriptor |
| `INVALID_ARGUMENT` | runtime adapter indexing; any create/upload/export; `GpuSpan.checked_subspan`; `submit`; `cmd_copy_buffer`/`cmd_fill_buffer`/buffer↔texture copies; `cmd_draw_indexed`(+indirect variants); `cmd_dispatch`/`cmd_draw`(+indirect variants); `cmd_set_viewport`/`cmd_set_scissor`; pipeline/shader creates; `cmd_texture_barrier`; `texture_transition`; `create_texture_descriptors` | null or malformed required input, zero size, undersized output buffer, out-of-range value, rectangle outside the active pass, or a subspan outside its parent/with overflowing metadata; mixed queue kinds in one submission; missing transfer/index usage flag or misaligned range; pipeline kind or shader stage mismatch; invalid texture use or `UNDEFINED` transition destination; `create_texture_descriptors`' `out_indices.len` does not equal `descs.len` |
| `INVALID_HANDLE` | runtime and adapter queries; `destroy_runtime`; `destroy_device`, `get_device_*`, any resource-handle-taking call, `cmd_*`, `end_commands`, `submit`, `alloc_frame_span`, `end_frame`, `resolve_readback` | zero, destroyed, or stale runtime, adapter, device, or resource token; consumed or stale command-list alias, `FrameToken`, or `ReadbackTicket` |
| `INVALID_RESOURCE_STATE` | swapchain lifecycle, `begin_frame`, `end_frame`, `destroy_recording_context`, `cmd_texture_barrier`, readback helpers | an acquired swapchain image is pending during resize or destruction; double begin or a frame boundary blocked by in-flight Tier S work; recording context still owns a live command record; or `old_layout` disagrees with the list's effective layout (its own pending transitions, else the tracked layout) |
| `OUT_OF_HOST_MEMORY` | creates | driver host-allocation failure |
| `OUT_OF_DEVICE_MEMORY` | buffer/texture creates | VMA/driver device-memory exhaustion |
| `DEVICE_LOST` | any Vulkan-backed operation | Vulkan explicitly returned `VK_ERROR_DEVICE_LOST`; unrecoverable |
| `DEVICE_BUSY` | public device operations; `destroy_device` | the operation observed a closing device, or destruction found an active host operation; retry without replacing the device token |
| `RESOURCE_IN_USE` | `destroy_runtime`, `destroy_surface`, `destroy_texture` | a runtime has a live surface or device, a surface has a live swapchain, or a live `TextureIndex` owns a texture. |
| `ARENA_FULL` | `alloc_frame_span`, staging/readback paths, persistent arena allocation | frame data or a persistent virtual block exceeded its configured capacity |
| `SLOT_TABLE_FULL` | runtime, device, and resource creates; `begin_commands` | the runtime or device registry, adapter token, handle table, or command-record table is at capacity |
| `DESCRIPTOR_HEAP_FULL` | descriptor pool creation/allocation, `create_texture_descriptor`, `create_texture_descriptors`, `create_sampler` | Vulkan descriptor-pool exhaustion or fragmentation, or capacity < live descriptors + same-frame retires (they recycle a frame later); `create_texture_descriptors` checks this as a pre-flight before creating anything, so a batch that would overflow leaves the heap untouched |
| `PIPELINE_CREATE_FAILED` | pipeline creates | driver rejected the state combination, shader, or compilation |
| `SHADER_INVALID` | `create_shader` | SPIR-V rejected by the driver |
| `SURFACE_LOST` | surface creation/query/enumeration, swapchain create/resize, acquire, present | native window or surface was destroyed or became unavailable; destroy the swapchain and create a new one from fresh native handles |
| `SWAPCHAIN_OUT_OF_DATE` | `create_swapchain`, `resize_swapchain`, `acquire_next_image`, `present` | swapchain no longer matches the surface; `resize_swapchain` and retry |
| `COMMAND_RECORDING_ERROR` | `cmd_*`, `end_commands`, `submit` | call outside its required recording state, duplicate command token in one submit batch, or token that is already being submitted |
| `READBACK_NOT_READY` | `resolve_readback` | ticket's timeline value not reached; `poll_readback` first |
| `WAIT_TIMEOUT` | `wait_semaphore`, `begin_frame`, `acquire_next_image` | bounded wait or transient image unavailability; retry without resizing |
| `BACKEND_ERROR` | any Vulkan-backed operation | unclassified or internal native failure; inspect backend diagnostics; does not imply device loss |

Backend-local Vulkan/VMA faults should not leak unless they carry useful public meaning. Map them to public faults and log backend details when validation/debug is enabled. `DEVICE_LOST` is reserved for an explicit native device-loss result; an unmapped native result becomes `BACKEND_ERROR`.

## 5. Memory API

### Memory kinds

```text
MemoryKind.FRAME_UPLOAD
MemoryKind.PERSISTENT_UPLOAD
MemoryKind.DEVICE
MemoryKind.READBACK
MemoryKind.STAGING
```

### Frame spans

The frame lifecycle is strict:

```text
IDLE --begin_frame(device)--> ACTIVE(token generation) --end_frame(token)--> IDLE

begin_frame(Device* device) -> FrameToken?
alloc_frame_span(FrameToken* frame, usz size, usz align) -> GpuSpan?
end_frame(FrameToken* frame) -> void?
```

`FrameToken` is a stack-only owner-bearing token no larger than 16 bytes. A
copy may allocate while its device-owned generation remains active. Its embedded
`Device` value does not borrow the variable passed to `begin_frame`. Successful
end clears the passed token and invalidates every alias; a consumed, malformed,
or stale token faults `INVALID_HANDLE`. `frame.is_valid()` checks whether the
token contains a generation so cleanup code can retry a failed end; operations
still validate liveness. Double begin faults
`INVALID_RESOURCE_STATE`. Rejections change no frame or arena state.

When end submission faults, `end_frame` returns the exact fault and preserves
the token, active generation, frame slot, retirement state, queue-use flags,
and prospective signal value. Retry with the same token; only a successful end
consumes it. Frame spans are transient and invalid after their frame arena
resets.

For fallible frame work, use the compile-time direct-call helper:

```c3
fn void? render_frame(gpu::FrameToken* frame, RenderState* state) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(frame, RootArgs::size, RootArgs::alignment)!;
    record_rendering(&frame.device, state, root_span)!;
}

gpu::FrameToken frame;
gpu::@with_frame(&frame, &device, render_frame, &state)!;
```

The worker must be a named optional-returning function whose first parameter is
`FrameToken*`; additional state is passed as ordinary arguments. The worker
must not end the frame itself; the helper owns the single end attempt. The
helper clears caller-owned token storage, begins the frame, calls the worker
directly, and attempts end exactly once after worker success or fault. Begin
failure calls neither worker nor end. If only the worker faults, its fault is
returned after end succeeds. If end faults, that exact fault takes precedence
even when the worker also faulted, and caller-owned `frame` remains live for
`end_frame(&frame)` retry. Callers needing both diagnostics should log the
worker fault before returning it. The helper performs no heap allocation,
runtime callback, virtual dispatch, or per-frame indirect call.

Frame-arena backing buffers are safe on the selected graphics and compute
families without a caller-supplied `shared_queues` flag. Concurrent sharing
does not provide execution or memory ordering; callers must still record the
required barriers and semaphore waits/signals.

Use cases:

```text
root structs
per-dispatch data
per-draw data
small per-frame tables
```

### Persistent spans

Persistent spans are suballocations from large VMA-backed buffers.

```text
PersistentAllocDesc
    usz size
    usz align
    BufferUsage usage
    MemoryKind memory_kind
    ZString debug_name

alloc_persistent_span(Device* device, PersistentAllocDesc* desc) -> GpuSpan?
free_persistent_span(Device* device, GpuSpan span) -> void?
```

Persistent backing is shared automatically across the exact selected graphics,
compute, and transfer queue families, remaining exclusive when those queues
alias one family. Per-span `usage.shared_queues` is accepted but does not change
ownership because sharing is a backing-buffer property; spans expose no queue-
family ownership API. Explicit buffers and textures consumed across distinct
families still require their own `shared_queues` flag.

Concurrent sharing does not provide visibility, execution ordering, completion,
or lifetime management. Callers must retain the required flushes, barriers,
semaphore dependencies, and completion waits, and may call
`free_persistent_span` only after all work referencing the span retires. Spans
are scoped to their owning `Device` and cannot be passed to another device.

### Explicit buffers

`BufferUsage` is a bitstruct of bool flags, composed by field-set
(`{ .storage, .addressable }`), not OR-combined enum values.

```text
bitstruct BufferUsage : uint
    bool transfer_src : 0
    bool transfer_dst : 1
    bool uniform      : 2
    bool storage      : 3
    bool addressable  : 4
    bool indirect     : 5
    bool index        : 6
    bool vertex       : 7

BufferDesc
    usz size
    BufferUsage usage
    MemoryKind memory_kind
    ZString debug_name

create_buffer(Device* device, BufferDesc* desc) -> BufferHandle?
destroy_buffer(Device* device, BufferHandle buffer) -> void?
get_buffer_span(Device* device, BufferHandle buffer) -> GpuSpan?
get_buffer_address(Device* device, BufferHandle buffer) -> GpuAddress?
flush_buffer(Device* device, BufferHandle buffer, usz offset, usz size) -> void?
invalidate_buffer(Device* device, BufferHandle buffer, usz offset, usz size) -> void?
```

`get_buffer_address` faults if the buffer was not created with the
`addressable` usage flag set. There is no map/unmap pair — mappable memory
kinds are persistently mapped and exposed through `GpuSpan.cpu`
(`get_buffer_span`), paired with `flush_buffer`/`invalidate_buffer` for
non-coherent ranges.

Upload/readback helpers round out the buffer path (signatures in the
generated reference): blocking `upload_buffer_data` / `upload_texture_data` /
`readback_buffer_data` / `readback_texture_data` (all take the resource's
current Stage/Hazard[/TextureLayout] and restore it), recorded
`cmd_upload_buffer` / `cmd_upload_texture`, and the non-blocking ticket flow
`cmd_readback_buffer`/`cmd_readback_texture` → `poll_readback` →
`resolve_readback` (see docs/memory.md §12). Frame lifecycle is
`begin_frame`/`end_frame` around each frame's work (or `@with_frame` for
fallible named-worker scopes); memory introspection is
`get_memory_stats` / `build_memory_report` / `get_persistent_stats`.

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
    bool shared_queues : 6

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
create_texture(Device* device, TextureDesc* desc) -> TextureHandle?
destroy_texture(Device* device, TextureHandle texture) -> void?
create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) -> TextureIndex?
destroy_texture_descriptor(Device* device, TextureIndex index) -> void?
create_texture_descriptors(Device* device, TextureDescriptorDesc[] descs, TextureIndex[] out_indices) -> void?
```

`get_texture_format_support` reports library-creatable support, not every raw Vulkan capability. Each usage bit comes from the same exact 2D optimal-tiling query used by creation, but the bits are independent; use `supports_texture_desc` for a usage combination. The backend profile masks every dimension except 2D and every sample count except one. Per-format usages and linear filtering remain adapter-dependent; D24S8 reports empty support until the rendering path supports it end to end.

`supports_texture_desc` checks the exact optimal-tiling format, combined usage (excluding the queue-sharing policy flag), normalized extent, mip and layer counts, and the required single-sample image properties without allocating. A false result caused by malformed or backend-unsupported input corresponds to `INVALID_ARGUMENT` at creation; a structurally valid descriptor rejected by the adapter corresponds to `UNSUPPORTED_FEATURE`. Memory exhaustion can still make creation fail after a true capability result.

`TextureHandle` owns the image. `TextureIndex` is a descriptor heap entry used by shaders.

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

create_sampler(Device* device, SamplerDesc* desc) -> SamplerIndex?
destroy_sampler(Device* device, SamplerIndex index) -> void?
```

Samplers are shader-visible indices. LOD values must be finite and `min_lod`
must not exceed `max_lod`. Anisotropy requires the reported device capability.

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

Thread-safety is tiered per entry point — see `docs/threading.md` for the
full table, lock order, and disciplines. Summary: resource creation and the
transfer helpers are thread-safe; frame lifecycle, submit/present, and
swapchain operations are externally synchronized; recording is confined —
`begin_commands(device, queue, ctx)` takes a `RecordingContextHandle`
(default `{}` = the device's built-in context), and each context records from
one thread at a time. Create one context per recording thread with
`create_recording_context`.

### Command lifecycle

```text
begin_commands(Device* device, QueueKind queue, RecordingContextHandle ctx = {}) -> CommandList?
end_commands(CommandList* commands) -> void?
submit(Device* device, SubmitDesc* desc) -> void?
wait_queue_idle(Device* device, QueueKind queue) -> void?

SubmitDesc
    CommandList[] command_lists
    SemaphoreWait[] waits           ({ semaphore, value, stage })
    SemaphoreSignal[] signals
    SwapchainHandle swapchain       (present-linked submits)

create_semaphore / destroy_semaphore / wait_semaphore   (timeline; SemaphoreValue = distinct ulong)
create_recording_context / destroy_recording_context    (one per worker thread; docs/threading.md)
```

`CommandList` is a small owner-bearing token; mutable lifecycle, binding cache,
pending texture layouts, and the backend command buffer are device-owned. Copies
alias the same record, and the embedded `Device` value does not borrow the
variable passed to `begin_commands`. A successful `submit` consumes every alias,
so later use faults `INVALID_HANDLE`. A submit batch is validated as a
transaction: duplicate tokens fault `COMMAND_RECORDING_ERROR`, mixed queue kinds
fault `INVALID_ARGUMENT`, and failures before a successful Vulkan queue call
leave all valid tokens executable. Device-token copies for the sole live device
are accepted; stale owners fault `INVALID_HANDLE`. An unsubmitted token becomes
stale when its frame-slot command pool resets.

Timeline signal values must be greater than the semaphore counter when they
execute. The caller orders pending signals across queues and keeps waits and
signals within `DeviceCaps.max_timeline_semaphore_value_difference` of current
and pending values. Waits may target future values.

`QueueKind.COMPUTE` routes to a real compute queue when
`DeviceCaps.async_compute` is true; resources used by both GRAPHICS and
COMPUTE must then carry the `shared_queues` usage flag (concurrent sharing;
no-op on single-queue devices). See docs/limitations.md.

Transfer/render helper descriptors (`BufferCopyDesc`, `BufferTextureCopyDesc`,
`TextureBufferCopyDesc`, `TextureUploadDesc`, `ClearColor`,
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

Argument spans must come from a buffer with `indirect` usage, 4-byte aligned,
with `draw_count` (or `max_draw_count`) times the tight argument size inside
the span. One vertex/fragment root pair applies to every draw in a
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
cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) -> void?
cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) -> void?
cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) -> void?
cmd_fill_buffer(CommandList* commands, BufferHandle buffer, usz offset, usz size, uint value) -> void?
```

### Readback tickets

Non-blocking readback: record now, resolve later.

```text
ReadbackTicket                   (generation-checked token)

cmd_readback_buffer(CommandList* commands, BufferHandle src, usz offset, usz size) -> ReadbackTicket?
cmd_readback_texture(CommandList* commands, TextureHandle src, uint mip) -> ReadbackTicket?
poll_readback(Device* device, ReadbackTicket* ticket) -> bool?
resolve_readback(Device* device, ReadbackTicket* ticket, char[] dest) -> void?
```

Recording copies the source into readback memory inside the caller's command
list and inserts only the internal transfer→host barrier on the destination;
source-side ordering (and, for textures, the `TRANSFER_SRC` layout) is the
caller's responsibility.

Readiness is frame-boundary granular: a ticket's copy is a Tier C recording
that always retires on the frame timeline, which only `end_frame` signals
(blocking helpers signal a separate helper timeline and never retire a
ticket, including one an unrelated helper happens to finish while it sits
unsubmitted — see docs/threading.md §Helper timeline). A ticket recorded in
frame N resolves after frame N ends; applications that never run the frame
loop never signal tickets.

Tickets hold a pinned readback-arena range until resolved — an unresolved
ticket blocks arena reclamation behind it (FIFO); when the arena is full,
tickets fall back to dedicated buffers destroyed at resolve. `resolve_readback`
faults `READBACK_NOT_READY` before the timeline signals, and
`INVALID_ARGUMENT` on a consumed token or a `dest` smaller than the copied
range. A stale alias faults `INVALID_HANDLE`. Each ticket resolves exactly
once; device teardown releases unresolved tickets.
An invalid or closing device faults `INVALID_HANDLE` or `DEVICE_BUSY` before the
ticket is inspected.

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
    BufferHandle buffer
    usz offset
    usz size
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

`Stage.NONE` is an empty execution scope. Use it only for a barrier side that
has no pipeline work to wait for. It is legal for semaphore operations but does
not order pipeline work. Explicit `Stage.PRESENT` and `Hazard.PRESENT_READ`
remain the legacy broad spelling and map to all commands and memory read.
They are retained for raw-barrier and semaphore compatibility, not used by the
`TextureUse.PRESENT` preset. The preset uses `COLOR_ATTACHMENT` with no access
scope so its barriers chain with the WSI wait and signal stages.

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

At the presentation boundary, both barrier sides use `COLOR_ATTACHMENT` so
they chain with the coupled submission's color-attachment-output semaphore
wait and signal. Access remains asymmetric by composition:
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

Representative tranche A public-contract coverage includes texture creation
and tracked layout barriers, buffer-barrier ranges, persistent-span allocation,
shader reflection/entry-point validation, and pipeline shader-stage validation.
Backend failures from queue submission and idle waits use the same callback
with `operation = "submit"` or `"wait_queue_idle"`, the unchanged public
fault, and the native result text.

Representative tranche B1 coverage adds command-list state, copy/fill and
render-attachment validation; upload and readback planning; descriptor and
sampler operations; and pipeline-cache I/O. Resource identity is included only
after the corresponding public handle has been resolved successfully. The B1
set is representative rather than exhaustive; later coverage work audits the
remaining command, descriptor, queue, and WSI rejection sites.

Frame and persistent-memory coverage reports double-begin and quiescence
violations, stale frame tokens, invalid alignment before zero size, arena
exhaustion, non-timeout frame-wait failures and retirement queries,
end-frame signal failures, persistent virtual-allocation exhaustion, and
invalid or repeated frees. Expected/retryable frame-wait `WAIT_TIMEOUT` outcomes remain
silent; other diagnostics preserve `ARENA_FULL` and retry state and never
claim public identity for frame tokens or persistent spans.

```text
cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) -> void?
cmd_end_label(CommandList* commands) -> void?
```

Labels group work for capture tools; they are valid while recording,
including inside render passes, and silently succeed when debug-utils is
absent. Balance is the caller's responsibility.

`destroy_device` scans for leaks when validation is enabled or a debug
callback is configured. With a callback, each leak is a synchronous
`WARNING`/`resource_lifetime` message with `operation = "destroy_device"`,
resource identity where available, the stored debug name, and backend detail.
Without a callback, validation-enabled teardown retains the stderr fallback.
Coverage includes buffers, textures, pipeline cache entries, shaders,
semaphores, recording contexts, persistent spans, and descriptor slots. Debug
names are stored as truncating 63-byte copies — no lifetime requirement on the
caller's string.

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

AcquiredImage
    TextureHandle texture   (frame-transient — resize stales it)
    uint index
    bool suboptimal
    TextureLayout prior_layout

PresentDesc
    SwapchainHandle swapchain

bitstruct PresentModeSupport : uint
    bool fifo
    bool immediate
    bool mailbox

create_swapchain(Device* device, Surface* surface, SwapchainDesc* desc) -> SwapchainHandle?
destroy_swapchain(Device* device, SwapchainHandle swapchain) -> void?
resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) -> void?
get_swapchain_info(Device* device, SwapchainHandle swapchain) -> SwapchainInfo?
acquire_next_image(Device* device, SwapchainHandle swapchain) -> AcquiredImage?
present(Device* device, PresentDesc* desc) -> void?
get_present_mode_support(Device* device, SwapchainHandle swapchain) -> PresentModeSupport?
```

An unsupported requested mode falls back to FIFO silently at creation; query
`get_present_mode_support` to choose deliberately (`present_mode_explorer`
sample). State-machine contracts: acquiring while an acquire is pending
faults INVALID_RESOURCE_STATE; present enforces the PRESENT tracked layout;
a failed resize parks the swapchain dormant (next acquire reports
SWAPCHAIN_OUT_OF_DATE).

Configured diagnostics preserve the public WSI operation and fault, report
exact lifecycle or descriptor context, and include the raw signed VkResult for
native failures. Expected recovery outcomes are silent: acquire `WAIT_TIMEOUT`
and dormant-swapchain `SWAPCHAIN_OUT_OF_DATE` return their faults without a
callback. Handle identity is present only after a live swapchain resolves.
Each swapchain is scoped to its creating `Device`; multiple devices may own
independent swapchains.

WSI recovery is fault-specific:

| Outcome | Meaning | Recovery |
|---|---|---|
| `WAIT_TIMEOUT` from acquire | no image was available during the bounded wait | retry acquire on the same swapchain; do not resize |
| `SWAPCHAIN_OUT_OF_DATE` | swapchain no longer matches the surface | call `resize_swapchain`, then retry |
| `SURFACE_LOST` | platform surface is no longer usable | destroy the swapchain, refresh native surface handles as needed, and create a new swapchain |
| acquired image with `suboptimal = true` | image is valid but presentation properties changed | finish the frame and resize when convenient |

`get_swapchain_info` reports the selected format, clamped extent,
driver-returned image count, and selected present mode (including FIFO
fallback), not the requested values. Re-query after every successful
`resize_swapchain`; if the format changed, rebuild format-dependent graphics
pipelines. The snapshot is published coherently after a complete build. A
zero extent or failed rebuild publishes the dormant sentinel:
`format = UNDEFINED`, zero width/height/image count, `present_mode = FIFO`,
and `dormant = true`. The handle remains valid and queryable while dormant.

`AcquiredImage.prior_layout` is the image's committed tracked layout at
acquire time. It is `UNDEFINED` for a newly wrapped image and `PRESENT` after
that image's submitted present transition, so callers can use it directly as
the first barrier's `old_layout` without a per-image seen table.

`resize_swapchain` (and swapchain/device teardown) release the old wrapped
swapchain textures directly, bypassing `destroy_texture` — so they never hit
the `RESOURCE_IN_USE` check (§4 Faults). A `TextureIndex` descriptor created
against a swapchain texture does **not** block a resize, and is not faulted
or freed by it: the descriptor is left dangling once the backing image is
gone. Destroy any descriptors on the current swapchain textures before
calling `resize_swapchain`.

Resize and destruction reject a pending acquire with
`INVALID_RESOURCE_STATE`. Otherwise, they wait for both graphics and
presentation work before releasing swapchain resources.

Each acquired image couples to exactly one GRAPHICS `submit` before present:
a second coupled submit or a present without a consumed submit faults
INVALID_RESOURCE_STATE (a failed submit leaves the acquire retryable), and
coupling a COMPUTE/TRANSFER submit faults INVALID_ARGUMENT. Wrapped swapchain
textures carry only the usage bits the surface actually granted, so a
transfer-src copy faults INVALID_ARGUMENT where TRANSFER_SRC was never
supported.

SDL helper functions should live in samples or an optional helper module, not in the core API.

## 11. Example: root-pointer compute

Pseudo-code:

```c3
import gpu;

struct RootArgs {
    gpu::GpuAddress input;
    gpu::GpuAddress output;
    uint count;
    uint _pad0, _pad1, _pad2;
}

struct ComputeWork {
    gpu::Device*       device;
    gpu::BufferHandle  input;
    gpu::PipelineHandle pipeline;
}

fn void? run_compute() {
    gpu::RuntimeDesc runtime_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "root_pointer_compute",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::Adapter adapter = adapters.get(0)!;
    gpu::DeviceRequest request = gpu::strict_device_request();
    gpu::DeviceRequestSupport support =
        gpu::supports_device_request(&adapter, &request)!;
    if (!support.supported) return gpu::UNSUPPORTED_FEATURE~;

    gpu::Device device = gpu::create_device(&adapter, &request)!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc input_desc = {
        .size = 4096,
        .usage = { .storage, .addressable, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name = "input",
    };

    gpu::BufferHandle input = gpu::create_buffer(&device, &input_desc)!;
    defer gpu::destroy_buffer(&device, input)!!;

    ComputeWork work = { .device = &device, .input = input, .pipeline = pipeline };
    gpu::FrameToken frame;
    gpu::@with_frame(&frame, &device, record_compute, &work)!;
}

fn void? record_compute(gpu::FrameToken* frame, ComputeWork* work) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(frame, RootArgs::size, RootArgs::alignment)!;
    RootArgs* root = (RootArgs*)root_span.cpu;
    root.input = gpu::get_buffer_address(work.device, work.input)!;
    root.count = 1024;
    gpu::CommandList commands = gpu::begin_commands(work.device, gpu::QueueKind.COMPUTE)!;
    gpu::cmd_dispatch(
        commands: &commands,
        pipeline: work.pipeline,
        root:     root_span.gpu,
        groups:   { 16, 1, 1 },
    )!;
    gpu::end_commands(&commands)!;
}
```

Exact C3 syntax should be verified during implementation against C3 0.8.0.

## 12. API acceptance criteria

The public API is acceptable when:

```text
no public signature exposes vk::, vma::, or sdl:: types
all fallible operations return optionals/faults
all resources have explicit destruction or frame ownership
root-pointer compute can be written without descriptor-set concepts
texture sampling can be written with TextureIndex and SamplerIndex
barriers are explicit and expressive enough for all samples
headless tests do not depend on SDL3
windowed samples depend on sdl3 only in sample project files
```
