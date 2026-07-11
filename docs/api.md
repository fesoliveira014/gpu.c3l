# gpu.c3l Public API

This document is the curated tour of the public API: shape, idiom, and the
examples worth reading. It is deliberately not exhaustive — the generated
reference (`api-reference` CI artifact, built by `c3c docgen` from the source
doc-strings; GitHub Pages publishing tracked in #29) covers every public
symbol. When this document and source disagree, source wins; the doc-strings
are the contract.

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

Windowed samples may also import:

```c3
import sdl;
```

but core `gpu` declarations should not mention `sdl::Window`, `vk::Device`, or `vma::Allocation`.

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

### Backend and device

```text
BackendKind
    VULKAN

DescriptorHeapMode
    AUTO                    (indexing preferred; buffer where proven)
    DESCRIPTOR_BUFFER
    DESCRIPTOR_INDEXING

DeviceDesc
    BackendKind backend
    bool enable_validation
    bool enable_debug_names
    bool enable_presentation
    DescriptorHeapMode descriptor_heap_mode
    uint texture_descriptor_capacity      (0 = default; docs/limitations.md)
    uint sampler_descriptor_capacity
    uint texture_capacity
    usz staging_arena_size
    usz readback_arena_size
    uint frames_in_flight
    char[] pipeline_cache_data            (warm-start blob; §8)
    ZString application_name

DeviceCaps
    bool buffer_device_address
    bool synchronization2
    bool dynamic_rendering
    bool timeline_semaphore
    bool shader_int64
    bool draw_indirect_count
    bool descriptor_buffer
    bool descriptor_indexing
    bool async_compute
    uint max_texture_descriptors
    uint max_sampler_descriptors
    uint max_push_constant_size
    usz min_uniform_alignment
    usz min_storage_alignment
    usz min_texel_buffer_alignment
    float max_sampler_anisotropy

Device
    BackendKind backend
    DeviceCaps caps
    BackendVTable* vtable
    void* backend_state
```

Descriptor capacities are exact creation requests, not clampable upper bounds.
For descriptor indexing, texture capacity contributes once to the sampled-image
binding and once to the storage-image binding. Per-stage resource usage is
`2 * texture_descriptor_capacity`; plain sampler descriptors do not count toward
that limit. All-pools usage is `2 * texture_descriptor_capacity +
sampler_descriptor_capacity`. `create_device` returns `INVALID_ARGUMENT` when a
requested capacity exceeds a per-type or aggregate device limit. On success,
`DeviceCaps.max_texture_descriptors` and
`DeviceCaps.max_sampler_descriptors` report the capacities of the created heap.

Creation:

```text
create_device(DeviceDesc* desc) -> Device?
destroy_device(Device* device) -> void?
```

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
zero-valued; `handle.is_valid()` answers liveness.

A valid handle packs slot index and generation. Public code should not inspect the packed representation.

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

## 4. Faults

Public operations use C3 optionals/faults. `faultdef` declares a flat list of globally-unique fault values (there is no braced/named fault group in C3 0.8.0); these live in `module gpu` and are referenced as `gpu::INVALID_HANDLE`, raised with the `~` suffix. `faults.c3` documents each fault at its definition; the table below maps them to the operations that raise them.

| Fault | Fired by | Typical cause |
|---|---|---|
| `UNSUPPORTED_BACKEND` | `create_device` | no Vulkan 1.3 driver / loader found no ICD |
| `UNSUPPORTED_FEATURE` | `create_device`, `create_swapchain`, sampler/aniso paths | validation layers not installed; presentation off; missing device feature |
| `INVALID_ARGUMENT` | any create/upload/export; `submit`; `end_commands`; `cmd_copy_buffer`/`cmd_fill_buffer`/buffer↔texture copies; `cmd_draw_indexed`(+indirect variants); `cmd_dispatch`/`cmd_draw`(+indirect variants); pipeline/shader creates; `cmd_texture_barrier`; `create_texture_descriptors` | malformed descriptor, zero size, undersized output buffer, out-of-range value; mixed-queue-kind or cross-device command submission/finalization; missing transfer/index usage flag or misaligned range; pipeline kind or shader stage mismatch; `create_texture_descriptors`' `out_indices.len` does not equal `descs.len` |
| `INVALID_HANDLE` | any handle-taking call, `cmd_*`, `end_commands`, `submit` | use after destroy (generation mismatch), never-live handle, consumed command-list alias, or abandoned command token after its frame-slot pool resets |
| `INVALID_RESOURCE_STATE` | `destroy_recording_context`, `cmd_texture_barrier`, readback helpers | recording context still owns a live command record, or `old_layout` disagrees with the list's effective layout (its own pending transitions, else the tracked layout) |
| `OUT_OF_HOST_MEMORY` | creates | driver host-allocation failure |
| `OUT_OF_DEVICE_MEMORY` | buffer/texture creates | VMA/driver device-memory exhaustion |
| `DEVICE_LOST` | submits, `wait_semaphore`/`begin_frame` on device loss | driver reported device loss; unrecoverable |
| `RESOURCE_IN_USE` | `destroy_texture` | a live `TextureIndex` descriptor still owns the texture; destroy the descriptor first (gpu.c3l#81). Frames-in-flight destroys — including a resource an off-frame `submit` referenced — are unaffected: those are handled by deferred backend release instead (gpu.c3l#44, gpu.c3l#80) |
| `ARENA_FULL` | `alloc_frame_span`, staging/readback paths | per-frame data outgrew the arena (sizing knobs: gpu.c3l#28) |
| `SLOT_TABLE_FULL` | creates, `begin_commands` | handle or command-record table at capacity; textures scale via `DeviceDesc.texture_capacity` |
| `DESCRIPTOR_HEAP_FULL` | `create_texture_descriptor`, `create_texture_descriptors`, `create_sampler` | capacity < live descriptors + same-frame retires (they recycle a frame later); `create_texture_descriptors` checks this as a pre-flight before creating anything, so a batch that would overflow leaves the heap untouched |
| `PIPELINE_CREATE_FAILED` | pipeline creates | driver rejected the state combination or failed compiling |
| `SHADER_INVALID` | `create_shader` | SPIR-V rejected by the driver |
| `SURFACE_LOST` | acquire/present | window/surface destroyed mid-frame |
| `SWAPCHAIN_OUT_OF_DATE` | `acquire_next_image`, `present` | surface changed (resize); `resize_swapchain` and retry |
| `COMMAND_RECORDING_ERROR` | `cmd_*`, `end_commands`, `submit` | call outside its required recording state, duplicate command token in one submit batch, or token that is already being submitted |
| `READBACK_NOT_READY` | `resolve_readback` | ticket's timeline value not reached; `poll_readback` first |
| `WAIT_TIMEOUT` | `wait_semaphore`, `begin_frame` | bounded host wait elapsed before the timeline reached its target value; safe to retry |

Backend-local Vulkan/VMA faults should not leak unless they carry useful public meaning. Map them to public faults and log backend details when validation/debug is enabled.

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

Frame spans are transient and invalid after the frame arena resets.

```text
alloc_frame_span(Device* device, usz size, usz align) -> GpuSpan?
```

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
`begin_frame`/`end_frame` around each frame's work; memory introspection is
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
D24_UNORM_S8_UINT   (backend-unsupported: creation faults INVALID_ARGUMENT)
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

TextureDimension
    TEX_1D   (backend-unsupported — faults at creation)
    TEX_2D
    TEX_3D   (backend-unsupported — faults at creation)
    CUBE     (backend-unsupported — faults at creation)

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
create_texture(Device* device, TextureDesc* desc) -> TextureHandle?
destroy_texture(Device* device, TextureHandle texture) -> void?
create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) -> TextureIndex?
destroy_texture_descriptor(Device* device, TextureIndex index) -> void?
create_texture_descriptors(Device* device, TextureDescriptorDesc[] descs, TextureIndex[] out_indices) -> void?
```

`TextureHandle` owns the image. `TextureIndex` is a descriptor heap entry used by shaders.

`create_texture_descriptors` batch-creates N descriptors under one lock hold, ending in one accumulated descriptor-set update in indexing mode (buffer mode writes per-item, already a mapped-memory store). `out_indices.len` must equal `descs.len` (`INVALID_ARGUMENT` otherwise); an empty `descs` is a no-op success. A zero-initialized `TextureDescriptorDesc.view` collapses to the default view, same as a null `view` to `create_texture_descriptor`. All-or-nothing: a mid-batch fault rolls back every index already created in the batch — release each returned index individually with `destroy_texture_descriptor`.

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

Samplers are shader-visible indices. The backend may store immutable sampler descriptors or a sampler heap.

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
end_commands(Device* device, CommandList* commands) -> void?
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
alias the same record. A successful `submit` consumes every alias, so later use
faults `INVALID_HANDLE`. A submit batch is validated as a transaction: duplicate
tokens fault `COMMAND_RECORDING_ERROR`, cross-device tokens fault
`INVALID_ARGUMENT`, and failures before a successful Vulkan queue call leave all valid
tokens executable. An unsubmitted token becomes stale when its frame-slot command
pool resets.

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

A pass names at least one color target or a depth target; depth-only passes
(the shadow-map shape) are valid. A depth target needs `depth_attach` usage
and the `DEPTH_STENCIL` tracked layout. `D32_FLOAT` is the only supported
depth format; pipelines name it in `GraphicsPipelineDesc.depth_format`.
Attachment extents matching the pass dimensions are the caller's
responsibility.

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
`docs/shader_abi.md`). Ordering between argument writes and indirect
consumption is the caller's barrier (`INDIRECT_COMMAND` / `INDIRECT_READ`).

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
ReadbackTicket
    GpuSpan span
    SemaphoreValue value
    (backend bookkeeping fields)

cmd_readback_buffer(CommandList* commands, BufferHandle src, usz offset, usz size) -> ReadbackTicket?
cmd_readback_texture(CommandList* commands, TextureHandle src, uint mip) -> ReadbackTicket?
poll_readback(Device* device, ReadbackTicket* ticket) -> bool
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
`INVALID_ARGUMENT` on an already-resolved ticket or a `dest` smaller than the
span. Each ticket resolves exactly once.

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

`TextureLayout` values: UNDEFINED, GENERAL, COLOR_ATTACHMENT, DEPTH_STENCIL,
SHADER_READ, TRANSFER_SRC, TRANSFER_DST, PRESENT. `old_layout` must match the
list's effective layout — its own pending transitions if it has recorded any
for the texture, else the tracked layout — or the barrier faults
`INVALID_RESOURCE_STATE`. A recorded transition is staged on the list and
only commits onto tracked state when the list submits; a list that never
submits leaves tracked state untouched.

No command helper should silently insert barriers for a later use.

### Debug labels and leak reporting

```text
cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) -> void?
cmd_end_label(CommandList* commands) -> void?
```

Labels group work for capture tools; they are valid while recording,
including inside render passes, and silently succeed when debug-utils is
absent. Balance is the caller's responsibility.

With validation enabled, `destroy_device` reports every leaked resource to
stderr before sweeping it: buffers by debug name and handle, textures by
name and allocation size, pipeline cache entries by alias count, shaders,
semaphores, and recording contexts by index, plus live persistent-span
counts. Debug names are stored as truncating 63-byte copies — no lifetime
requirement on the caller's string.

## 10. Swapchain API

Core WSI types should be platform-neutral.

```text
SurfaceDesc
    PlatformKind platform
    void* native_display
    void* native_window

PlatformKind: WAYLAND, X11, WIN32
PresentMode: FIFO, IMMEDIATE, MAILBOX

SwapchainDesc
    uint width
    uint height
    Format preferred_format
    PresentMode present_mode
    uint image_count
    bool srgb
    ZString debug_name

AcquiredImage
    TextureHandle texture   (frame-transient — resize stales it)
    uint index
    bool suboptimal

PresentDesc
    SwapchainHandle swapchain

bitstruct PresentModeSupport : uint
    bool fifo
    bool immediate
    bool mailbox

create_swapchain(Device* device, SurfaceDesc* surface, SwapchainDesc* desc) -> SwapchainHandle?
destroy_swapchain(Device* device, SwapchainHandle swapchain) -> void?
resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) -> void?
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

`resize_swapchain` (and swapchain/device teardown) release the old wrapped
swapchain textures directly, bypassing `destroy_texture` — so they never hit
the `RESOURCE_IN_USE` check (§4 Faults). A `TextureIndex` descriptor created
against a swapchain texture does **not** block a resize, and is not faulted
or freed by it: the descriptor is left dangling once the backing image is
gone. Destroy any descriptors on the current swapchain textures before
calling `resize_swapchain`.

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

fn void? run_compute() {
    gpu::DeviceDesc device_desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .enable_debug_names = true,
        .frames_in_flight = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.AUTO,
        .application_name = "root_pointer_compute",
    };

    gpu::Device device = gpu::create_device(&device_desc)!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc input_desc = {
        .size = 4096,
        .usage = { .storage, .addressable, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name = "input",
    };

    gpu::BufferHandle input = gpu::create_buffer(&device, &input_desc)!;
    defer gpu::destroy_buffer(&device, input)!!;

    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, RootArgs::size, RootArgs::alignment)!;
    RootArgs* root = (RootArgs*)root_span.cpu;
    root.input = gpu::get_buffer_address(&device, input)!;
    root.count = 1024;

    gpu::CommandList commands = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!;
    gpu::cmd_dispatch(
        commands: &commands,
        pipeline: pipeline,
        root:     root_span.gpu,
        groups:   { 16, 1, 1 },
    )!;
    gpu::end_commands(&device, &commands)!;
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
