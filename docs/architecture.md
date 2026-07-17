# gpu.c3l Architecture

## 1. Purpose

`gpu.c3l` is a C3 library that exposes a direct GPU programming model suitable for modern explicit rendering and compute workloads. It is not a renderer, render graph, material system, asset system, or platform abstraction layer.

The API uses root pointers, bindless heap indices, explicit barriers, and
backend-neutral GPU allocation. Its design follows Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api);
the [strict GPU architecture](strict_gpu_profile.md) records the broader design.

The API centers on:

```text
GpuAllocation   -> owning generic GPU storage
GpuSpan         -> non-owning identity and range
GpuAddress      -> shader-visible data address
TextureIndex    -> shader-visible texture heap index
SamplerIndex    -> shader-visible sampler heap index
```

Draw and dispatch commands pass root GPU addresses. Shaders follow pointers to structured data and use texture/sampler indices for image access. Barriers are explicit. Resource lifetimes are explicit.

## 2. Layer model

```text
application / engine / sample
        |
        v
gpu public API, module gpu
        |
        v
backend dispatch layer
        |
        v
gpu::vk Vulkan backend
        |
        +--> vk.c3l         -> Vulkan API calls
        +--> vma.c3l        -> Vulkan memory allocation
        +--> spvreflect.c3l -> SPIR-V shader reflection
```

The public API does not expose Vulkan or VMA types. SDL3 integration belongs to the separate `gpu.c3l-samples` repository and is not a backend dependency.

## 3. Package structure

`gpu.c3l` uses a C3 library package layout. Shipped library source files live under one `gpu/` subtree.

```text
gpu.c3l/
├── abi/                     shader ABI schemas
├── manifest.json
├── gpu/
│   ├── gpu.c3i
│   ├── gpu.c3
│   ├── types.c3
│   ├── faults.c3
│   ├── runtime.c3
│   ├── adapter.c3
│   ├── caps.c3
│   ├── device.c3
│   ├── queue.c3
│   ├── memory.c3
│   ├── buffer.c3
│   ├── texture.c3
│   ├── descriptor_heap.c3
│   ├── shader_abi.c3
│   ├── pipeline.c3
│   ├── command.c3
│   ├── sync.c3
│   ├── surface.c3
│   ├── surface/
│   │   ├── win32/surface.c3
│   │   ├── wayland/surface.c3
│   │   └── x11/surface.c3
│   ├── swapchain.c3
│   ├── debug.c3
│   └── vk/
│       └── *.c3
├── include/
│   └── shaders/        published shader-side ABI includes only (no application shaders)
├── lib/                     vendored C3 bindings
├── scripts/                 ABI, shader, and documentation checks
├── test/
├── tools/
└── docs/
```

### Library files

Files under `gpu/` declare:

```c3
module gpu;
```

Backend files under `gpu/vk/` declare:

```c3
module gpu::vk @private;
```

The public `gpu` module explicitly imports this implementation module with a
visibility override. White-box tests do the same; consumers should not depend on
backend declarations.

Samples are standalone consumers and may declare their own sample modules.

### Shader ownership

The library ships **no application shaders**. Shader entry points are written and owned by the consuming project. The only shader-side artifacts the library publishes are ABI includes under `include/shaders/` (descriptor-heap helpers and generated ABI structs/offsets) that a consumer's shaders `#include`. Samples and tests own their shaders inside their own trees, because they are consumers like any other.

## 4. Public object model

### Runtime and adapters

`Runtime` owns backend discovery, diagnostics, and borrowed adapters. Creating one is the first operation that may initialize native backend state. Multiple runtimes may coexist.

Public shape:

```text
RuntimeDesc
    BackendKind backend
    bool enable_validation
    ZString application_name
    DebugMessageCallback debug_callback
    void* debug_user_data

create_runtime(RuntimeDesc*)       -> Runtime?
enumerate_adapters(Runtime*)       -> AdapterList?
AdapterList.get(uint)              -> Adapter?
get_adapter_info(Adapter*)         -> AdapterInfo?
get_adapter_diagnostics(Adapter*)  -> AdapterDiagnostics?
destroy_runtime(Runtime*)          -> void?
```

`AdapterList` is an allocation-free view. Its adapters and the read-only strings in adapter query results are borrowed until their runtime is destroyed. Destroying a runtime consumes its token, invalidates its adapter views and handles, and returns `RESOURCE_IN_USE` while a dependent surface or device is live.

Canonical `create_device(Adapter*, DeviceRequest*)` uses the exact borrowed adapter, retains its runtime, and reuses the runtime-owned backend instance. `supports_device_request` is read-only and does not enable state. The transitional `create_device_from_desc(DeviceDesc*)` path is headless, performs independent discovery, and owns a separate backend instance.

### Surfaces

`Surface` is an opaque token owned by one runtime. Platform modules expose
distinct native handle types:

```text
gpu::surface::win32::create_surface(Runtime*, InstanceHandle, WindowHandle)
gpu::surface::wayland::create_surface(Runtime*, DisplayHandle, SurfaceHandle)
gpu::surface::x11::create_surface(Runtime*, DisplayHandle, WindowHandle)

supports_presentation(Adapter*, Surface*) -> bool?
destroy_surface(Surface*)                 -> void?
request_presentation(DeviceRequest, Surface*) -> DeviceRequest?
```

Query presentation support before device creation, then add the surface to the
immutable request. The device is bound to that exact surface and selects a
presentation-capable private queue, which may differ from its graphics queue.
Presentation requests require at least one public graphics queue.
A surface must outlive its swapchains; destroying a live dependency returns
`RESOURCE_IN_USE`.

### Device

`Device` owns all backend resources.

Responsibilities:

```text
backend lifetime
queue ownership
resource slot tables, including independent allocations
VMA allocator through backend state
descriptor heaps
frame upload arenas
persistent arenas
readback/staging arenas
pipeline cache
debug and stats state
```

Public shape:

```text
Device                         (slot | generation | reserved)
get_device_backend(Device*)    -> BackendKind?
get_device_caps(Device*)       -> DeviceCaps?
```

Multiple live `Device` values may coexist. Each is a compact slot and
generation token resolved through the synchronized process-wide registry.
Most public device operations pin the slot before reading backend state.
`begin_commands` transfers its pin to the command token; hot recording calls
borrow the retained pin without mutating the registry pin count.

Destruction first rejects known live children, then closes the slot. Closing
blocks new pins while active pins, a second child check, and queue completion
are evaluated. Live children return `RESOURCE_IN_USE`; active pins or incomplete
queue work return `DEVICE_BUSY`. Every failure restores the live state without
changing the token or generation. Successful teardown increments the generation
and invalidates the passed token. Device loss bypasses child and progress checks
after pins retire; command tokens remain discardable after loss.

Device-owned table handles, including `GpuAllocation`, carry an opaque
device-and-kind owner plus a local slot and generation. Backend tables reject
foreign owners before validating liveness and generation. `GpuSpan` carries
the same identity plus offset and size, but does not own storage. Frame and
command tokens derive ownership from their device. Shader-visible indices and
GPU addresses remain caller-lifetime values rather than ownership tokens. The
transitional descriptor release API still accepts raw indices; see
`docs/limitations.md`.

### Queues

The API exposes semantic roles and device-local queue identities rather than raw
family indices or backend queue handles:

```text
QueueKind.GRAPHICS
QueueKind.COMPUTE
QueueKind.TRANSFER
Queue { device, id, roles }
```

`DeviceRequest` carries semantic queue counts and optional distinct-role
requirements. The strict request defaults to one queue under each role, and
`request_queues` adds one immutable explicit group. The backend chooses native
families and indices privately. The core stores every selected identity and its
canonical role mask; unsupported counts or alias constraints make the request
unsupported.

`get_queue_counts` reports selected counts by role. `get_queue` returns a
device-owned identity for a role/index pair, and aliased roles return the same
identity with a shared role mask. The Vulkan backend allocates every selected
native identity. Each identity owns a private completion timeline and monotonic
submission sequence. Command entry points take `QueueKind`.

`AllocationDesc`, `BufferDesc`, `TextureDesc`, and
`PersistentAllocDesc` declare a non-empty `QueueRoles` access set. The
backend stores it as immutable resource metadata. Span resolution recovers that
metadata and validates liveness, device ownership, bounds, and the recording
role before backend state changes. A span cannot widen access because it
contains no public access field. Root-addressed shader access remains a caller
precondition because nested pointers are opaque.

The backend deduplicates only admitted roles' native families: one family stays
exclusive; multiple families use private concurrent sharing.

### Command lists

A command list is a transient, owner-bearing token for a device-owned command
record. The public token contains only a `Device` value and generation-checked handle;
the Vulkan command buffer, bind cache, pending layouts, context, queue, frame slot,
and lifecycle state remain backend-owned. Copies therefore alias one record.

State transitions:

```text
RECORDING -> RECORDING_RENDER_PASS -> RECORDING -> EXECUTABLE -> SUBMITTING -> consumed
```

`begin_commands` creates a record in `RECORDING`. Render passes nest into
`RECORDING_RENDER_PASS` and return to `RECORDING` on end. `end_commands` closes
the record to `EXECUTABLE`. `submit` atomically preflights and claims the whole
batch as `SUBMITTING`. Validation or native failure restores it without publishing
queue progress. Success publishes one `CompletionPoint` and invalidates every
submitted token and alias. Frame-slot reuse is rejected while abandoned records
remain; callers must submit or discard them. Invalid transitions return faults,
and render-pass command constraints remain enforced.

### Independent allocations

`allocate_memory` creates a `GpuAllocation` for generic GPU data.
`MemoryClass` selects CPU-write, GPU-private, or CPU-read behavior without
exposing backend heaps. `AllocationInfo` reports immutable properties and
actual mapping, coherence, and address capabilities.

The device owns each allocation in a generation-checked table and keeps native
storage private. `get_allocation_span` borrows the complete range; mapping and
address queries resolve live spans. Stale, foreign, and out-of-bounds spans are
rejected before native use.

`flush_mapped_span` publishes CPU writes; after GPU completion,
`invalidate_mapped_span` publishes GPU writes to the CPU. Coherent allocations
skip native visibility calls. Non-coherent atom alignment remains private.

`free_allocation` consumes its token only after success. It requires quiescent
GPU use and destroys storage immediately.

### Buffers

Buffers are backend-owned resources with optional CPU mapping and optional shader-visible GPU address.

Public uses:

```text
copy source/destination
shader-readable/writable storage
indirect command buffers
index buffers
fixed vertex buffers when needed
arena backing buffers
readback buffers
```

Backend slot:

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
```

### Textures

Textures represent Vulkan images and their default views.

Public uses:

```text
sampled textures
storage textures
color attachments
depth/stencil attachments
transfer sources/destinations
```

Backend slot:

```text
TextureSlot
    vk::Image image
    vma::Allocation allocation
    vma::AllocationInfo allocation_info
    vk::ImageView default_view
    vk::ImageLayout layout
    Format format
    TextureUsage usage
    uint width, height, depth
    uint mip_levels
    ushort generation
    bool used
```

### Texture and sampler descriptors

`TextureHandle` owns the image. `TextureIndex` is the shader-visible descriptor heap index.

This separation matters:

```text
TextureHandle -> lifetime and commands
TextureIndex  -> shader-visible sampled/storage reference
```

Destroying a texture must invalidate or reject descriptors pointing at it according to the final debug policy. The safer initial policy is to require descriptor destruction before texture destruction in debug builds and report a fault otherwise.

### Pipelines

Pipelines are immutable shader execution objects. Creation is split by kind:

```text
create_compute_pipeline(device, ComputePipelineDesc)   -> PipelineHandle?
create_graphics_pipeline(device, GraphicsPipelineDesc) -> PipelineHandle?
```

Graphics pipelines include the Vulkan-required immutable state. Viewport, scissor, cull mode, front face, and supported depth state are dynamic. The pipeline cache deduplicates the remaining blend/depth/raster state and fronts a serializable driver cache: `get_pipeline_cache_size` / `get_pipeline_cache_data` export the driver blob, and `DeviceDesc.pipeline_cache_data` warm-starts it at device creation.

### Synchronization

Each successful submission returns a reusable `CompletionPoint`. Cross-queue
submission dependencies use `SubmitDesc.completion_waits`; same-queue order is
implicit. Native timelines and swapchain semaphores remain backend-private.

### Swapchains

Swapchains are optional. Headless compute and offscreen graphics must work without a swapchain.

A swapchain borrows a runtime-owned `Surface`. SDL3 supplies native handles to the platform surface module in samples; SDL types do not enter the core API.

## 5. Backend dispatch

Preferred backend connection:

```text
public Device token -> private device state -> private backend dispatch
```

Public functions validate the token before dispatch. Each Vulkan runtime owns
surface discovery and optional debug instance dispatch. A device retains only
the groups required by its request and loads only their dispatch. Headless
devices create no presentation queue, mutex, table, or dispatch. Backend
pointers and dispatch declarations remain private.

## 6. Resource lifetime

### Creation

Creation functions return fallible values:

```text
Device?
GpuAllocation?
BufferHandle?
TextureHandle?
PipelineHandle?
GpuSpan?
```

Failures return specific faults.

### Destruction

Destruction is explicit and validates handles. `free_allocation` consumes a
`GpuAllocation*` only after success; faults preserve it.

Policy:

```text
invalid handle              -> INVALID_HANDLE
resource still referenced   -> RESOURCE_IN_USE or INVALID_RESOURCE_STATE
valid destruction           -> retire slot and increment generation
```

### Deferred destruction

Vulkan resources cannot be destroyed while in use by the GPU. The backend maintains per-frame deferred destruction queues:

```text
retire_frame(frame_index)
    destroy resources whose retire_timeline <= completed_timeline
```

Calling `destroy_buffer` removes the public handle immediately, but backend
destruction may be deferred. Independent allocations instead require quiescence
and are freed immediately.

## 7. Frame model

Each frame-in-flight has:

```text
frame upload arena
command pool(s)
deferred destruction list
last submit timeline value
```

Frame lifecycle and flow:

```text
IDLE --begin_frame(device)--> ACTIVE(token generation) --end_frame(token)--> IDLE

begin_frame(device) -> token      // valid only in IDLE
    wait if frame slot is still in flight
    reset command pools
    reset frame upload arena
    set VMA current frame index

alloc_frame_span(token, ...)     // current active generation only
record work
submit work

end_frame(token)                 // consumes only after success
    record frame timeline value
```

The public `FrameToken` embeds its owning `Device` value plus a nonzero
device-owned generation. Copies alias one active generation.
Successful end clears the passed copy and invalidates every alias through
device-owned state. Failed end preserves the token and all prospective
retirement state so the caller can retry. Invalid lifecycle transitions fault
before changing the frame slot, arena, pools, retirement state, or queue
submissions.

`@with_frame` is a compile-time direct-call helper, not a runtime callback. It
calls a named optional-returning worker and attempts end exactly once after the
worker completes or faults. End faults take precedence and retain the caller's
token for retry. This adds no heap allocation or per-frame indirect dispatch.

Headless tests may skip swapchain-specific acquire/present steps.

## 8. Command model

`begin_commands(queue)` returns a thread-confined recording token. Successful
`end_commands` consumes it and returns a one-shot executable token. Submission
or explicit discard consumes the executable token. Native recording pools are
private and cached per worker; see `docs/threading.md`.

### Compute

```text
cmd_dispatch(command_list, pipeline, root_gpu, groups)
```

The command binds the compute pipeline, pushes the root pointer, and dispatches.

### Graphics

```text
cmd_begin_render_pass(command_list, render_pass_desc)
cmd_set_viewport(command_list, viewport)
cmd_set_scissor(command_list, scissor)
cmd_draw(command_list, pipeline, vertex_root, fragment_root, vertex_count, instance_count)
cmd_draw_indexed(command_list, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count, index_type = IndexType.U32)
cmd_end_render_pass(command_list)
```

Pass begin records full-pass viewport/scissor defaults. The public setters
record one portable pass-bounded rectangle each; their state persists across
pipeline binds until another setter or the next pass begin. Viewport/scissor
remain outside pipeline keys and pipeline-state replay, so handle aliasing
cannot overwrite caller-selected rectangles.

Vertex data can be shader-loaded through GPU addresses. Fixed-function vertex input is allowed for simple paths, but it is not the preferred data model.

### Transfer

```text
cmd_copy_buffer
cmd_copy_buffer_to_texture
cmd_copy_texture_to_buffer
cmd_fill_buffer
```

Transfer helpers do not imply next-use barriers.

### Barriers

Barriers are explicit:

```text
cmd_buffer_barrier(command_list, BufferBarrier)
cmd_texture_barrier(command_list, TextureBarrier)
cmd_global_barrier(command_list, GlobalBarrier)
```

Render pass boundaries do not imply shader-read or transfer-read readiness.

## 9. Descriptor heap model

The public API exposes descriptor allocation and updates:

```text
create_texture_descriptor(device, texture, view_desc) -> TextureIndex?
destroy_texture_descriptor(device, index) -> void?
create_sampler(device, desc) -> SamplerIndex?
destroy_sampler(device, index) -> void?
```

The strict request initializes one inline, device-owned heap group. Heap mode,
capacities, descriptor objects, native features, dispatch, pipeline state, and
published capabilities derive from that group. An unrequested group owns none
of that state; current public request validation requires strict semantics.

`AUTO` prefers descriptor indexing and falls back to descriptor buffers when
indexing is unavailable. Callers may force either path. Neither changes shader
material records.

## 10. Swapchain model

```text
acquired = acquire_next_image(device, swapchain)
rendered = submit(graphics, command lists + acquired.readiness)
present(device, acquired, rendered)
```

Acquisition returns a borrowed texture and compact one-shot readiness value.
Submission validates the exact device, swapchain generation, acquisition
identity, and graphics role before waiting the private native acquire bridge.
Only successful native submission consumes readiness and records its returned
`CompletionPoint` as the acquisition's render completion.

Presentation consumes the exact `AcquiredImage` and accepts only that render
completion point. The point remains reusable for host observation. Native
binary synchronization stays in the backend; public ordering uses readiness
and completion values only.

`SwapchainInfo` reports the selected format, extent, image count, present mode,
and dormant state. `AcquiredImage.prior_layout` comes from committed texture
state. Resize stales borrowed textures and never reuses acquisition identities.

Surface creation remains platform-specific. SDL helpers live outside `gpu`.

## 11. Debug model

Debug builds should track:

```text
resource names
live resource counts
allocation names
allocation user data
slot generation errors
resource state errors
outstanding frame allocations
leaked descriptors
leaked backend objects
```

`destroy_device` rejects live public resources. Validation diagnostics during
accepted teardown cover internal, partial-initialization, and device-loss state.

## 12. Supported baseline

The current architecture supports:

```text
headless root-pointer compute
bindless texture compute
offscreen graphics readback
SDL3 windowed triangle sample
GPU-driven indirect draw sample
independent allocation ownership, mapping, and address queries
VMA memory budget reporting
debug resource leak reporting
shader ABI docs and generated layout checks
```
