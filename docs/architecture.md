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
TextureView     -> owner-bearing published-view lifetime
TextureIndex    -> shader-visible texture heap index
Sampler         -> immutable device-interned sampler identity
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

Canonical `create_device(Adapter*, DeviceRequest*)` uses the exact borrowed adapter, retains its runtime, and reuses the runtime-owned backend instance. `supports_device_request` is read-only and does not enable state. The direct `create_device_from_desc(DeviceDesc*)` path is headless, performs independent discovery, and owns a separate backend instance.

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
caller-owned allocation and completion lifetimes
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

Device-owned table handles and values, including `GpuAllocation`, `Sampler`, and
`TextureView`, carry an opaque device-and-kind owner plus a local slot and
generation. Backend tables reject foreign owners before validating liveness and
generation. `GpuSpan` carries the same identity plus offset and size, but does
not own storage. Command tokens derive ownership from their device.
Shader-visible `TextureIndex`, `SamplerIndex`, and `GpuAddress` values contain no
owner or generation metadata. They are direct device-local values whose lifetime
the caller must preserve. `TextureView` owns a recyclable texture index;
published sampler indices remain stable until device destruction.

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

`AllocationDesc` and `TextureDesc` declare a non-empty `QueueRoles` access
set. The backend stores it as immutable resource
metadata. Span resolution validates liveness, device ownership, bounds, and the
recording role before backend state changes. A span cannot widen access because
it contains no public access field. Root-addressed shader access remains a
caller precondition because nested pointers are opaque.

The backend deduplicates only admitted roles' native families: one family stays
exclusive; multiple families use private concurrent sharing.

### Command lists

A command list is a transient, owner-bearing token for a device-owned command
record. The public token contains only a `Device` value and generation-checked
handle; the Vulkan command buffer, pool, bind cache, pending layouts, context,
queue, and lifecycle state remain backend-owned. Copies therefore alias one record.

State transitions:

```text
RECORDING -> RECORDING_RENDER_PASS -> RECORDING -> EXECUTABLE -> SUBMITTING -> consumed
```

`begin_commands` creates a record in `RECORDING`. Render passes nest into
`RECORDING_RENDER_PASS` and return to `RECORDING` on end. `end_commands` closes
the record to `EXECUTABLE`. `submit` atomically preflights and claims the whole
batch as `SUBMITTING`. Validation or native failure restores it without publishing
queue progress. Success publishes one `CompletionPoint` and invalidates every
submitted token and alias. Completion observation and discard retire native
buffers to their recording context. The context owner reclaims them before its
next allocation; device teardown relies on command-pool destruction.
Invalid transitions return faults,
and render-pass command constraints remain enforced.

### Independent allocations

`allocate_memory` creates a `GpuAllocation` for generic data or placed
textures. `MemoryClass` selects CPU-write, GPU-private, CPU-read, or texture
behavior without exposing backend heaps. `AllocationInfo` reports immutable
properties and actual mapping, coherence, and address capabilities. Texture
memory has none of the generic span, mapping, or address capabilities.

The device owns each allocation in a generation-checked table and keeps native
storage private. `get_allocation_span` borrows the complete range; mapping and
address queries resolve live spans. Stale, foreign, and out-of-bounds spans are
rejected before native use.

`flush_mapped_span` publishes CPU writes; after GPU completion,
`invalidate_mapped_span` publishes GPU writes to the CPU. Coherent allocations
skip native visibility calls. Non-coherent atom alignment remains private.

`free_allocation` consumes its token only after success. It requires quiescent
GPU use and destroys storage immediately. Long-lived CPU-written data stays in
a caller-owned `CPU_WRITE` allocation: borrow its span, mapping, and address as
needed, write and flush, record and submit, wait for or poll completion, then
free the allocation. Readback waits or polls before invalidating and reading a
`CPU_READ` mapping.

### Private buffer backing

The public API has no buffer object. Generic GPU data is owned by
`GpuAllocation`, borrowed as `GpuSpan`, and addressed as `GpuAddress`.
Copy, fill, index, indirect, upload, and readback workflows resolve spans to
private native buffers.

The Vulkan backend may use private `BufferHandle`, `BufferDesc`, and
`BufferUsage` declarations for generic allocation backing. They remain in
`gpu::vk` and never cross backend dispatch.

### Textures

Textures represent images and views without exposing backend objects.
`create_texture` owns its storage. `get_texture_requirements` and
`create_placed_texture` let the application group compatible textures into
explicit allocations. Requirements are immutable device-owned values.

Placement validates memory class, compatibility, size, alignment, access, and
overlap before native creation. Texture destruction never releases caller-owned storage.

### Texture and sampler heap publication

`TextureHandle` owns the image. `TextureView` is the owner- and
generation-checked CPU token for one published image view; its `TextureIndex`
field is the raw 32-bit shader-visible heap value. `Sampler` is an immutable,
owner-bearing identity interned from semantic state. `SamplerIndex` is its
stable strict shader-heap publication.

This separation matters:

```text
TextureHandle -> image lifetime and commands
TextureView   -> published-view lifetime and CPU validation
TextureIndex  -> generation-free sampled/storage shader value
Sampler       -> device-lifetime immutable identity
SamplerIndex  -> generation-free, device-lifetime shader value
```

Destroying a texture with live views returns `RESOURCE_IN_USE`. Destroying a
view immediately recycles its index, so every GPU reference must already be
complete and stale shader data must be removed.

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
valid destruction           -> invalidate slot and increment generation
```

### Immediate resource lifetime

Non-WSI resource destruction never waits or queues backend work. The caller
must keep every GPU-visible resource alive until recording tokens are discarded
and submitted completion points finish. Swapchain destruction and resize remain
under the current presentation contract until strict presentation integration.

With validation enabled, the backend rejects destruction when an explicitly
named span, texture, or pipeline is referenced by a recording token, executable
token, or incomplete submission. GPU addresses and shader indices are opaque to
the command stream, so their lifetime remains a caller precondition.

## 7. Work and storage lifetime model

The root module does not define application work boundaries or storage rotation.
Every submission returns a `CompletionPoint`, and the caller uses that point to
decide when commands and resources may be retired, reused, or destroyed.

A typical CPU-authored root-data flow is:

```text
allocate CPU_WRITE storage
write through the mapped GpuSpan
flush_mapped_span
record commands that use its GpuAddress
submit and retain the returned CompletionPoint
wait or poll before rewriting or freeing the allocation
```

Readback reverses visibility: GPU work writes a `CPU_READ` span, the caller
waits for the covering completion, calls `invalidate_mapped_span`, and then
reads the mapping. The visibility calls do not wait.

Applications choose their own allocation granularity, pooling, and number of
concurrent work sets. A ring or pool is application policy built from
`GpuAllocation`, `GpuSpan`, and `CompletionPoint`; it is not device
configuration. Headless and windowed programs use the same ownership model.

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
cmd_copy_buffer(command_list, { src_span, dst_span })
cmd_copy_buffer_to_texture
cmd_copy_texture_to_buffer
cmd_fill_buffer(command_list, dst_span, value)
```

Transfers operate on caller-owned spans. Applications allocate and map
`CPU_WRITE` or `CPU_READ` storage, record copies and barriers, and retain that
storage until the returned `CompletionPoint` is reached. Transfer commands own no
storage and perform no blocking work.

### Barriers

Barriers are explicit:

```text
cmd_buffer_barrier(command_list, BufferBarrier)
cmd_texture_barrier(command_list, TextureBarrier)
cmd_global_barrier(command_list, GlobalBarrier)
```

`BufferBarrier` scopes the hazard to one exact span and has no zero-size
shorthand.

Render pass boundaries do not imply shader-read or transfer-read readiness.

## 9. Descriptor heap model

The public API exposes device-wide view and sampler publication:

```text
create_texture_view(device, texture, view_desc) -> TextureView?
destroy_texture_view(device, view) -> void?
create_texture_views(device, descs, out_views) -> void?
intern_sampler(device, desc) -> Sampler?
publish_sampler(device, sampler) -> SamplerIndex?
```

The strict request initializes one inline, device-owned heap group. Heap mode,
capacities, descriptor objects, native features, dispatch, pipeline state, and
published capabilities derive from that group. An unrequested group owns none
of that state; current public request validation requires strict semantics.

Sampler interning is independent of that optional heap group. The backend keeps
one append-only sampler table per device, deduplicates equal effective state,
and destroys every native sampler during device teardown. Strict publication
adds at most one heap entry per identity and never transfers native ownership.

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
outstanding allocation and command references
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
