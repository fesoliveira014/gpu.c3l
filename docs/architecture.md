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
SamplerIndex    -> stable device-lifetime sampler heap index
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
    ContractValidation contract_validation
    bool track_resource_lifetimes
    bool enable_vulkan_validation
    bool enable_debug_names
    uint texture_heap_capacity
    uint sampler_heap_capacity
    uint texture_capacity
    uint pipeline_capacity
    char[] pipeline_cache_data
    ZString application_name
    DebugMessageCallback debug_callback
    void* debug_user_data

create_runtime(RuntimeDesc*)       -> Runtime?
enumerate_adapters(Runtime*)       -> AdapterList?
AdapterList.get(uint)              -> Adapter?
get_adapter_info(Adapter*)         -> AdapterInfo?
get_adapter_diagnostics(Adapter*)  -> AdapterDiagnostics?
destroy_runtime(Runtime*)          -> void?
full_validation_runtime_desc()     -> RuntimeDesc
```

Runtime policy has independent axes. `ContractValidation.TRUSTED` is the zero
value and uses the direct command path. Every level rejects stale/foreign
identities at public resolve/destroy boundaries as mandatory safety;
`OBJECT_BOUNDARIES` adds structured diagnostics for those failures at covered
public resource and command-lifecycle boundaries while sharing that direct
path. `FULL` selects detailed command semantic checks.
`track_resource_lifetimes` independently selects command reference retention,
and `enable_vulkan_validation` independently requests the Khronos layer.
Debug names request best-effort native naming, while a callback only selects
delivery for diagnostics already produced. The all-zero descriptor is therefore
trusted/no-tracking/no-layer. `full_validation_runtime_desc()` returns the
former all-enabled development configuration.

`AdapterList` is an allocation-free view. Its adapters and the read-only strings in adapter query results are borrowed until their runtime is destroyed. Destroying a runtime consumes its token, invalidates its adapter views and handles, and returns `RESOURCE_IN_USE` while a dependent surface or device is live.

Canonical `create_device(Adapter*, DeviceRequest*)` uses the exact borrowed adapter, retains its runtime, and reuses the runtime-owned backend instance. `supports_device_request` is read-only and enables no state. Backend-neutral device defaults are copied by `create_runtime` and inherited by devices created from that runtime.

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
caller-owned command allocators and fixed recording storage
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
`begin_commands` transfers its pin to the command token and publishes one
stable, backend-opaque encoder cell. Hot recording calls validate that cell and
dispatch through its immutable command-operation table without resolving the
device registry, borrowing another pin, or loading the lifecycle vtable.

Destruction first rejects known live children, then closes the slot. Closing
blocks new pins while active pins, a second child check, and queue completion
are evaluated. Live children return `RESOURCE_IN_USE`; active pins or incomplete
queue work return `DEVICE_BUSY`. Every failure restores the live state without
changing the token or generation. Successful teardown increments the generation
and invalidates the passed token. Device loss bypasses child and progress checks
after pins retire; command tokens remain discardable after loss.

Device-owned table handles and values, including `GpuAllocation` and
`TextureView`, carry an opaque device-and-kind owner plus a local slot and
generation. Backend tables reject foreign owners before validating liveness and
generation. `GpuSpan` carries the same identity plus offset and size, but does
not own storage. Command tokens derive ownership from their device.
Shader-visible `TextureIndex`, `SamplerIndex`, and `GpuAddress` values contain no
owner or generation metadata. They are direct device-local values whose lifetime
the caller must preserve. `TextureView` owns a recyclable texture index;
sampler indices remain stable until device destruction.

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

### Command allocators and lists

A `CommandAllocator` is a caller-owned, generation-checked device child bound
to one exact selected `Queue`. Its backend slot owns one native command pool for
that queue family, a fixed set of native command buffers, stable per-buffer
reference and generated-reservation-index slices, a recycling stack, and a
fixed generated-reservation table. Creation performs every host allocation,
pool creation, and command-buffer allocation before publication. Destroying a
live allocator never waits and returns `RESOURCE_IN_USE` until all its command
units are discarded or completion-retired.

A command list is a transient, owner-bearing token for a device-owned command
record and one allocator buffer/scratch index. The public token contains a
`Device` value, generation-checked handle, and one opaque encoder pointer; the
Vulkan command buffer, pool, bind cache, allocator identity, queue, and lifecycle
state remain backend-owned. The encoder pointer is part of the command-token
ABI. Copies alias one encoder phase and record.
Each device slot owns a fixed `MAX_DEVICE_COMMANDS` encoder array. Across all
device slots this zero-initialized storage is capped at 16 MiB; operating-system
pages commit as encoder cells are touched.

State transitions:

```text
RECORDING -> RECORDING_RENDER_PASS -> RECORDING -> EXECUTABLE -> SUBMITTING -> consumed
```

`begin_commands(allocator)` pops one preallocated index and creates a record in
`RECORDING`. If all configured indices are live it returns `DEVICE_BUSY`
without allocation or state change. Render passes nest into
`RECORDING_RENDER_PASS` and return to `RECORDING` on end. `end_commands` closes
the record to `EXECUTABLE`. `submit` atomically preflights and claims the whole
batch as `SUBMITTING`. Duplicate detection visits the submitted records once,
using a device-local epoch stored on each record; ordinary work is proportional
to the command-list count rather than the square of it. Epoch rollover scans the
allocated command table once before reusing epoch one. Validation or native
failure restores it without publishing queue progress. Success publishes one
`CompletionPoint` and invalidates every submitted token and alias. A submission
may mix allocators only when every token targets the exact submit queue.
Completion observation and discard return each native buffer and scratch index
to its originating allocator. The next begin resets and reuses that unit. Warm
begin, recording, end, submit, discard, retirement, and reuse never allocate
host storage, native command buffers, pools, VMA storage, or C3 temporary-pool
memory. Per-list reference or generated-index exhaustion returns
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` transactionally. Device teardown rejects
even a quiescent live public allocator. Invalid
transitions return faults, and render-pass command constraints remain enforced.
A render pass records its attachment formats and sample count; a graphics
pipeline must match them before begin or bind mutates native command state.
Resolve and pass boundaries do not add implicit synchronization.

Render targets name explicit `AttachmentViewHandle` children created before
recording. Each immutable view selects one texture mip and layer. User-created
views retain their texture and own any non-default native image view. Borrowed
swapchain views do not retain their jointly owned texture. Render-pass begin
resolves the fixed view table into fixed-size local arrays. It does not create
image views, grow a cache, or allocate host storage.

Generated commands consume preprocess buffers from explicit reservations on
their originating command allocator. Each reservation is keyed by
pipeline and generated-work kind; its count bound is translated into exact
driver-reported size, alignment, and memory-type requirements. The allocator
fixes the exact queue, reservation-table capacity, total native byte budget,
and per-list retained-index ceiling. Warm recording returns
`GENERATED_SCRATCH_EXHAUSTED` instead
of allocating when the count or compatible-buffer supply is exhausted. Discard
and completion return reserved buffers to the same allocator; another allocator
never acquires them implicitly.

Pipeline bind generation-checks the public handle, resolves the cache entry,
optionally retains tracked ownership, and publishes a complete bound snapshot: native
pipeline and layout, kind, render compatibility, cache identity, generated-work
layout, and public diagnostic identity. Later draw, dispatch, render-pass,
indirect, and generated commands use that snapshot without reading a pipeline
cell, pipeline table, or cache. Ending a render pass clears a graphics snapshot;
a legal compute snapshot remains available. Command-policy variants are outside
this mechanism: policy selection happens when the device or encoder is created,
never in a warm recording call.

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

Textures represent implicit 2D images, including mip chains and array layers,
and same-format views without exposing backend objects.
`create_texture` owns its storage. `get_texture_requirements` and
`create_placed_texture` let the application group compatible textures into
explicit allocations. Requirements are immutable device-owned values.

Placement validates memory class, compatibility, size, alignment, access, and
overlap before native creation. Texture destruction never releases caller-owned storage.

### Texture and sampler heap publication

`TextureHandle` owns the image. `TextureView` is the owner- and
generation-checked CPU token for one published image view; its `TextureIndex`
field is the raw 32-bit shader-visible heap value. `SamplerIndex` is returned
directly by semantic-state interning and remains stable for the device lifetime.

This separation matters:

```text
TextureHandle -> image lifetime and commands
TextureView   -> published-view lifetime and CPU validation
TextureIndex  -> generation-free sampled/storage shader value
SamplerIndex  -> generation-free, device-lifetime shader value
```

Destroying a texture with live views returns `RESOURCE_IN_USE`. Destroying a
view immediately recycles its index, so every GPU reference must already be
complete and stale shader data must be removed.

### Shader code and pipelines

`ShaderCode` is borrowed CPU-side SPIR-V prepared by the library. Its opaque,
process-local identity is derived from the bytes; stage and entry point complete
the identity, while debug names do not. Callers keep the borrowed inputs
immutable and alive for each creation call. Each device interns exact identity
into one owned SPIR-V clone and represents it internally with a compact ID. The
same prepared code may be reused for multiple pipelines and devices; each device
owns independently. Native shader modules are temporary pipeline-construction
details, never public resources.

Pipelines are immutable shader execution objects. Creation is split by kind:

```text
create_compute_pipeline(device, ComputePipelineDesc)    -> PipelineHandle?
create_graphics_pipeline(device, GraphicsPipelineDesc)  -> PipelineHandle?
create_compute_pipelines(device, descriptions, outputs)  -> void?
create_graphics_pipelines(device, descriptions, outputs) -> void?
```

Graphics pipeline identity includes shaders, per-target format/blend/write-mask
state, depth format, sample count, and polygon mode. Topology, cull mode,
front face, depth bias, depth state, viewport, and scissor are command-time
state. Compute pipelines share one device-owned `RootPush` layout and, when
generated work is available, one generated-dispatch layout. Batch creation
reuses temporary modules by shader ID, deduplicates pipeline identity, and
publishes every handle transactionally. Pipeline-cache entries own refcounted
IDs, and the device recycles an identity after its last cache entry releases it.
The pipeline cache fronts a serializable driver cache:
`get_pipeline_cache_size` /
`get_pipeline_cache_data` export the driver blob, and
`RuntimeDesc.pipeline_cache_data` warm-starts it for devices created from that runtime.

### Synchronization

Each successful submission returns a reusable `CompletionPoint`. Cross-queue
submission dependencies use stage-scoped `SubmitDesc.completion_waits`;
same-queue order is implicit after point and stage validation. Swapchain
readiness likewise names its first destination stages. Native timelines and
swapchain semaphores remain backend-private.

### Swapchains

Swapchains are optional. Headless compute and offscreen graphics must work without a swapchain.

A swapchain borrows a runtime-owned `Surface`. Its images are ordinary borrowed
texture handles used by shared transitions, render passes, and command lifetime
validation. SDL3 supplies native handles to the platform surface module in
samples; SDL types do not enter the core API.

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

GPU-visible resource destruction, including swapchain destruction and resize,
never waits, submits hidden work, or queues deferred release. The caller keeps
every resource alive until recording tokens are discarded, submitted completion
points finish, and private presentation resource retirement completes. A pending acquired
image returns `INVALID_RESOURCE_STATE`; detected command/view or presentation
use returns `RESOURCE_IN_USE`; faults preserve the owning token for retry.

With `track_resource_lifetimes` enabled, the backend retains explicitly named
spans, textures, attachment views, allocations, and pipelines across recording,
executable, and incomplete-submission phases; destruction returns
`RESOURCE_IN_USE` until discard or retirement releases the reference. With
tracking disabled, records contain no reference storage and destruction adds no
implicit wait or deferred release. The caller must observe completion before
destroying every referenced owner. GPU addresses and shader indices are opaque
to the command stream under either setting, so their lifetime always remains a
caller precondition.

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

`begin_commands(allocator)` returns a thread-confined recording token from an
explicit exact-queue allocator. Successful
`end_commands` consumes it and returns a one-shot executable token. Submission
or explicit discard consumes the executable token. Native recording pools are
owned by allocators, not workers or frame boundaries; see `docs/threading.md`.

Vulkan device creation selects one immutable command table from contract depth
and tracking: trusted/no-tracking, trusted/tracking, checked/no-tracking, or
checked/tracking. `OBJECT_BOUNDARIES` uses the trusted command entries because
its additional work belongs at public boundaries. The encoder snapshots the
chosen pointer once; repeated `cmd_*` calls do not inspect contract, tracking,
layer, callback, or naming policy. All four tables retain mandatory host
pointer/slice/range safety, overflow protection, internal state integrity,
public ownership, Vulkan result handling, and rollback. Detailed command misuse
outside that floor is a caller contract violation unless `FULL` is selected.

### Compute

```text
cmd_bind_pipeline(command_list, pipeline)
cmd_dispatch(command_list, root_gpu, groups)
```

Binding selects the active compute pipeline and descriptor heap. Dispatch
requires that active kind, pushes the root address unchanged, and executes
without native pipeline creation. Zero is valid when the shader avoids
dereferencing it.

### Graphics

```text
cmd_begin_render_pass(command_list, render_pass_desc)
cmd_bind_pipeline(command_list, pipeline)
cmd_set_raster_state(command_list, raster_state)
cmd_set_depth_state(command_list, depth_state)
cmd_set_viewport(command_list, viewport)
cmd_set_scissor(command_list, scissor)
cmd_draw(command_list, vertex_root, fragment_root, vertex_count, instance_count)
cmd_draw_indexed(command_list, vertex_root, fragment_root, index_span, index_count, instance_count, index_type = IndexType.U32)
cmd_end_render_pass(command_list)
```

Pass begin records full-pass viewport/scissor defaults and the zero raster
default: triangles, no culling, counter-clockwise front faces, and disabled
depth bias. It requires a fresh depth-state command before drawing. Raster,
viewport, and scissor state persist across pipeline binds until another setter
or the next pass begin. Their setters, including `cmd_set_raster_state`, require
an active render pass. They remain outside pipeline keys, so handle aliasing
cannot overwrite caller-selected command state. Draws require an active
graphics pipeline, push both roots unchanged, and perform no native pipeline
creation.

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
cmd_barrier(command_list, Barrier)
cmd_texture_barrier(command_list, TextureBarrier)
```

`Barrier` declares a global semantic dependency between nonempty stage masks.
It carries no resource identity, address, range, layout, or queue family;
special draw-argument, descriptor, and depth/stencil hazards are explicit.
`TextureBarrier` declares a texture, subresource range, and compositional
previous and next states. A `TextureState` carries an independent
`TextureLayout`, `StageMask`, and read/write `TextureAccess`; sampled and
storage helpers construct common states without recording work. Cross-queue
dependencies use completion waits.

The caller owns texture history. `before` asserts the state established by
earlier ordering; the backend validates and lowers it but never consults or
updates a shared layout table. Each accepted barrier resolves its texture and
range once, lowers its two states once, and emits one native barrier.

Render pass boundaries do not imply shader-read or transfer-read readiness.

## 9. Descriptor heap model

The public API exposes device-wide view and sampler publication:

```text
create_texture_view(device, texture, view_desc) -> TextureView?
destroy_texture_view(device, view) -> void?
create_texture_views(device, descs, out_views) -> void?
intern_sampler(device, desc) -> SamplerIndex?
```

The strict request initializes one inline, device-owned heap group. Callers
request texture and sampler capacities; descriptor objects, native features,
dispatch, and pipeline state remain private. An unrequested group owns none of
that state; current public request validation requires strict semantics.

The backend keeps one append-only sampler table per strict-enabled device.
Canonical sampler keys are byte-hashed into fixed power-of-two buckets; a hash
match still requires complete canonical equality. Interning holds the resource
mutex and publishes the native sampler, descriptor, stable index, cell, and
bucket link as one transaction. Device teardown destroys every native sampler
and releases both slot and bucket storage.

The backend prefers descriptor indexing when it satisfies the requested
capacities and falls back to descriptor buffers when available. Callers cannot
select or branch on the native implementation; shader material records are
identical on either path.

## 10. Swapchain model

```text
acquired = acquire_next_image(device, swapchain, timeout_ns)
rendered = submit(graphics, command lists + acquired.readiness before color output)
present(device, acquired, rendered)
```

Acquisition returns a borrowed texture, its swapchain-owned color attachment
view, and a compact one-shot readiness value. Callers render with the view but
do not destroy it. The caller-selected timeout reaches native acquisition
unchanged; zero is nonblocking and is the public default. Timeout and every
failed native result preserve the complete pending-acquisition and semaphore-
retirement state. Success validates the returned image and publishes the
acquisition sequence, selected semaphore, image, and readiness together.
Submission validates the exact device, swapchain generation, acquisition
identity, graphics role, and caller-provided first destination stages before
waiting the private native acquire bridge.
Only successful native submission consumes readiness and records its returned
`CompletionPoint` as the acquisition's render completion.

Presentation consumes the exact `AcquiredImage` and accepts only that render
completion point. The point remains reusable for host observation. Native
binary synchronization and presentation-retirement fences stay in the backend;
public ordering uses readiness and completion values only. If an image's prior
presentation fence is not yet reusable, `present` returns `WAIT_TIMEOUT` and
preserves the acquired image.

`SwapchainInfo` reports the selected format, extent, image count, present mode,
and dormant state. `AcquiredImage.prior_state` is the empty `UNDEFINED` state
before first use and the empty `PRESENT` state after presentation. Callers pass
it directly as the first barrier's source. Resize stales borrowed texture and
attachment view handles and never reuses acquisition identities. Resize and destruction
reject a pending acquisition or live command/view/presentation use without
waiting, preserving the swapchain for retry.

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

`destroy_device` rejects live public resources. Accepted teardown scans report
internal, partial-initialization, and device-loss state under
`OBJECT_BOUNDARIES`/`FULL`, or whenever a callback is present. A callback only
selects structured delivery; it does not change contract checks, tracking,
layers, or naming. Vulkan layer messages remain native routing and are
independent of library contract diagnostics.

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
