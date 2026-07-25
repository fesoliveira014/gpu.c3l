# gpu.c3l Architecture

## 1. Purpose

`gpu.c3l` is a C3 library that exposes a direct GPU programming model suitable for modern explicit rendering and compute workloads. It is not a renderer, render graph, material system, asset system, or platform abstraction layer.

The API uses root pointers, bindless heap indices, explicit barriers, and
backend-neutral GPU allocation. Its design follows Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api);
the [device baseline architecture](device_baseline.md) records the broader design.

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
gpu::internal registry and lifetime policy
        |
        v
gpu::internal::vk Vulkan backend
        |
        +--> vk.c3l         -> Vulkan API calls
        +--> vma.c3l        -> Vulkan memory allocation
        +--> spvreflect.c3l -> SPIR-V shader reflection
```

Caller-supplied descriptors and callable signatures do not expose Vulkan or
VMA binding types. SDL3 integration belongs to the separate
`gpu.c3l-samples` repository and is not a backend dependency.

There is no runtime backend plugin interface. Adding another backend is future
source work, not a current stable private ABI.

## 3. Package structure

`gpu.c3l` uses a C3 library package layout. Shipped library source files live under one `gpu/` subtree.

```text
gpu.c3l/
├── abi/                     shader ABI schemas
├── manifest.json
├── gpu/
│   ├── gpu.c3i              public non-callable declarations
│   ├── gpu.c3               public callable implementations and contracts
│   ├── surface/
│   │   ├── win32/
│   │   │   ├── surface.c3i  public native handle types
│   │   │   └── surface.c3   public surface callable
│   │   ├── wayland/
│   │   │   ├── surface.c3i
│   │   │   └── surface.c3
│   │   └── x11/
│   │       ├── surface.c3i
│   │       └── surface.c3
│   └── internal/
│       ├── *.c3              backend-independent private implementation
│       └── vk/
│           └── *.c3          private Vulkan backend
├── include/
│   └── shaders/        published shader-side ABI includes only (no application shaders)
├── lib/                     vendored C3 bindings
├── scripts/                 ABI, shader, and documentation checks
├── test/
├── tools/
└── docs/
```

### Library files

`gpu/gpu.c3i` is the only source of public non-callable declarations in the
root module. `gpu/gpu.c3` contains every public method, operator, macro, and
free-function implementation together with its contract and default arguments.
Both declare:

```c3
module gpu;
```

Each platform surface keeps its public native typedefs in `surface.c3i` and its
single `create_surface` callable in the adjacent `surface.c3` under
`gpu::surface::{win32,wayland,x11}`.

Backend-independent implementation files under `gpu/internal/` declare:

```c3
module gpu::internal @private;
```

Vulkan files under `gpu/internal/vk/` declare:

```c3
module gpu::internal::vk @private;
```

Public callable implementations and platform surface implementations import
private implementation modules with a scoped visibility override. White-box
tests do the same; consumers should not depend on either internal module.
C3 0.8.0 has no package-private visibility, so the state types shared across
`gpu::internal` and `gpu::internal::vk` use declaration-level `@public`.
Generated metadata can therefore name `VkRuntimeState`, `VkDeviceState`,
`CommandRecord`, and `CommandOps`, including the library-owned `record` member
of command tokens. This compiler visibility is not a supported consumer API;
the token fields remain opaque and library-owned.

Samples are standalone consumers and may declare their own sample modules.

### Shader ownership

The library ships **no application shaders**. Shader entry points are written and owned by the consuming project. The only shader-side artifacts the library publishes are ABI includes under `include/shaders/` (descriptor-heap helpers and generated ABI structs/offsets) that a consumer's shaders `#include`. Samples and tests own their shaders inside their own trees, because they are consumers like any other.

## 4. Public object model

### Runtime and adapters

`Runtime` owns Vulkan discovery, diagnostics, and borrowed adapters. Its opaque
token resolves to typed private Vulkan state; adapter and support queries call
that implementation directly. Creating a runtime is the first operation that
may initialize native state. Multiple runtimes may coexist.

Public shape:

```text
RuntimeDesc
    ContractValidation contract_validation
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

Runtime policy has two coherent contract modes. A zero-initialized
`RuntimeDesc` selects `ContractValidation.TRUSTED`, which performs no command
resource lifetime tracking, with Vulkan validation layers off. Every contract
mode retains the always-checked public and
command-token identity, authoritative-phase, host-safety, safe-lowering,
lifecycle, cold-path, and runtime-result floor. `FULL` selects diagnostic command
tables with detailed semantic diagnostics, command reference retention, and
teardown leak scans. `enable_vulkan_validation` independently requests the
Khronos layer. Debug names request best-effort native naming, while a callback
only selects delivery for diagnostics already produced and never enables FULL
behavior. The all-zero descriptor is therefore TRUSTED with layers disabled.
`full_validation_runtime_desc()` selects FULL and the Vulkan validation layer.

`AdapterList` is an allocation-free view. Its adapters and the read-only strings in adapter query results are borrowed until their runtime is destroyed. Destroying a runtime consumes its token, invalidates its adapter views and handles, and returns `RESOURCE_IN_USE` while a dependent surface or device is live.

Canonical `create_device(Adapter*, DeviceDesc* = null)` uses the exact borrowed
adapter, retains its runtime, and reuses the runtime-owned backend instance.
An omitted, null, or zero-initialized descriptor selects the same default
non-presenting device. `supports_device_desc` is an optional, read-only adapter
selection query and enables no state. Backend-neutral device defaults are
copied by `create_runtime` and inherited by devices created from that runtime.

### Surfaces

`Surface` is an opaque token owned by one runtime. Platform modules expose
distinct native handle types:

```text
gpu::surface::win32::create_surface(Runtime*, InstanceHandle, WindowHandle)
gpu::surface::wayland::create_surface(Runtime*, DisplayHandle, SurfaceHandle)
gpu::surface::x11::create_surface(Runtime*, DisplayHandle, WindowHandle)

supports_presentation(Adapter*, Surface*) -> bool?
destroy_surface(Surface*)                 -> void?
```

Store a surface directly in `DeviceDesc` to request presentation. The support
query may preflight it during adapter selection, but callers may instead create
the device directly and handle `UNSUPPORTED_FEATURE`. A presentation device is
bound to that exact surface and selects a presentation-capable private queue,
which may differ from its graphics queue. Presentation requires at least one
public graphics queue.
A platform surface implementation resolves its runtime token to the typed
private Vulkan runtime state and calls the corresponding WSI operation
directly. There is no runtime forwarding table between the public surface
module and the Vulkan implementation.
A device records but does not retain its descriptor's surface token. A surface
must outlive its swapchains; destroying a live swapchain dependency returns
`RESOURCE_IN_USE`. Destroying a requested surface with no live swapchain
succeeds, and later swapchain creation through the device rejects the stale
token with `INVALID_HANDLE`.

### Device

`Device` owns all backend resources.

Responsibilities:

```text
backend lifetime
queue ownership
resource slot tables, including independent allocations
caller-owned command allocators and fixed recording storage
VMA allocator through typed Vulkan state
descriptor heaps
caller-owned allocation and completion lifetimes
pipeline cache
debug and stats state
```

Public shape:

```text
Device                         (slot | generation | reserved)
get_device_caps(Device*)       -> DeviceCaps?
```

`DeviceDesc` contains the presentation surface and semantic queue requirements.
The library's device baseline is implicit rather than requested. Texture states
use one mandatory explicit native layout mapping on every device. Command-time
color state is part of every device's mandatory graphics-state model.

Multiple live `Device` values may coexist. Each is a compact slot and
generation token resolved through the synchronized process-wide registry.
Most public device operations pin the slot before reading its typed
`VkDeviceState*`.
`begin_commands` transfers its pin to one stable allocator-owned command
record. Hot recording calls reach that record through the build-selected
command token and dispatch through its immutable command-operation table
after one acquire-load of the static device slot proves its liveness and
generation. They do not borrow another pin, resolve a retained device
operation, or look up the command table.

Destruction first rejects known live children, then closes the slot. Closing
blocks new pins while active pins, a second child check, and queue completion
are evaluated. Live children return `RESOURCE_IN_USE`; active pins or incomplete
queue work return `DEVICE_BUSY`. Every failure restores the live state without
changing the token or generation. Successful teardown increments the generation
and invalidates the passed token. Device loss bypasses child and progress checks
after pins retire; command tokens remain discardable after loss.

Device-owned table handles and values, including `GpuAllocation` and
`TextureView`, carry an opaque device-and-kind owner plus a local slot and
generation. Private resource tables reject foreign owners before validating
liveness and generation. `GpuSpan` carries the same identity plus offset and
size, but does not own storage. Command tokens derive ownership from their
device.
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

`DeviceDesc.queues` carries required semantic roles and optional distinct-role
constraints. An all-zero `QueueRequest` selects graphics, compute, and transfer,
with cross-role aliasing allowed. Any other value is an explicit semantic
topology. A distinct role must also be required and cannot alias another
required role. The backend chooses native families and indices privately;
unavailable roles or distinctness constraints make the descriptor unsupported.

`get_queue` returns the single device-owned identity selected for a role, and
aliased roles return the same identity with a shared role mask. Each selected
identity owns a private completion timeline and monotonic submission sequence.
Command entry points take `QueueKind`.

`AllocationDesc` and `TextureDesc` declare a non-empty `QueueRoles` access
set. The backend stores it as immutable resource
metadata. Span resolution validates liveness, device ownership, bounds, and the
recording role before native state changes. A span cannot widen access because
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

A command list is a transient token for one device-table command record paired
with one originating allocator native buffer/scratch unit. Its fixed public
payload carries a library-owned typed pointer to that address-stable record,
its reuse generation, and a packed static device-slot identity. Recording first
checks the slot's liveness and generation, then compares the record generation
and authoritative phase under every policy, without a retained
device-operation borrow or command-table lookup. The token must originate from
`begin_commands`; callers must not inspect, construct, or mutate its fields.

Every preallocated command cell owns one address-stable authoritative
Vulkan `CommandRecord`. It contains the selected immutable `CommandOps`, typed
device state, retained device ownership, originating allocator and native
buffer identity, fixed reference and generated-work scratch, native recording
state, submission linkage, and the sole lifecycle state. Copies of a public
token alias the authoritative record; they do not copy or fork its state.
Device loss is reported by lifecycle operations; explicit discard remains
available so a lost device can release retained command state.

State transitions:

```text
INACTIVE -> RECORDING
RECORDING <-> RECORDING_RENDER_PASS
RECORDING -> EXECUTABLE -> SUBMITTING
SUBMITTING -> EXECUTABLE          (failed submission)
SUBMITTING -> SUBMITTED
SUBMITTED -> INACTIVE             (ordered completion retirement)
RECORDING/EXECUTABLE -> INACTIVE  (discard)
```

`begin_commands(allocator)` claims one inactive device command cell, pops one
preallocated allocator index, pairs them, initializes the record while
inactive, and publishes `RECORDING` last. If either fixed capacity is exhausted,
it returns `DEVICE_BUSY` without allocation or state change. Render passes nest
into `RECORDING_RENDER_PASS` and return to `RECORDING` on end. `end_commands`
closes the same record to `EXECUTABLE`.

For a nonempty submit, the backend reads each direct token exactly once under
one command-lock transaction. It compares the reuse generation and validates
device ownership, authoritative phase, duplicate epoch, and the exact
`Queue` stored by the command record before claiming the complete batch as
`SUBMITTING`. The public wrapper performs no preliminary executable-token
resolution, and submission does not resolve an allocator merely to re-prove its
immutable queue. Duplicate detection visits each inspected token once, so
ordinary work is proportional to the batch length. Epoch rollover accounts
separately for the command-table cells it resets.

Validation, preparation, or native failure publishes no pending record or
completion point and preserves tokens, readiness, allocator units, and scratch
for retry. A fault after the complete claim restores every still-`SUBMITTING`
record to `EXECUTABLE` under one rollback command-lock acquisition. After native
acceptance, each command record embeds its completion metadata and links into
the exact selected queue's intrusive pending list. The queue publishes the
completion sequence only after the pending records and `SUBMITTED` states are
visible, then consumes the caller tokens.

The selected queue's long submission boundary covers fixed-scratch preparation,
native submission, pending-list append, readiness commit, and completion-point
publication. A separate short retirement boundary protects its intrusive list
and covered-record release. Polling an earlier published point can therefore
retire it while a later same-queue submission is paused before publication.
Ordered retirement first moves each covered record to `INACTIVE`, then releases
record-owned references and reservations, returns each buffer/scratch index to
its originating allocator, releases retained ownership, clears its embedded
pending link, and finally invalidates or generation-advances its private table
identity. A submission may mix allocators only when every record names the exact
submit queue. The next begin completely reinitializes a retired unit before
publishing `RECORDING`.

Warm begin, recording, end, submit, discard, retirement, and reuse never allocate
host storage, native command buffers, pools, VMA storage, or C3 temporary-pool
memory. Per-list reference or generated-index exhaustion returns
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` transactionally. Device teardown rejects
even a quiescent live public allocator. Invalid transitions return faults, and
render-pass command constraints remain enforced. A render pass records its
attachment formats and sample count; a graphics pipeline must match them before
begin or bind mutates native command state. Resolve and pass boundaries do not
add implicit synchronization.

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
and per-list retained-index ceiling. The reservation query's maximum count
covers every smaller recording without a second native query, and even a
zero-byte native requirement consumes one retained reservation slot. Warm
recording returns
`GENERATED_SCRATCH_EXHAUSTED` instead
of allocating when the count or matching-slot supply is exhausted. Discard
and completion return reserved buffers to the same allocator; another allocator
never acquires them implicitly.

Pipeline bind generation-checks the public handle, resolves the cache entry,
optionally retains tracked ownership, and publishes a complete bound snapshot: native
pipeline and layout, kind, render compatibility, cache identity, generated-work
layout, and public diagnostic identity. Later draw, dispatch, render-pass,
indirect, and generated commands use that logical snapshot without reading a
pipeline cell, pipeline table, or cache. Native graphics and compute pipeline
snapshots are tracked independently by bind point, so switching bind points
does not lose either native selection. Ending a render pass clears only active
render compatibility; a compatible logical graphics binding may remain selected
for a later pass. Command-policy variants are outside
this mechanism: policy selection happens when the device or record is created,
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
`gpu::internal::vk` and never cross the private implementation boundary.

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

### Shader input and pipelines

Pipeline descriptors embed `ShaderDesc` directly. Its SPIR-V bytes, entry point,
and debug name are borrowed only until the synchronous creation call returns;
a null entry point selects `main`. The enclosing compute, vertex, or fragment
field supplies the expected role. Each device interns the exact role, normalized
entry point, length, and bytes into one owned private identity represented by a
compact ID. Debug names do not participate in identity, and no surviving state
retains caller pointers.

Native shader modules are temporary pipeline-construction details, never public
resources. After the private shader identity is retained, selected-entry
reflection validates the field-derived role, descriptor convention, and, when
a push-constant block is declared, its complete generated root ABI. Validation
completes before native module creation, cache insertion, or pipeline
publication; a fault releases the retained identity and leaves no surviving
store mutation.

Pipelines are immutable shader execution objects. Creation is split by kind:

```text
create_compute_pipeline(device, ComputePipelineDesc)    -> PipelineHandle?
create_graphics_pipeline(device, GraphicsPipelineDesc)  -> PipelineHandle?
```

Graphics pipeline identity includes shaders, the ordered color-format domain,
depth format, sample count, and polygon mode. Blend equations and write masks
are complete caller-owned command packets and never enter the cache key.
Topology, cull mode, front face, depth bias, depth state, viewport, scissor,
and color state are command-time state.
Compute pipelines share one device-owned `RootPush` layout and, when
generated work is available, one generated-dispatch layout. Repeated singular
creation deduplicates pipeline identity. Pipeline-cache entries own refcounted
IDs, and the device recycles an identity after its last cache entry releases it.
The pipeline cache fronts a serializable driver cache:
`get_pipeline_cache_size` /
`get_pipeline_cache_data` export the driver blob, and
`RuntimeDesc.pipeline_cache_data` warm-starts it for devices created from that runtime.

### Synchronization

Each successful submission returns a reusable `CompletionPoint`. Cross-queue
submission dependencies use consumer-scoped `SubmitDesc.completion_waits`:
ordinary stages compose with a narrow draw-argument consumer. Same-queue order
is implicit after point and scope validation. Swapchain readiness remains
stage-scoped. Native timelines and swapchain semaphores remain backend-private.
Texture layout transitions remain resource-specific because a global barrier
carries neither texture identity nor subresource range.

### Swapchains

Swapchains are optional. Headless compute and offscreen graphics must work without a swapchain.

A swapchain borrows a runtime-owned `Surface`. Its images are ordinary borrowed
texture handles used by shared transitions, render passes, and command lifetime
validation. SDL3 supplies native handles to the platform surface module in
samples; SDL types do not enter the core API.

## 5. Direct private implementation

Device operation flow:

```text
public Device token -> pinned typed VkDeviceState* -> private Vulkan function
```

Public functions validate and pin the token, then call private Vulkan functions
directly with the typed state. Each Vulkan runtime owns surface discovery and
optional debug instance dispatch. A device retains only the native dispatch
groups required by its request and loads only those groups. Headless devices
create no presentation queue, mutex, table, or native dispatch. Vulkan state
pointers and native dispatch declarations remain private.

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

Under `FULL`, the backend retains explicitly named
spans, textures, attachment views, allocations, and pipelines across recording,
executable, and incomplete-submission phases; destruction returns
`RESOURCE_IN_USE` until discard or retirement releases the reference. Under
`TRUSTED`, records contain no reference storage and destruction adds no
implicit wait or deferred release. The caller must observe completion before
destroying every referenced owner. GPU addresses and shader indices are opaque
to the command stream under either setting, so their lifetime always remains a
caller precondition.

FULL allocators give each command scratch unit one fixed sequential reference
list sized at allocator creation. Duplicate detection scans the accumulated
list and compares the complete owner/index/generation identity. A new unique
entry retains the resource once and stores its canonical retained counter, so
discard, rollback, and completion retirement release directly through that
counter. Compound recording operations preflight their unique candidates
against the remaining list capacity and roll back only entries appended after
their checkpoint. Capacity failure therefore leaves prior references and
native command state unchanged.

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

Vulkan device creation selects one immutable command table from the contract
mode: TRUSTED or FULL. The authoritative record
stores the chosen pointer once; repeated `cmd_*` calls do not inspect contract,
layer, callback, naming policy, or separate command-capability fields.
Both tables retain mandatory host
pointer/slice/range safety, overflow protection, internal state integrity,
public ownership, Vulkan result handling, and rollback. Detailed command misuse
outside that floor is a caller contract violation unless `FULL` is selected.
One-shot use, alias confinement, and non-fabricated token storage remain caller
preconditions. Static device-slot identity, record generation, and
authoritative-phase checks reject stale or consumed aliases before native
mutation, including after the originating device's record storage is released.

The direct recording path retains the record-owned runtime `CommandOps`
dispatch and fallible `cmd_*` signatures.

Focused behavioral tests invoke every command family through both operation
tables. Allocator, work, resolution, reference-state, fault, and native-emission
observations are the authority for warm behavior; there is no source-policy
scanner. Timing is blocking only for an explicitly pinned runner, driver, and
comparison profile.

The shader-visible descriptor heap is device-global and bound lazily on the
first pipeline selection in a command record. Its descriptor-indexing set 0
binds once per used graphics or compute bind point. The two bind-point cache
bits remain independent when commands alternate between compute and graphics.
Ordinary pipeline changes, descriptor publication, draws, dispatches,
and global barriers do not replay that setup. Fresh command records start with
empty binding state after native command-buffer reset.

### Compute

```text
cmd_bind_pipeline(command_list, pipeline)
cmd_dispatch(command_list, root_gpu, groups)
```

Binding selects the active compute pipeline and descriptor heap. A valid
dispatch uses that active kind, pushes the root address unchanged, and executes
without native pipeline creation. Zero is valid under every validation policy;
the shader branches before dereference unless the application relies on
defined robustness behavior.

### Graphics

```text
render_geometry_state(width, height)
cmd_begin_render_pass(command_list, render_pass_desc)
cmd_bind_pipeline(command_list, pipeline)
cmd_set_graphics_state(command_list, graphics_state)
cmd_set_raster_state(command_list, raster_state)
cmd_set_depth_state(command_list, depth_state)
cmd_set_color_state(command_list, color_state)
cmd_set_viewport(command_list, viewport)
cmd_set_scissor(command_list, scissor)
cmd_draw(command_list, vertex_root, fragment_root, vertex_count, instance_count)
cmd_draw_indexed(command_list, vertex_root, fragment_root, index_span, index_count, instance_count, index_type = IndexType.U32)
cmd_end_render_pass(command_list)
```

Minimal pass begin validates, lowers, and tracks only attachments, then emits
one native begin-rendering command. It does not change command-buffer graphics
state. A failed begin leaves the command outside a pass and retains no new
attachment reference.

`cmd_set_graphics_state` emits that complete packet before or during a pass on
a graphics-capable command list after a compatible pipeline is bound. The
raster, depth, color, viewport, and scissor setters remain optional partial
updates in either recording phase. Graphics state persists across compatible
pipeline binds and render-pass boundaries until another setter supplies it; an
incompatible color-format domain clears color readiness, and a minimal begin
never replays or replaces it. Under `FULL`,
regular and generated draws require one successful complete packet in the
current command-buffer recording. Fresh command-buffer reuse clears that
initialization. Dynamic state remains outside pipeline keys, so handle aliasing
cannot overwrite caller-selected command state. Draws require an active
graphics pipeline, push both roots unchanged, and perform no native pipeline
creation.

The canonical fresh sequence is begin, bind, set, draw, and end. If an
incompatible pipeline remains selected from an earlier pass, bind the next
compatible pipeline before begin. A failed complete-state update after begin
leaves the pass and its attachment references active; callers may retry the
setter, end the pass, or discard the recording.

Direct, indirect, and generated compute and graphics work share that ABI rule:
every root argument or generated-record root field is forwarded unchanged,
including zero, independent of contract policy. Shader control flow owns zero
dereference safety.

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

The caller owns texture history. `TextureState.layout` is an operational
recorded requirement: `before` asserts the layout established by earlier
ordering, and `after` names the layout required by the next use. The backend
never consults or updates a shared layout table. Identity, range, and
safe-lowering checks remain active under every policy, while `FULL` adds
semantic layout, stage, access, usage, queue, and presentation diagnostics.
Each accepted barrier resolves its texture and range once, lowers its two
states once, and emits one native barrier.

Every device uses one private native mapping: `UNDEFINED` maps to
`VK_IMAGE_LAYOUT_UNDEFINED`; transfer source and destination map to their
respective transfer-optimal layouts; sampled color maps to
`VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`; sampled depth/stencil maps to
`VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL`; storage maps to
`VK_IMAGE_LAYOUT_GENERAL`; color and depth attachments map to their respective
attachment-optimal layouts; and `PRESENT` maps to
`VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`. Texture barriers, descriptors, rendering
attachments, and buffer-image copies all use this same mapping. There is no
device-selected alternative and no hidden state tracking.

Presentation barriers separate external WSI ordering from queue-side access.
Leaving `PRESENT` uses no source access and anchors the layout transition to
the paired first consumer stage covered by the acquire readiness wait. Entering
`PRESENT` preserves the last queue producer and uses destination `NONE`/`NONE`;
the existing presentation semaphore orders the external consumer. Public
`PRESENT` stages/access remain empty, and swapchain sharing and queue-family
rules are unchanged.

Render pass boundaries do not imply shader-read or transfer-read readiness.

## 9. Descriptor heap model

The public API exposes device-wide view and sampler publication:

```text
create_texture_view(device, texture, view_desc) -> TextureView?
destroy_texture_view(device, view) -> void?
create_texture_views(device, descs, out_views) -> void?
intern_sampler(device, desc) -> SamplerIndex?
```

Every device initializes one inline, device-owned heap group. Runtime
configuration selects texture and sampler capacities; descriptor objects,
native features, dispatch, and pipeline state remain private.

The backend keeps one append-only sampler table per device.
The public frontend validates sampler semantics before calling the Vulkan
implementation. Enabled
anisotropy is accepted only in the inclusive range `[1,
DeviceCaps.max_sampler_anisotropy]`; accepted values are preserved exactly in
the canonical key, while inactive anisotropy and comparison values normalize to
zero. Invalid requests cannot reach lookup, native creation, or publication.
Canonical sampler keys are byte-hashed into fixed power-of-two buckets; a hash
match still requires complete canonical equality. Interning holds the resource
mutex and publishes the native sampler, descriptor, stable index, cell, and
bucket link as one transaction. Device teardown destroys every native sampler
and releases both slot and bucket storage.

The backend implements the heap with one update-after-bind descriptor set.
Adapter support and device creation require the descriptor-indexing feature
set and validate the exact runtime-configured texture and sampler capacities
against cached per-type, per-stage aggregate, and all-pools limits. Native-limit
mismatches return `UNSUPPORTED_FEATURE`; capacities are never clamped.
Descriptor updates remain independent from command-buffer binding state and
require no public descriptor synchronization hazard.

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
internal, partial-initialization, and device-loss state under `FULL`.
A callback only selects structured delivery; it does not change contract
checks, tracking, leak scans, layers, or naming. Vulkan layer messages remain
native routing and are independent of library contract diagnostics.

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
