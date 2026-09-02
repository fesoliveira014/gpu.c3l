# Architecture

## Purpose

`gpu.c3l` exposes a small, explicit GPU API for C3. Its public model is shaped
around devices, queues, allocations, resources, commands, and completion—not
around Vulkan objects. The private backend currently targets Vulkan 1.3.

The design favors predictable ownership and low hidden work:

- imports are runtime-inert;
- creation is transactional;
- state transitions and synchronization are explicit;
- destruction never inserts an unrequested device wait; and
- applications own policies such as pooling, streaming, frame graphs, and
  transient-data reuse.

The library is not an engine and does not provide a render graph, scene model,
resource streamer, frame scheduler, compatibility descriptor API, or automatic
hazard tracking.

## Modules and backend boundary

The public API is `gpu`. Native-window integration is split into
`gpu::surface::wayland`, `gpu::surface::x11`, and
`gpu::surface::win32`. Importing any of these modules creates no runtime,
device, thread, or native object.

Shared implementation lives in private `gpu::internal` modules. Vulkan and VMA
types are confined to `gpu::internal::vk`; no `vk::` or `vma::` type appears in
a public signature. Capability fields describe semantic behavior rather than
the native API version or extension that implements it.

SDL3 is not a library dependency. Applications may use it to create a window
and pass the active platform's native display/window properties to a public
surface module.

## Object and ownership model

Public resources use strongly typed, device-scoped generational handles.
Handles are copyable identifiers, not owning C3 pointers. A handle is accepted
only by the device that created it, and stale generations are rejected when
contract validation can observe them.

The normal lifetime hierarchy is:

```text
Runtime
  Adapter (borrowed immutable view)
  Surface
  Device
    Queue (borrowed immutable value)
    GpuAllocation
    TextureHandle
      TextureView / AttachmentView
    AccelerationStructureHandle
      AccelerationStructureView (TLAS descriptor owner)
    PipelineHandle
    TimestampPool
    CommandAllocator
      RecordingCommands -> ExecutableCommands -> submitted work
    Swapchain
```

Creation calls either publish a complete public object or return a fault with
no live result. Destruction invalidates the supplied owner only on success.
Live children return `RESOURCE_IN_USE`; active operations or incomplete work
may return retryable `DEVICE_BUSY`. Applications quiesce work explicitly with
completion points before teardown.

`ContractValidation.FULL` retains explicitly named resources referenced by a
command list. It does not retain arbitrary memory reached through a GPU
pointer, nor resources named only by shader-visible indices. In
`ContractValidation.TRUSTED`, all such lifetime tracking is caller-owned.

## Runtime, adapters, devices, and surfaces

A `Runtime` owns adapter discovery and optional Vulkan-layer configuration.
`AdapterList` and adapter strings are borrowed from the runtime. A semantic
`DeviceDesc` requests queue roles, presentation, capacities, and validation;
`supports_device_desc` can preflight it, while `create_device` remains the
authoritative operation.

Surfaces retain their runtime but borrow native instance/display/window
objects. A presentation device and swapchain use the exact surface supplied in
the device description. Keep native windowing objects alive until the surface
is destroyed.

Queue roles may alias one native queue. `QueueRequest.single_queue` opts into
one exact native family-and-queue-index identity for every required role. With
a presentation surface, that identity must also be the selected graphics
queue's presentation identity; the private presentation fallback remains
available only when `single_queue` is `false`. A valid policy with no matching
topology is unsupported, while a contradictory policy is invalid.

The private Vulkan backend keeps the request policy through the same canonical
selection used by support preflight and authoritative creation. Alias checks
compare both family and queue index, because family-level presentation support
does not make two different native queues equal. When `single_queue` is false,
the existing required/distinct_roles validation, default normalization, and
preference order—including asynchronous-compute and transfer preferences—
remain unchanged. The API reports semantic queue capabilities and requires the
caller to express every cross-queue dependency.

## Memory and resources

`GpuAllocation` owns storage. `GpuSpan` is a checked, non-owning byte range
inside an allocation. Subspans preserve the allocation identity and bounds.
The public memory classes describe intended behavior:

- `CPU_WRITE`: persistently mapped host-write storage;
- `CPU_READ`: persistently mapped host-read storage;
- `GPU_PRIVATE`: addressable device-local data; and
- `TEXTURE`: storage suitable for placed textures.

Mapped spans expose borrowed mappings. Flush CPU writes before GPU reads and
invalidate before host reads where required. Coherent memory makes those
operations no-ops, but callers use the same API.

`GpuAddress` is the numeric address of an addressable span. Allocations are not
relocated, so the value is stable for the allocation's lifetime. It carries no
owner or generation and becomes invalid as soon as the allocation is freed.
Completion points order use but do not own data reached through an address.

Textures may use dedicated storage or a validated range of a texture-class
allocation. Placed texture creation does not transfer ownership of the
allocation. Sparse texture binding similarly does not create a residency map
or retain arbitrary backing bytes; the application owns residency, overlap,
and retirement policy.

`TextureView` owns a descriptor-heap slot and exposes a raw `TextureIndex`.
Destroying the view recycles that slot immediately. `SamplerIndex` is returned
by device-wide sampler interning and remains stable until device destruction.
Neither index is an ownership token.

Acceleration structures use the same explicit storage model. A BLAS owns
triangle or AABB capacity metadata; a TLAS owns instance capacity. Hidden
creation owns its backing internally, while placed and dedicated forms expose
allocation ownership. Build inputs and scratch are caller-owned spans. Device
clone copies a completed structure into a distinct caller-created matching
destination without hidden storage or scratch.
Capability-gated indirect construction reuses the same descriptors, resource
tables, geometry lowering, and command-unit scratch. Each command unit owns a
fixed dense maximum-count slice, so warm recording allocates nothing. Private
construction state records whether completed counts are exact or only CPU
maxima; direct builds establish exact counts, indirect builds establish
maximum-only knowledge, and clones preserve that distinction.
`AccelerationStructureView` owns a recyclable descriptor slot and exposes a
raw `AccelerationStructureIndex`; packed TLAS instances contain an ordinary
BLAS `GpuAddress`. Neither raw value retains its owner. A cloned BLAS has a new
address, while existing instance bytes keep the source address; a cloned TLAS
needs its own view.

## Shaders and pipelines

Pipeline creation borrows SPIR-V bytes and entry-point strings only for the
call. The backend validates the selected entry, root-push ABI, descriptor-heap
convention, formats, and device capabilities before publishing a private
pipeline identity. Equivalent pipeline descriptions converge through a
device-wide deduplication cache.

Generic shader data is reached through root pointers. Compute commands push
one `GpuAddress`; graphics commands push separate vertex and fragment root
addresses; ray-tracing commands push one address to all six ray stages.
Textures and samplers are selected by raw heap indices stored in root data.
See [Shader ABI](shader_abi.md).

Ray-query shaders explicitly opt in through `ray_query.glsl` and select a TLAS
through binding 5. The ordinary shader path remains extension-free. Procedural
AABB candidates require shader-side intersection and explicit confirmation.
Ray-tracing shaders independently opt in through `ray_tracing.glsl`.
Their pipeline identity encodes the ordered stage and hit-group structure plus
recursion depth. The application owns SBT allocation, packing, synchronization,
and lifetime; command recording emits no hidden allocation or barrier. Direct
tracing supplies dimensions on the host. Capability-gated basic indirect
tracing reads only dimensions from one caller-owned span; the root and SBT
stay direct and use the same ownership model.

Pipeline cache import/export deals with opaque driver data. A cache blob may be
empty or minimally useful on a particular driver; it is an optimization, not
an application compatibility format.

## Commands, queues, and completion

A `CommandAllocator` is bound to one exact selected queue and owns a fixed
number of reusable command units. `begin_commands` returns a one-shot recording
token. `end_commands` consumes it and returns an executable token. Submission
consumes accepted executable tokens; rejection preserves them for retry or
discard.

Successful submission returns a reusable `CompletionPoint`. Polling or waiting
does not consume the point. The point identifies ordered completion on one
queue and can be used as a wait dependency for another queue. It is the
application's fence for allocator reuse, transient-memory reuse, and resource
destruction.

Direct, indirect, and generated work share the same command lifecycle.
Generated work is capability-gated. Each allocator reserves generated work
explicitly while quiescent, naming the pipeline, kind, maximum record count,
and concurrency; the backend owns the private storage those imply.

Acceleration-structure builds reserve fixed geometry/range lowering arrays in
each command unit. Full builds and in-place updates use caller-owned scratch,
while clone uses no scratch. All record no hidden barriers and complete through
the ordinary submission and retirement lifecycle. A completed full build or
clone establishes update eligibility from its completed shape. Full validation
retains exactly the clone source and destination; trusted recording performs no
reference work.

## Synchronization and texture state

Global `Barrier` values express execution and memory dependencies. Texture
layout changes use `TextureBarrier`, which includes subresources and explicit
before/after semantic states. The library does not maintain a hidden layout
history or repair a mismatched transition.

Cross-queue ordering is expressed with completion waits and destination
stages. Queue submission itself is externally synchronized per selected native
queue; aliased semantic roles therefore share that boundary.

The acceleration-structure-build stage orders BLAS/TLAS build, update, and
clone reads/writes. Shader query access belongs to the calling compute, vertex,
fragment, or ray-tracing stage. Applications insert construction-to-
construction, construction-to-query, and cross-submit dependencies explicitly.

Host mapping operations do not imply GPU completion. The application orders a
flush before submission, waits for completion before invalidation/readback,
and keeps all backing allocations alive through the interval.

## Rendering and presentation

Rendering uses dynamic render passes. `RenderPassDesc` supplies attachment
views, load/store behavior, and clear values. A compatible graphics pipeline
and a complete `GraphicsState` must be set before drawing. Viewport and scissor
overrides mutate command-buffer state; pass begin does not create hidden
defaults or replay previous state.

Swapchain acquisition returns an image, the image's prior semantic state, and
one-shot readiness. The application transitions the acquired texture to an
attachment state, renders, transitions it to `PRESENT`, submits while consuming
readiness, and then presents. Acquisition uses a caller-selected timeout and
defaults to nonblocking. Resize and out-of-date recovery are explicit and do
not wait for work behind the application's back.

## Threading model

Public operations fall into three categories:

- **Externally synchronized:** runtime/surface registry mutation, operations on
  one swapchain, and submit/present/sparse bind on one native queue.
- **Thread-safe:** immutable adapter queries, allocation and most resource
  operations, acceleration-structure/view lifecycle operations, pipeline
  creation, completion polling/waiting, and operations on distinct independent
  objects.
- **Thread-confined:** a recording token and all aliases, plus the allocator
  while it has live recordings. Distinct allocators may record concurrently.

Passing an executable token or allocator to another thread requires an
application happens-before edge. Token copies are aliases of one one-shot
record; using aliases concurrently or after another alias consumes the record
is invalid.

### Lock implementation snapshot (not API)

The following describes the current backend so contributors can reason about
contention. Lock names and decomposition are not compatibility promises.

- Device resource state protects creation/destruction and fixed resource
  tables; texture-view heap publication has a subordinate cache domain. TLAS
  view publication and destruction serialize with the target structure under
  the resource domain.
- Each selected queue has a submission domain and a shorter retirement domain.
- Each command allocator has an independent domain; recording is normally
  confined there instead of serialized through a device-wide recording lock.
- A command-record domain protects lifecycle claims and reclamation.

The forward acquisition order is:

```text
device operation pin
  device/resource
    texture-view cache or queue submission
      queue retirement
        allocator
          command record
```

A path never reacquires the device/resource domain while holding a queue,
retirement, allocator, or command-record domain. No path holds two queue
retirement domains at once.

Consequently, recording through distinct allocators can proceed in parallel;
resource operations on independent tables use short device/resource critical
sections; submissions to distinct native queues are independent; and
completion retirement uses the short queue boundary without serializing a
native submit. Operations targeting the same native queue remain serialized.

## Diagnostics and performance characteristics

Full validation adds ownership, generation, state, limit, and retained-resource
checks. Structured callbacks select message delivery; they do not enable or
disable returned faults. Callbacks may run synchronously on arbitrary threads
and must not reenter the library.

The architecture avoids per-call heap allocation on normal recording paths,
preallocates command scratch, uses stable fixed tables, and keeps distinct
allocator recording independent. Full retained-resource validation performs a
linear duplicate scan up to the configured per-list limit. Vulkan validation
layers and software drivers may dominate measurements; record the driver,
validation policy, queue topology, and environment with every benchmark.

## Platform and dependency boundary

The library supports `linux-x64` and `windows-x64`, targets C3 0.8.3, and
requires a Vulkan 1.3 loader, VMA, and SPIR-V reflection support. Binding
packages are vendored submodules and native VMA artifacts use the consuming
target's CRT/link configuration. Shader compilation is an application build
step. See [Getting started](getting_started.md) for the consumer setup and
[Features and limitations](features_and_limitations.md) for the exact
capability profile.
