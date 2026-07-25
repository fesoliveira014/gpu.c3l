# Strict GPU Architecture Design

## Status

This document defines the approved target architecture. It supersedes the design that moved the current API and backend into `gpu::compat`.

The implementation remains in `gpu` and evolves in place. Compatibility is added after the shared architecture and strict semantics are stable.

## Architectural invariants

1. `gpu` owns the canonical public types.
2. `gpu::compat` is additive and explicitly enabled.
3. Strict and compatibility capabilities share one runtime, device, backend, resource model, command model, synchronization model, and presentation model.
4. A semantic difference gets a distinct public type or function. A backend implementation difference stays private.
5. Capability support, requested capabilities, and enabled capabilities are separate states.
6. Vulkan API version never selects a public capability mode.
7. Imports are runtime-inert.
8. Device creation is transactional.
9. Strict operations never emulate compatibility behavior.
10. Hot recording paths do not acquire a global registry lock or compile pipelines.

## Target module layout

```text
gpu/
├── runtime.c3
├── adapter.c3
├── device.c3
├── queue.c3
├── memory.c3
├── texture.c3
├── sampler.c3
├── shader.c3
├── pipeline.c3
├── command.c3
├── sync.c3
├── render_pass.c3
├── swapchain.c3
├── surface/
│   ├── win32/
│   ├── wayland/
│   └── x11/
├── compat/
│   ├── request.c3
│   ├── descriptor_layout.c3
│   ├── descriptor_arena.c3
│   ├── descriptor_set.c3
│   ├── pipeline.c3
│   └── command.c3
└── vk/
    ├── runtime.c3
    ├── adapter.c3
    ├── device.c3
    ├── queue.c3
    ├── memory.c3
    ├── texture.c3
    ├── descriptor_heap.c3
    ├── pipeline.c3
    ├── command.c3
    ├── sync.c3
    ├── render_pass.c3
    ├── swapchain.c3
    └── compat/
        ├── descriptors.c3
        ├── pipeline.c3
        └── command.c3
```

The final file split may change as code is migrated. Module ownership may not.

`gpu::internal::vk` is private. Compatibility Vulkan code remains in that module; there is no public `gpu::compat::vk` backend.

## Runtime and discovery

### Runtime

`Runtime` owns backend discovery state, diagnostics, adapters, and surfaces. Creating a runtime is the first operation that may initialize native state.

A runtime slot contains:

- a generation;
- backend discovery state;
- diagnostic configuration;
- adapter slots;
- surface slots;
- live surface and device counts.

Runtime destruction faults while a surface or device created from it is live. Adapter handles are borrowed and become stale when their runtime is destroyed. Multiple runtimes are valid. Importing modules creates no runtime and loads no compatibility state.

### Adapters

Adapter enumeration is explicit. `AdapterInfo` reports GPU-shaped information:

- name;
- vendor and device identity;
- integrated, discrete, virtual, or software class;
- semantic memory information;
- queue roles and counts;
- strict support;
- general limits.

Module-specific queries report module-specific support. For example, `gpu::compat` reports descriptor-set support and limits without adding descriptor fields to the root adapter structure.

Backend name, native API version, driver name, and driver version are available through a diagnostic query. They are not inputs to normal device creation.

### Surfaces

`Surface` is owned by `gpu` and belongs to a runtime. Native creation functions live in `gpu::surface::<platform>`, where their platform handle types are meaningful.

The root API contains no `PlatformKind` plus untyped native-handle pair. SDL integration remains outside the core package.

Presentation support is queried against an adapter and a surface before device creation. Adding presentation to a device request names the surface requirements that the selected queues must satisfy.

## Device request composition

`DeviceRequest` is a canonical `gpu` value with private capability storage. Public modules contribute requirements through functions rather than exposing a root enum containing every extension concept.

The intended flow is:

```text
runtime = create_runtime(...)
adapters = enumerate_adapters(runtime)
request = strict_device_request()
gpu::compat adds descriptor-set requirements when requested
supports_device_request(adapter, request)
device = create_device(adapter, request)
```

A compatibility-only request originates from `gpu::compat` but still produces `gpu::Device`.

All contributed capability groups are explicit requirements. Applications query support before adding a group when they want conditional behavior. Merely detecting support or importing a module enables nothing.

`RuntimeDesc` owns backend-neutral device defaults. Adapter strict-support caching and `supports_device_request` evaluate those defaults together with the explicit request, so a supported result describes the same configuration that `create_device` will attempt.

Device creation:

1. resolves the adapter;
2. validates the complete semantic request;
3. selects backend features and queues;
4. allocates private state;
5. initializes only the requested capability state;
6. publishes the device slot.

Failure before publication releases all temporary state. No partial device or partially enabled request is observable.

## Multi-device ownership

`Device` remains a compact strongly typed generational handle. Its packed representation contains a device-slot index and generation.

The device registry replaces the single process-wide active device. Registry mutation during create and destroy is synchronized. Most public entry points acquire a short-lived operation pin while resolving the slot, then recheck its generation and live state before dereferencing backend state.

`destroy_device` is non-blocking and retryable:

1. Validate the live token and reject known live children with `RESOURCE_IN_USE`.
2. Mark the slot closing so new operations return `DEVICE_BUSY`.
3. If operation pins remain, restore the live state and return `DEVICE_BUSY`.
4. Recheck children under the closed state; restore live and return `RESOURCE_IN_USE` if any publication raced closing.
5. Poll queue readiness while new pins are excluded. Restore live and return `DEVICE_BUSY` if work is incomplete.
6. Recheck children, destroy backend state, increment the generation, and release the slot.

Every failed attempt preserves the token, generation, and backend state. Device loss bypasses child and progress checks so unreachable state can be released.

A live device slot owns:

- its runtime and adapter identity;
- immutable enabled capabilities and limits;
- backend state;
- typed private Vulkan state and the shared native device dispatch table;
- optional compatibility dispatch and state;
- queue slots;
- queue-owned completion state;
- allocation, texture, view, sampler, pipeline, swapchain, and command tables.

Resource tokens are device-local. Public operations accept or derive the owning device and validate the token in that device's table. Passing a valid token to a different device faults before backend mutation.

C3 explicit casts can manufacture the underlying bits of a handle. Runtime ownership and generation validation remain mandatory even where nominal typing prevents ordinary mistakes.

## Queues

`DeviceRequest` contains semantic queue requirements. A request identifies roles, counts, and whether a role must use a distinct queue. The backend maps these requirements to native families and indices.

`Queue` is a device-owned generational handle. A queue reports its semantic roles, not native family data. One queue may satisfy several roles.

Allocation and texture descriptions declare the semantic queue roles that may access them. Operations that explicitly name a span or texture reject it when the recording queue has no admitted role. For allocations reached only through root GPU addresses, using an unadmitted queue role is a caller contract violation because the command stream cannot discover nested pointers. When admitted roles resolve to different native queue families, the backend creates the backing buffer, image, or swapchain with concurrent native sharing. Roles resolved to one family retain exclusive native sharing without a public distinction. Narrow access declarations avoid unnecessary cross-family sharing.

The initial strict core has no exclusive queue-ownership transfer operation. Root-addressed shader access does not identify every allocation to the command stream, so the backend cannot safely infer such transfers. Cross-queue execution order uses completion-point waits; visibility and representation changes remain explicit through global barriers and texture transitions.

Command recording starts from a queue. Submission targets the same queue. Command tokens retain their queue and device ownership, preventing cross-device or cross-queue submission.

The public contract exposes only admitted semantic roles. The backend's same-family or cross-family native sharing selection remains private.

### Completion points

`CompletionPoint` is an opaque value of at most two machine words containing queue identity and a monotonically increasing submission sequence. The queue owns the native completion timeline. Creating a point allocates no table entry and exposes no counter management.

The submission shape is:

```text
point = submit(queue, commands, waits)
poll_completion(point)
wait_completion(point, timeout)
```

A successful submit consumes its executable command tokens and publishes the next point for that queue. A retryable failure publishes no point and preserves the tokens. After device loss, command tokens remain discardable so their retained pins can retire.

Waits accept reusable completion points from other queues on the same device. Same-queue order is inherent. Sequence exhaustion faults before native submission, so values never wrap or repeat within a live device. A point remains valid until its device is destroyed; stale or cross-device points fail deterministically.

Completion points also establish host completion for mapped readback, resource reuse, presentation, and compatibility-arena reset. They do not own resources or implement deferred destruction.

## Memory model

### Allocations

`GpuAllocation` is an owning device-local token paired with immutable metadata:

- size;
- alignment;
- memory class;
- optional CPU mapping;
- optional GPU address range;
- backend-private placement identity.

`free_allocation` consumes the owning token immediately. It returns `RESOURCE_IN_USE` without consuming the token when placed textures remain. A copied stale token fails by generation. A `GpuSpan` is non-owning and cannot free memory.

Every non-WSI strict-core `destroy_*` operation is also immediate: it never waits or enters a deferred-release queue. No live recording command list, executable command token, or incomplete submission may reference the resource. Strict presentation integration applies this rule to swapchain destruction and resize.

Debug validation may detect references to explicitly named resources. References reached through GPU pointers or shader heap indices remain caller-managed because the command stream cannot enumerate them.

Mapped allocations report whether CPU/GPU visibility is coherent. `flush_mapped_span` makes completed CPU writes visible to the GPU. After the relevant GPU completion point, `invalidate_mapped_span` makes GPU writes visible to the CPU. Both operations validate that the span belongs to a live mapped allocation, round ranges to backend atom boundaries privately, and become no-ops for coherent memory. Exposing a CPU pointer never implies coherence.

The strict memory classes describe behavior, not Vulkan heaps:

- CPU-writeable GPU data;
- GPU-private data;
- CPU-cached readback;
- texture placement.

An allocation reports which uses it supports. Not every allocation must expose both a generic GPU address and texture placement. Texture requirements identify compatible allocation behavior before allocation or placement.

VMA may allocate native memory and report budgets, but VMA objects and create-resource helpers do not define public ownership.

### Generic GPU data

There is no public `BufferHandle` in the target strict API. Addressable data allocations use private backing buffers where the backend requires them. `GpuSpan` retains enough private allocation and offset identity for copy, fill, index, indirect, descriptor, upload, and readback operations to resolve the native backing object.

GPU addresses are stable for the allocation lifetime. Allocations whose addresses have escaped never move transparently.

Readback is ordinary memory use:

```text
allocate a CPU-cached span
record a copy into the span
submit and obtain a completion point
wait for or poll the point
invalidate the mapped span
read through the CPU mapping
```

### Textures

A texture requirements query runs from `TextureDesc` before creation. It returns size, alignment, placement compatibility, and whether dedicated allocation is required.

`create_placed_texture(device, desc, allocation, offset)` validates compatibility, bounds, alignment, live-placement conflicts, and the dedicated requirement before native mutation. If dedicated backing is required, it faults without mutation.

`create_dedicated_texture(device, desc, allocation_desc)` publishes separate `Texture` and `GpuAllocation` tokens.

The dedicated path privately creates the image first, allocates and binds compatible dedicated memory, then publishes both tokens atomically. Failure releases temporary state and publishes neither token.

Both paths produce ordinary ownership tokens. After completion, the caller destroys the texture before freeing its allocation.

Overlapping live texture placements are rejected initially. Explicit aliasing can be added later only with a complete hazard and lifetime contract.

### Future allocation utilities

Frame arenas, persistent arenas, staging rings, readback pools, and deferred release policies are not strict-core APIs. A future `gpu::alloc` extension may build them from allocations, spans, and completion points.

The root module has no frame token, frame begin/end operation, scoped frame helper, or readback-ticket type.

Samples may contain local allocator utilities until that extension exists.

## Texture and sampler access

Strict devices initialize one semantic texture heap and one sampler heap. Public configuration requests capacities and shader-visible index widths. The Vulkan backend uses one update-after-bind descriptor-indexing set and rejects adapters or configured capacities that cannot satisfy it.

A texture-view allocation returns:

- a generation-checked CPU ownership token;
- a fixed-width `TextureIndex` stored in shader data.

Destroying the ownership token immediately releases the index for reuse. No live recording command list, executable command token, or incomplete submission may reference it. Shader indices contain no generation bits, so stale shader data is a caller lifetime violation.

On a device with the strict capability, `intern_sampler` validates and interns a
semantic sampler description directly into the sampler heap. Identical semantic
descriptions return the same stable fixed-width `SamplerIndex`; debug names do
not participate in identity. Enabled anisotropy is accepted only in the inclusive
range `[1, DeviceCaps.max_sampler_anisotropy]`; over-limit values fault
`INVALID_ARGUMENT` rather than being clamped. Inactive anisotropy and comparison
values normalize to zero. The index, heap entry, and native sampler live
until device destruction. Compatibility-only devices create no strict sampler
heap and reject interning before backend mutation. There is no separate public
sampler identity or publication step.

## Shader code and pipeline creation

### Shader code

`ShaderCode` is a borrowed CPU-side value. `prepare_shader_code` validates the IR and computes an opaque digest. Callers do not provide the digest.

The IR remains immutable and alive while the value is used. The digest is internal, process-local metadata and has no persistence or ABI guarantee. Hash collisions are resolved by length and byte comparison.

One-off creation may accept raw IR and prepare it internally. Reused shaders should be prepared once.

There is no public `ShaderHandle`. Backend shader modules are temporary pipeline-creation objects.

### Pipeline creation

Pipeline creation is explicit and is the only operation that may compile native pipeline code.

- A compute pipeline contains compute shader code.
- A graphics pipeline contains shader code, per-target format/blend/write-mask
  state, depth format, sample state, and polygon mode.
- Depth/stencil state is a separate immutable object.
- Topology, culling, front face, depth bias, viewport, and scissor are dynamic
  command state.
- Every compute pipeline shares the fixed `RootPush` device layout.

Batch pipeline creation deduplicates identical shader modules within the batch and uses the device pipeline cache. It publishes either the documented successful handles or a transactional failure result.

Strict pipeline handles belong to `gpu`. Compatibility pipeline handles belong to `gpu::compat`; their distinct types prevent accidental binding-model crossover.

Binding is separate from execution:

```text
set strict pipeline
set raster state
set depth/stencil state
draw or dispatch with root addresses
```

Compatibility binding and execution use `gpu::compat` functions because their shader inputs are descriptor sets rather than root addresses.

The backend must not lazily compile an unseen state combination during draw or dispatch. A future separately bindable blend-state capability requires an explicit support and precompilation contract.

## Command lifecycle

Public recording uses two states:

1. a recording `CommandList`;
2. a one-shot executable command token returned by successful end.

`create_command_allocator(device, queue, desc)` allocates one exact-queue native pool, a fixed command-buffer set, and stable per-list scratch before publication. `begin_commands(allocator)` draws one unit from that explicit owner and transfers its device-operation pin to the returned token. Native pool and scratch representations remain backend-private.

Recording calls borrow the token's exact retained pin without changing the registry pin count. A nonmatching stale or forged token cannot borrow another command's pin and takes a short validation pin before backend access.

`discard_commands` consumes an unfinished recording, including after device loss, and releases its pin. Successful submission consumes executable tokens, releases their pins, and returns a completion point. A recording or submission failure preserves each token and pin for retry or discard.

Concurrent workers may record for the same device through distinct allocators. One allocator is confined to one recording thread while any recording remains live; executable tokens may cross to a synchronized submit thread. Individual command calls do not take a process-global lock or allocate host/native/VMA/temporary-pool storage.

Strict and compatibility commands share the same command list. Binding a pipeline records the active binding model. Strict draw and dispatch require a strict pipeline; compatibility draw and dispatch require a compatibility pipeline and the required descriptor sets.

## Synchronization

Generic GPU data uses global barriers. A barrier contains semantic before and after stages plus memory hazards. It contains no buffer, address range, layout, or queue family.

Textures use explicit transitions because representation and presentation state remain observable requirements. A transition contains a texture or view plus semantic previous and next uses. Public texture state never contains a Vulkan layout.

No operation infers a barrier or transition. Debug builds may validate expected texture state, but release behavior is determined entirely by explicit commands.

Cross-queue submission waits accept reusable completion points. Barriers and transitions remain explicit. Queue-family ownership transfers are unnecessary because resources admitted to multiple native families use concurrent sharing established at creation.

There is no public semaphore type or user-managed synchronization counter. Native synchronization objects and values are queue-owned backend state.

Render-pass begin/end commands add no barriers.

## Rendering and presentation

A render-pass descriptor names texture views, load/store operations, and clear values directly. There are no public render-pass or framebuffer objects.

Vulkan 1.3 may use dynamic rendering. A Vulkan 1.2 path may synthesize and cache compatible native render passes and framebuffers. This does not justify a compatibility render-pass API because the public semantics are unchanged.

Swapchains belong to a device and surface. Acquisition returns a borrowed texture plus a one-shot readiness token. The rendering submission consumes readiness and returns its ordinary completion point.

Presentation consumes the acquired image and accepts its render completion point. The point remains observable, must belong to the same device, and must cover the rendering work. Texture transitions and render-pass synchronization remain explicit.

Native binary or timeline synchronization used to bridge acquire, submit, and present remains private.

Resize, dormant surfaces, out-of-date swapchains, surface loss, and device loss retain distinct public faults and retry contracts.

## Compatibility descriptor model

`gpu::compat` adds one descriptor-set capability group. It does not add another runtime or device.

### Public types

- `DescriptorLayout` describes bindings and shader visibility.
- `DescriptorArena` owns descriptor-set allocation capacity and lifetime.
- `DescriptorSet` is allocated from an arena.
- Compatibility compute and graphics pipelines reference descriptor layouts directly.

There is no public pipeline-layout object unless an independent semantic need is demonstrated.

Descriptor kinds use GPU-shaped terms for spans, sampled textures, storage
textures, and samplers. Batched writes accept shared `GpuSpan` and texture-view
values. The compatibility sampler-write value requires a separate design before
that extension is implemented; the current public surface exposes no standalone
sampler identity.

Transient arenas reset only after the caller-provided completion point is complete. Persistent arenas support individual set release. Native pool allocation, fragmentation, and rollover remain private.

### Coexistence

A device may enable strict and descriptor-set capabilities together. Strict and compatibility pipelines can alternate within one command list. Shared resources and synchronization do not change meaning when the binding model changes.

Compatibility shaders are authored for explicit descriptor layouts. The library does not translate strict shader IR, synthesize descriptor layouts from root data, or emulate root pointers.

The compatibility per-draw data contract requires a separate design review before compatibility pipeline and command tasks are generated. This architecture exposes no placeholder binding API.

A Vulkan 1.2 backend enables the core features and extensions needed for the requested public semantics. If a 1.2 implementation can preserve a root operation exactly, the fallback remains in `gpu::internal::vk`. Only descriptor-set authoring and other unavoidable semantic differences belong in `gpu::compat`.

## Backend implementation

The backend selects facilities from semantic requirements and actual feature queries. Version checks are insufficient by themselves.

Examples:

| Public semantic | Possible Vulkan implementation |
|---|---|
| Root GPU addresses | Vulkan buffer device address |
| Global texture and sampler heaps | one update-after-bind descriptor-indexing set |
| Global memory barriers | synchronization2 |
| Texture transitions | image memory barriers with private layouts |
| Dynamic render passes | dynamic rendering or cached legacy render passes |
| Placed texture requirements | maintenance4 queries or an equivalent private probe |
| Dynamic raster state | Vulkan 1.3 promoted dynamic state with unrestricted topology classes |
| Optional GPU-generated work | device-generated commands |

No specific extension name is part of the public contract. The backend can use a promoted core feature, an extension variant, or another exact implementation.

The device owns one typed private Vulkan state and shared native device
dispatch table. Compatibility adds an optional subtable and state block only
when requested. It never owns a parallel backend.

## Capabilities, limits, and diagnostics

Root capability reporting contains:

- enabled semantic groups;
- queue roles;
- address and alignment limits;
- texture and render limits;
- strict heap capacities;
- dispatch and indirect limits;
- presentation support.

Requirements guaranteed by the strict group are not repeated as Vulkan-flavored booleans.

`gpu::compat` reports descriptor counts, set counts, binding limits, alignment, and stage visibility through its own query type.

Diagnostic backend information may report backend name, API version, driver name, driver version, and native diagnostic text. Application behavior must not depend on this information when a semantic query exists.

## Failure and mutation policy

- Invalid pointers, handles, ownership, ranges, alignment, state, and request composition fault before backend mutation.
- Unsupported semantic requirements fail request validation or device creation with the unmet requirement identified.
- Device creation, dedicated texture creation, descriptor writes, and batch pipeline creation are transactional.
- Invalid placed texture creation faults before backend mutation.
- Device destruction returns `RESOURCE_IN_USE` for live children and retryable `DEVICE_BUSY` for incomplete work or active operations.
- Non-WSI GPU-visible resource destruction has a caller-completion precondition and never waits or defers; strict presentation uses the same rule for WSI destruction and resize.
- Releasing an allocation with live placed textures faults without consuming it.
- Descriptor and sampler exhaustion publishes no partial allocation.
- Failed command validation leaves recording state unchanged.
- Retryable submit failure preserves executable command tokens and publishes no completion point.
- Native device loss invalidates the affected device and returns the public device-loss fault.
- Unmapped native failures become backend errors with diagnostic detail, not leaked native result codes.

## Performance constraints

- Device resolution is a lock-free slot and generation check in steady state.
- Command calls perform no process-global lock and no hidden pipeline compilation.
- Recording storage is acquired in batches at command-list begin.
- Completion points require no public allocation, table insertion, or caller-managed counter.
- Non-WSI resource and device destruction perform no hidden wait or deferred-release work; strict presentation extends that rule to swapchain destruction and resize.
- GPU allocations are explicit so applications can batch and suballocate.
- Shader hashing can be amortized through `prepare_shader_code`.
- Pipeline creation can be batched and deduplicate shared IR.
- Strict descriptor indices are direct shader values; CPU generation metadata is separate.
- Compatibility descriptor arenas amortize native pool management.
- Debug tracking and detailed validation can add cost only when enabled.
- No API promise depends on a particular Vulkan object count or driver cache behavior.

## Implementation order

1. Replace the runtime singleton with `Runtime`, adapters, device requests, multiple device slots, explicit queues, and runtime-owned surfaces.
2. Move the backend onto per-device shared state and implement non-blocking, retryable destruction.
3. Introduce queue-owned completion points; remove the root frame lifecycle, public semaphores, and readback tickets; migrate presentation and readback.
4. Introduce allocations, spans, placed and dedicated textures, immediate lifetime rules, and future allocator extension points.
5. Add strict heaps, shader-code values, explicit pipeline binding, transient commands, global barriers, and semantic texture transitions in `gpu`.
6. Migrate tests, samples, shader ABI tooling, and benchmarks with each in-place API change.
7. Design compatibility per-draw data, then add `gpu::compat` descriptors, pipelines, and commands on the shared backend.
8. Add and verify Vulkan 1.2 backend fallbacks after the strict architecture and compatibility contract are coherent.

Regenerated tasks retain completed stabilization work as a concise historical record. The superseded wholesale compatibility extraction is not reused.

## Pitfalls and gotchas

- Recursive C3 imports may expose submodule declarations; runtime activation must remain explicit.
- A shared public type does not make every operation valid on every device. Strict operations require the strict capability.
- Device-request extension storage must not expose raw numeric capability identifiers.
- Device destruction must reject known children before closing, reject new pins while closing, then recheck children and queue progress; generation changes only on success.
- C3 explicit casts can bypass nominal typing; runtime validation still protects ownership.
- Generic data and texture placements may have different native memory compatibility.
- Exposed CPU mappings do not imply coherent memory; callers must use the mapped-span visibility operations.
- Exposed GPU addresses prohibit transparent relocation.
- Resource queue access must remain semantic, validate the recording queue, and avoid unnecessary native concurrent sharing through narrow access declarations.
- Caller-managed lifetime includes resources reached through GPU pointers and shader indices; backend validation cannot discover every reference.
- Completion points prove queue progress but do not track resource ownership or make destruction automatic.
- Pipeline-state separation must not cause hidden draw-time native variants.
- Descriptor-arena reset is unsafe until the caller's covering completion point is complete.
- Frame-scoped allocation and deferred release belong in `gpu::alloc`, not the root module.
- Vulkan version, feature promotion, and extension presence are not interchangeable.
- Target architecture documents must not be presented as current API documentation before implementation.
- Milestone or review identifiers belong only in planning records, never source identifiers, tests, or comments.

## Verification plan

### Pure CPU

- Handle packing, generation, cross-device rejection, closing-state transitions, active-operation pinning, and concurrent registry tests.
- Device-request composition and transactional failure tests.
- Allocation-range, mapped-memory visibility, placed and dedicated texture, immediate-release, queue-access, descriptor-arena, sampler interning transactionality and exhaustion, and shader-hash tests.
- Completion-point packing, monotonicity, poll/wait, cross-queue, stale, and failed-submit tests.
- Command and submission state-machine tests.
- Strict/compat nominal type compile-fail fixtures in both directions.
- Import-inert tests for every public module.

### Native backend

- Runtime, adapter, surface, multi-device, queue, and teardown tests.
- Strict-only, compatibility-only, and combined device creation.
- Allocation, placed and dedicated texture, upload, readback, render, and presentation tests.
- Host completion, cross-queue waits, acquire readiness, and present completion tests.
- Global barrier and texture-transition validation.
- Strict and compatibility pipelines alternating in one command list.
- Vulkan 1.2 and 1.3 semantic-equivalence tests where runners support them.
- Validation-clean and leak-free teardown.

### Tooling and documentation

- C3 0.8.0 compilation on Windows and Linux.
- Shader ABI generation and drift checks.
- Generated API scans for backend leakage.
- Documentation link and walkthrough checks.
- Sample builds against the packaged library.
- Strict API scans reject frame lifecycle, public semaphore, and readback-ticket symbols.
- Benchmarks for allocation, descriptors, pipeline creation, command recording, submission, completion polling, barriers, and indirect work.

Hardware-dependent positive claims must record adapter, driver, backend API version, and enabled native features. Missing hardware evidence does not justify advertising unsupported semantics.
