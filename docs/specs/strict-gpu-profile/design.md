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

`gpu::vk` and `gpu::vk::compat` are private. There is no public `gpu::compat::vk` backend.

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

The device registry replaces the single process-wide active device. Registry mutation during create and destroy is synchronized. Public entry points acquire an active-operation pin while resolving the slot, then recheck its generation and live state before dereferencing backend state. Destruction marks the slot as closing, rejects new pins, and reclaims state only after existing pins retire and the live-child checks pass. A failed destruction attempt restores the live state without changing the generation.

Recording command lists and executable command tokens retain a pin for their lifetime. Hot recording commands therefore use their already-pinned state and perform no registry lookup, atomic pin, or mutation-lock acquisition per command.

A live device slot owns:

- its runtime and adapter identity;
- immutable enabled capabilities and limits;
- backend state;
- the shared backend dispatch table;
- optional compatibility dispatch and state;
- queue slots;
- allocation, texture, view, sampler, pipeline, semaphore, swapchain, and command tables.

Resource tokens are device-local. Public operations accept or derive the owning device and validate the token in that device's table. Passing a valid token to a different device faults before backend mutation.

C3 explicit casts can manufacture the underlying bits of a handle. Runtime ownership and generation validation remain mandatory even where nominal typing prevents ordinary mistakes.

## Queues

`DeviceRequest` contains semantic queue requirements. A request identifies roles, counts, and whether a role must use a distinct queue. The backend maps these requirements to native families and indices.

`Queue` is a device-owned generational handle. A queue reports its semantic roles, not native family data. One queue may satisfy several roles.

Allocation and texture descriptions declare the semantic queue roles that may access them. Operations that explicitly name a span or texture reject it when the recording queue has no admitted role. For allocations reached only through root GPU addresses, using an unadmitted queue role is a caller contract violation because the command stream cannot discover nested pointers. When admitted roles resolve to different native queue families, the backend creates the backing buffer, image, or swapchain with concurrent native sharing. Roles resolved to one family retain exclusive native sharing without a public distinction. Narrow access declarations avoid unnecessary cross-family sharing.

The initial strict core has no exclusive queue-ownership transfer operation. Root-addressed shader access does not identify every allocation to the command stream, so the backend cannot safely infer such transfers. Cross-queue visibility and execution ordering remain explicit through global barriers, texture transitions, and semaphore waits and signals.

Command recording starts from a queue. Submission targets the same queue. Command tokens retain their queue and device ownership, preventing cross-device or cross-queue submission.

The public contract exposes only admitted semantic roles. The backend's same-family or cross-family native sharing selection remains private.

## Memory model

### Allocations

`GpuAllocation` is an owning device-local token paired with immutable metadata:

- size;
- alignment;
- memory class;
- optional CPU mapping;
- optional GPU address range;
- backend-private placement identity.

`free_allocation` consumes the owning token. A copied stale token fails by generation. A `GpuSpan` is non-owning and cannot free memory.

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

### Textures

A texture requirements query runs from `TextureDesc` before creation. It returns size, alignment, placement compatibility, and whether dedicated allocation is required.

Texture creation accepts an allocation and offset. It validates compatibility, bounds, alignment, and live-placement conflicts before native mutation. Destroying a texture releases the texture object and views but not its allocation.

Overlapping live texture placements are rejected initially. Explicit aliasing can be added later only with a complete hazard and lifetime contract.

### Future allocation utilities

Frame arenas, persistent arenas, staging rings, readback pools, and deferred release policies are not strict-core ownership APIs. A future `gpu::alloc` extension may build them from `GpuAllocation` and `GpuSpan`.

Samples may contain local allocator utilities until that extension exists.

## Texture and sampler access

Strict devices initialize one semantic texture heap and one sampler heap. Public configuration requests capacities and shader-visible index widths. The backend chooses descriptor indexing, descriptor buffers, descriptor heaps, or another mechanism that satisfies the same contract.

A texture-view allocation returns:

- a generation-checked CPU ownership token;
- a fixed-width `TextureIndex` stored in shader data.

Destroying the ownership token retires the index until every referencing submission completes. Shader indices contain no generation bits.

`Sampler` is an immutable device-local value returned by interning a semantic sampler description. Identical descriptions return the same value. Native sampler objects live until device destruction, and compatibility descriptor writes accept this value even on a compatibility-only device.

Strict sampler-heap publication is separate from sampler identity. On a device with the strict capability, publishing a sampler returns a stable fixed-width `SamplerIndex` or a heap-capacity fault. The index and its heap entry live until device destruction. Compatibility-only devices create no strict sampler heap and cannot publish or query a `SamplerIndex`.

## Shader code and pipeline creation

### Shader code

`ShaderCode` is a borrowed CPU-side value. `prepare_shader_code` validates the IR and computes an opaque digest. Callers do not provide the digest.

The IR remains immutable and alive while the value is used. The digest is internal, process-local metadata and has no persistence or ABI guarantee. Hash collisions are resolved by length and byte comparison.

One-off creation may accept raw IR and prepare it internally. Reused shaders should be prepared once.

There is no public `ShaderHandle`. Backend shader modules are temporary pipeline-creation objects.

### Pipeline creation

Pipeline creation is explicit and is the only operation that may compile native pipeline code.

- A compute pipeline contains compute shader code.
- A graphics pipeline contains shader code, attachment compatibility, topology, culling, sample state, and baseline blend state.
- Depth/stencil state is a separate immutable object.
- Viewport and scissor are dynamic command state.

Batch pipeline creation deduplicates identical shader modules within the batch and uses the device pipeline cache. It publishes either the documented successful handles or a transactional failure result.

Strict pipeline handles belong to `gpu`. Compatibility pipeline handles belong to `gpu::compat`; their distinct types prevent accidental binding-model crossover.

Binding is separate from execution:

```text
set strict pipeline
set depth/stencil state
draw or dispatch with root addresses
```

Compatibility binding and execution use `gpu::compat` functions because their shader inputs are descriptor sets rather than root addresses.

The backend must not lazily compile an unseen state combination during draw or dispatch. A future separately bindable blend-state capability requires an explicit support and precompilation contract.

## Command lifecycle

Public recording uses two states:

1. a recording `CommandList`;
2. a one-shot executable command token returned by successful end.

`begin_commands(queue)` acquires backend recording storage. Native pools are device-managed and may be cached per worker or sharded internally. They are not public resources.

`discard_commands` consumes an unfinished recording. Successful submission consumes executable tokens. Validation failure before native submission preserves retryable tokens. Device loss invalidates affected tokens.

Concurrent workers may record for the same device. Synchronization may occur during begin, end, and submit. Individual command calls do not take a process-global lock or allocate a public recording context.

Strict and compatibility commands share the same command list. Binding a pipeline records the active binding model. Strict draw and dispatch require a strict pipeline; compatibility draw and dispatch require a compatibility pipeline and the required descriptor sets.

## Synchronization

Generic GPU data uses global barriers. A barrier contains semantic before and after stages plus memory hazards. It contains no buffer, address range, layout, or queue family.

Textures use explicit transitions because representation and presentation state remain observable requirements. A transition contains a texture or view plus semantic previous and next uses. Public texture state never contains a Vulkan layout.

No operation infers a barrier or transition. Debug builds may validate expected texture state, but release behavior is determined entirely by explicit commands.

Cross-queue ordering uses explicit semaphore waits and signals. The backend derives cache operations from semantic hazards. Queue-family ownership transfers are unnecessary because resources admitted to multiple native families use concurrent sharing established at creation.

Render-pass begin/end commands add no barriers.

## Rendering and presentation

A render-pass descriptor names texture views, load/store operations, and clear values directly. There are no public render-pass or framebuffer objects.

Vulkan 1.3 may use dynamic rendering. A Vulkan 1.2 path may synthesize and cache compatible native render passes and framebuffers. This does not justify a compatibility render-pass API because the public semantics are unchanged.

Swapchains belong to a device and surface. Acquired images are borrowed textures with explicit lifetime. Presentation uses the shared queue, semaphore, texture-transition, and render-pass model.

Resize, dormant surfaces, out-of-date swapchains, surface loss, and device loss retain distinct public faults and retry contracts.

## Compatibility descriptor model

`gpu::compat` adds one descriptor-set capability group. It does not add another runtime or device.

### Public types

- `DescriptorLayout` describes bindings and shader visibility.
- `DescriptorArena` owns descriptor-set allocation capacity and lifetime.
- `DescriptorSet` is allocated from an arena.
- Compatibility compute and graphics pipelines reference descriptor layouts directly.

There is no public pipeline-layout object unless an independent semantic need is demonstrated.

Descriptor kinds use GPU-shaped terms for spans, sampled textures, storage textures, and samplers. Batched writes accept shared `GpuSpan`, texture-view, and `Sampler` values.

Transient arenas reset only after the caller-provided synchronization point is complete. Persistent arenas support individual set release. Native pool allocation, fragmentation, and rollover remain private.

### Coexistence

A device may enable strict and descriptor-set capabilities together. Strict and compatibility pipelines can alternate within one command list. Shared resources and synchronization do not change meaning when the binding model changes.

Compatibility shaders are authored for explicit descriptor layouts. The library does not translate strict shader IR, synthesize descriptor layouts from root data, or emulate root pointers.

A Vulkan 1.2 backend enables the core features and extensions needed for the requested public semantics. If a 1.2 implementation can preserve a root operation exactly, the fallback remains in `gpu::vk`. Only descriptor-set authoring and other unavoidable semantic differences belong in `gpu::compat`.

## Backend implementation

The backend selects facilities from semantic requirements and actual feature queries. Version checks are insufficient by themselves.

Examples:

| Public semantic | Possible Vulkan implementation |
|---|---|
| Root GPU addresses | Vulkan buffer device address |
| Global texture and sampler heaps | descriptor indexing, descriptor buffers, or descriptor heaps |
| Global memory barriers | synchronization2 |
| Texture transitions | image memory barriers with private layouts |
| Dynamic render passes | dynamic rendering or cached legacy render passes |
| Placed texture requirements | maintenance4 queries or an equivalent private probe |
| Dynamic raster state | Vulkan core or extended dynamic state |
| Optional GPU-generated work | device-generated commands |

No specific extension name is part of the public contract. The backend can use a promoted core feature, an extension variant, or another exact implementation.

The device owns one shared backend state and common dispatch table. Compatibility adds an optional subtable and state block only when requested. It never owns a parallel backend.

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
- Device creation, texture placement, descriptor writes, and batch pipeline creation are transactional.
- Destroying an owner with live children faults.
- Releasing an allocation with live placed textures faults.
- Descriptor and sampler exhaustion publishes no partial allocation.
- Failed command validation leaves recording state unchanged.
- Submit validation failure preserves executable command tokens.
- Native device loss invalidates the affected device and returns the public device-loss fault.
- Unmapped native failures become backend errors with diagnostic detail, not leaked native result codes.

## Performance constraints

- Device resolution is a lock-free slot and generation check in steady state.
- Command calls perform no process-global lock and no hidden pipeline compilation.
- Recording storage is acquired in batches at command-list begin.
- GPU allocations are explicit so applications can batch and suballocate.
- Shader hashing can be amortized through `prepare_shader_code`.
- Pipeline creation can be batched and deduplicate shared IR.
- Strict descriptor indices are direct shader values; CPU generation metadata is separate.
- Compatibility descriptor arenas amortize native pool management.
- Debug tracking and detailed validation can add cost only when enabled.
- No API promise depends on a particular Vulkan object count or driver cache behavior.

## Implementation order

1. Replace the runtime singleton with `Runtime`, adapters, device requests, multiple device slots, explicit queues, and runtime-owned surfaces.
2. Move the current backend onto per-device shared state without changing binding semantics.
3. Introduce allocations, spans, placed textures, strict heaps, shader-code values, explicit pipeline binding, transient commands, global barriers, and semantic texture transitions in `gpu`.
4. Migrate tests, samples, shader ABI tooling, and benchmarks with each in-place API change.
5. Add `gpu::compat` descriptor layouts, arenas, sets, pipelines, and commands on the shared backend.
6. Add and verify Vulkan 1.2 backend fallbacks after the strict architecture is coherent.

The existing stabilization work remains valid. The superseded wholesale compatibility extraction is not reused.

## Pitfalls and gotchas

- Recursive C3 imports may expose submodule declarations; runtime activation must remain explicit.
- A shared public type does not make every operation valid on every device. Strict operations require the strict capability.
- Device-request extension storage must not expose raw numeric capability identifiers.
- Registry slots must remain generation-safe and pin backend state under concurrent use and destruction.
- C3 explicit casts can bypass nominal typing; runtime validation still protects ownership.
- Generic data and texture placements may have different native memory compatibility.
- Exposed CPU mappings do not imply coherent memory; callers must use the mapped-span visibility operations.
- Exposed GPU addresses prohibit transparent relocation.
- Resource queue access must remain semantic, validate the recording queue, and avoid unnecessary native concurrent sharing through narrow access declarations.
- Pipeline-state separation must not cause hidden draw-time native variants.
- Descriptor-arena reset is unsafe until all referencing submissions retire.
- Vulkan version, feature promotion, and extension presence are not interchangeable.
- Target architecture documents must not be presented as current API documentation before implementation.
- Milestone or review identifiers belong only in planning records, never source identifiers, tests, or comments.

## Verification plan

### Pure CPU

- Handle packing, generation, cross-device rejection, active-operation pinning, and concurrent registry tests.
- Device-request composition and transactional failure tests.
- Allocation-range, mapped-memory visibility, placement, queue-access, descriptor-arena, sampler-interning, strict sampler publication, and shader-hash tests.
- Command and submission state-machine tests.
- Strict/compat nominal type compile-fail fixtures in both directions.
- Import-inert tests for every public module.

### Native backend

- Runtime, adapter, surface, multi-device, queue, and teardown tests.
- Strict-only, compatibility-only, and combined device creation.
- Allocation, placed texture, upload, readback, render, and presentation tests.
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
- Benchmarks for allocation, descriptors, pipeline creation, command recording, submission, barriers, and indirect work.

Hardware-dependent positive claims must record adapter, driver, backend API version, and enabled native features. Missing hardware evidence does not justify advertising unsupported semantics.
