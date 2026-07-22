# Strict GPU Architecture Requirements

## Goal

Evolve the canonical `gpu` API in place into a pointer-first, bindless, explicit GPU interface. Add `gpu::compat` only for semantic capabilities that the strict interface intentionally excludes, with Vulkan 1.2 support remaining secondary.

## Scope and deliverables

- One canonical public type family in `gpu`.
- An explicit runtime, adapter discovery, semantic capability queries, and immutable device requests.
- Multiple live devices with independent backend and resource state.
- Explicit device-owned queues selected by semantic role.
- Runtime-owned platform surfaces created through `gpu::surface::<platform>`.
- Address-based GPU allocations and checked non-owning spans.
- Placed and dedicated textures with requirements queried before creation.
- Device-wide strict texture and sampler heaps with backend-neutral indices.
- Root-pointer shader inputs and address-based direct and indirect work.
- Explicit pipeline creation with small, deterministic identity.
- Transient one-shot command recording with queue-owned completion points.
- Global generic-memory barriers and explicit semantic texture transitions.
- Dynamic render passes shared by strict and compatibility devices.
- An additive `gpu::compat` descriptor-set model using shared devices, resources, queues, commands, synchronization, and presentation.
- Backend-neutral public faults, limits, diagnostics, documentation, tests, samples, and benchmarks.

## Non-goals

- Preserving the current public API unchanged.
- Moving the current API or Vulkan backend wholesale into `gpu::compat`.
- Selecting a library capability mode from the Vulkan API version.
- Automatically enabling supported optional capability groups.
- Silent strict-to-compatibility fallback.
- Shader translation between root-pointer and descriptor-set interfaces.
- Public Vulkan handles, layouts, feature names, queue families, descriptor mechanisms, or result codes.
- A renderer, render graph, material system, shader compiler, asset system, or windowing dependency.
- Allocation-owning convenience arenas in the strict core.
- Implementing the possible future `gpu::alloc` extension in this initiative.
- A strict-core frame lifecycle, deferred-release policy, readback ticket, or public semaphore API.
- Treating complete Vulkan 1.2 coverage as a prerequisite for the strict architecture.

## User-visible behavior

### Runtime and discovery

- Importing `gpu`, `gpu::compat`, or a surface module performs no runtime initialization.
- `create_runtime` explicitly initializes backend discovery and diagnostics.
- Adapters are borrowed runtime-owned handles and require no individual destruction.
- Surfaces belong to a runtime and have explicit lifetimes.
- A runtime cannot be destroyed while a dependent surface or device is live.
- Applications can inspect semantic adapter support before device creation.
- Backend and driver versions are available only through diagnostic queries.

### Device requests and ownership

- A strict request is created through `gpu`.
- `gpu::compat` can add descriptor-set requirements to the same request.
- A request may contain both strict and compatibility requirements.
- Device creation succeeds with the complete immutable request or publishes no device.
- Compatibility state is absent unless explicitly requested.
- Multiple devices may coexist.
- Stale runtime, adapter, surface, device, queue, allocation, texture, pipeline, and command tokens fail deterministically.
- Device-local handles cannot be used with another device.
- Device destruction never waits. Live children produce `RESOURCE_IN_USE`; incomplete queue work or active operations produce retryable `DEVICE_BUSY` without changing the device generation.
- Once destruction begins, new operations fail with `DEVICE_BUSY` until destruction succeeds or the device returns to its live state.

### Queues and commands

- Device requests describe queue roles and counts without backend family indices.
- Applications retrieve explicit queue handles from the created device.
- Command recording begins from an explicit allocator bound to one exact queue.
- Allocator-owned recording storage and native command pools remain backend-opaque; different allocators may record concurrently.
- Command lists are one-shot.
- Successful submission consumes submitted command tokens and returns one queue-owned `CompletionPoint`.
- Submission failure publishes no completion point and preserves retryable command tokens.
- A completion point fits within two machine words and identifies one queue plus a monotonically increasing submission sequence without allocating a public synchronization object.
- Applications can poll or wait for a completion point and can pass points as cross-queue submission waits.
- Same-queue submission order is inherent and requires no explicit wait.
- Abandoned recording has an explicit discard operation.
- No recorded command performs a global device-registry lock.
- Allocation and texture descriptions declare the semantic queue roles that may access them.
- Resources shared by roles on different native queue families use backend-managed concurrent sharing; the core API performs no implicit exclusive-ownership transfer.

### Memory and resources

- An owning GPU allocation has one explicit release operation.
- A `GpuSpan` cannot release its parent allocation.
- CPU mappings and GPU addresses are reported only when valid for the selected memory class.
- Mapped-span flush and invalidate operations define CPU/GPU visibility and become no-ops for coherent memory.
- Generic GPU data, copies, fills, index data, indirect arguments, upload, and readback use spans or addresses rather than `BufferHandle`.
- Texture requirements are available before texture creation.
- Placed texture creation validates allocation compatibility, size, alignment, offset, and dedicated-allocation requirements before backend mutation.
- A placed texture request that requires dedicated backing faults without mutation.
- Dedicated texture creation transactionally creates and binds compatible memory, then publishes separate texture and allocation tokens.
- Destroying a texture does not release its placement.
- Destroying a GPU-visible resource is immediate and never waits or defers. No live recording command list, executable command token, or incomplete submission may reference it.
- Releasing an allocation with live placed textures faults without consuming the allocation.
- On strict devices, sampler descriptions intern directly to stable shader-visible `SamplerIndex` values that are released with the device.
- Compatibility-only devices have no strict sampler heap and reject sampler interning before backend mutation; there is no separate public sampler identity or publication operation.
- VMA types and allocation policies remain private.

### Strict binding and pipelines

- Strict shaders receive one root GPU address per participating shader stage.
- Strict textures and samplers use device-wide shader-visible heaps.
- Heap implementation selection is private.
- Public strict configuration contains no descriptor-indexing, descriptor-buffer, or descriptor-heap mode.
- Shader IR may be reused through a lightweight `ShaderCode` value whose identity is computed by the library.
- There is no public shader-module handle.
- Batch pipeline creation can deduplicate shared shader IR.
- Pipeline creation performs native compilation explicitly.
- Pipeline binding is separate from draw and dispatch.
- Draw and dispatch carry root addresses and execution arguments, not pipeline handles.
- Depth/stencil state is separate from graphics pipeline identity.
- Topology, culling, front face, depth bias, viewport, and scissor are dynamic.
- Per-target format, blend, and write mask remain part of graphics pipeline creation.
- Polygon mode remains immutable graphics pipeline state.
- Compute pipelines use one fixed `RootPush` device layout.
- Native pipeline compilation never occurs during draw or dispatch.
- Optional GPU-generated work and root records are exposed as an explicit semantic capability and are never emulated through a CPU loop.

### Synchronization and rendering

- Generic GPU-memory barriers describe global execution and memory hazards.
- Generic-memory barriers contain no resource handles or ranges.
- Texture transitions name a texture and semantic previous and next uses, not Vulkan layouts.
- No barrier or texture transition is inferred.
- Debug validation may track expected texture state without changing release semantics.
- Render-pass begin/end never add implicit synchronization.
- Cross-queue ordering is explicit through completion-point waits.
- Explicitly named resources reject unadmitted queue roles; queue-role compliance for allocations reached only through GPU pointers is a caller precondition.
- Render passes name attachments, load/store operations, and clear values directly.
- Vulkan 1.2 render-pass and framebuffer objects may be synthesized privately while preserving the same public behavior.
- Swapchain images use the shared texture, transition, render-pass, and queue model.
- Swapchain acquisition returns a one-shot readiness token. Submission consumes readiness, and presentation consumes the acquired image while accepting its render completion point.
- Native presentation synchronization remains private.

### Compatibility extension

- `gpu::compat` owns descriptor layouts, descriptor arenas, descriptor sets, descriptor writes, compatibility pipelines, and binding commands.
- Descriptor arenas express allocation and reset lifetime without exposing native descriptor pools.
- Compatibility pipeline types remain distinct from strict pipeline types.
- Both pipeline families operate on shared command lists and resources.
- Both pipeline families use the shared completion and lifetime model.
- Strict and compatibility pipelines may alternate in one command list when both requirements were enabled.
- Compatibility shaders are authored explicitly for descriptor layouts.
- The library does not translate or emulate the strict shader interface.
- Vulkan 1.2 fallbacks that preserve public semantics remain private backend code.

## Acceptance checks

- Generated public documentation contains no `vk::`, `vma::`, Vulkan feature names, layouts, queue families, native result codes, or backend dispatch types.
- `gpu::compat` extends `gpu`; it does not duplicate the runtime, device, queue, memory, texture, command, synchronization, render-pass, or swapchain APIs.
- One private backend device serves every enabled capability group.
- Import-only tests prove that all public modules are runtime-inert.
- Device-request tests cover strict-only, compatibility-only, combined, unsupported, and transactional failure paths.
- Two devices can create, record, submit, and destroy resources independently in one process.
- Stale and cross-device handle tests fail before backend mutation.
- Concurrent device-use and destruction tests prove that active backend state is never reclaimed.
- Device-destruction tests cover live children, incomplete work, active operations, closing-state rejection, retry, and generation change only after success.
- Hot command recording uses preallocated allocator storage and performs no registry lock, host/native/VMA allocation, or temporary-pool access per command.
- Allocation and placed-texture ownership tests cover every destruction order.
- Dedicated-texture tests prove transactional failure and publication of separate texture and allocation tokens.
- Resource-release tests prove that non-WSI core destruction never waits or defers and rejects live placements; strict presentation tests prove the same for swapchain destruction and resize.
- Mapped-memory tests cover coherent and non-coherent flush/invalidate ranges.
- Readback tests use a CPU-cached span, copy, completion point, mapped-span invalidation, and direct CPU access.
- Queue-access tests cover single-role, same-family multi-role, and cross-family concurrent sharing.
- Public strict source and generated shader ABI contain no buffer-binding objects.
- Shader ABI tests pin address and shader-visible index widths.
- Pipeline tests prove shared IR deduplication and absence of draw-time compilation.
- Synchronization tests cover transfer, shader, indirect, descriptor, color, depth, host, and presentation hazards.
- Submission tests cover compact monotonic queue points, exhaustion, host poll/wait, reusable cross-queue waits, same-queue order, failed submission, stale points, and no per-point allocation.
- The strict public surface contains no `FrameToken`, root frame begin/end helper, `@with_frame`, public semaphore, or readback-ticket type.
- Compatibility tests cover descriptor arena reset, persistent set release, batched writes, mixed strict/compat command recording, and compatibility-only devices.
- Vulkan 1.2 and 1.3 paths that implement identical semantics pass the same public tests.
- Samples teach strict usage first and keep compatibility samples focused.
- Benchmarks measure allocation, descriptor work, pipeline creation, command recording, submission, completion polling, barriers, and indirect execution.
