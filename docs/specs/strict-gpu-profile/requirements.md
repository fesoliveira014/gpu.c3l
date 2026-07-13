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
- Placed textures with requirements queried before creation.
- Device-wide strict texture and sampler heaps with backend-neutral indices.
- Root-pointer shader inputs and address-based direct and indirect work.
- Explicit pipeline creation with small, deterministic identity.
- Transient one-shot command recording.
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

### Queues and commands

- Device requests describe queue roles and counts without backend family indices.
- Applications retrieve explicit queue handles from the created device.
- Command recording begins from a queue and is safe across worker threads.
- Recording storage and native command pools remain private.
- Command lists are one-shot.
- Successful submission consumes submitted command tokens.
- Abandoned recording has an explicit discard operation.
- No recorded command performs a global device-registry lock.

### Memory and resources

- An owning GPU allocation has one explicit release operation.
- A `GpuSpan` cannot release its parent allocation.
- CPU mappings and GPU addresses are reported only when valid for the selected memory class.
- Generic GPU data, copies, fills, index data, indirect arguments, upload, and readback use spans or addresses rather than `BufferHandle`.
- Texture requirements are available before texture creation.
- Texture creation validates allocation compatibility, size, alignment, and offset before backend mutation.
- Destroying a texture does not release its placement.
- Samplers are immutable device-interned values and are released with the device.
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
- Viewport and scissor are dynamic.
- Baseline blend state remains part of graphics pipeline creation.
- Native pipeline compilation never occurs during draw or dispatch.
- Optional GPU-generated work and root records are exposed as an explicit semantic capability and are never emulated through a CPU loop.

### Synchronization and rendering

- Generic GPU-memory barriers describe global execution and memory hazards.
- Generic-memory barriers contain no resource handles or ranges.
- Texture transitions name a texture and semantic previous and next uses, not Vulkan layouts.
- No barrier or texture transition is inferred.
- Debug validation may track expected texture state without changing release semantics.
- Render-pass begin/end never add implicit synchronization.
- Render passes name attachments, load/store operations, and clear values directly.
- Vulkan 1.2 render-pass and framebuffer objects may be synthesized privately while preserving the same public behavior.
- Swapchain images use the shared texture, transition, render-pass, and queue model.

### Compatibility extension

- `gpu::compat` owns descriptor layouts, descriptor arenas, descriptor sets, descriptor writes, compatibility pipelines, and binding commands.
- Descriptor arenas express allocation and reset lifetime without exposing native descriptor pools.
- Compatibility pipeline types remain distinct from strict pipeline types.
- Both pipeline families operate on shared command lists and resources.
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
- Hot command recording performs no registry lock or hidden allocation per command.
- Allocation and placed-texture ownership tests cover every destruction order.
- Public strict source and generated shader ABI contain no buffer-binding objects.
- Shader ABI tests pin address and shader-visible index widths.
- Pipeline tests prove shared IR deduplication and absence of draw-time compilation.
- Synchronization tests cover transfer, shader, indirect, descriptor, color, depth, host, and presentation hazards.
- Compatibility tests cover descriptor arena reset, persistent set release, batched writes, mixed strict/compat command recording, and compatibility-only devices.
- Vulkan 1.2 and 1.3 paths that implement identical semantics pass the same public tests.
- Samples teach strict usage first and keep compatibility samples focused.
- Benchmarks measure allocation, descriptor work, pipeline creation, command recording, submission, barriers, and indirect execution.
