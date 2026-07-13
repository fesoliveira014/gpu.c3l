# Strict GPU Profile Design

## Scope

This design turns the [strict GPU profile](../../strict_gpu_profile.md) into an implementation sequence. It covers the profile boundary, module layout, runtime model, migration order, failure policy, and verification gates.

The current architecture is stabilized first. Its behavior then moves into `gpu::compat`. Strict `gpu` is implemented only after that boundary is established.

## Delivery sequence

| Order | Outcome | Gate |
|---|---|---|
| 1 | Stable current architecture | Current tests, samples, docs, and benchmarks agree. |
| 2 | Isolated compatibility profile | Current behavior works through `gpu::compat`; imports remain inert. |
| 3 | Strict device boundary | Strict types and semantic capability probing exist without fallback. |
| 4 | Placement-first memory | Allocations and resource placement have independent ownership. |
| 5 | Raw descriptor heaps | Shaders use raw contiguous indices; CPU ownership stays separate. |
| 6 | Resource-agnostic hazards | Strict barriers contain no resources or layouts. |
| 7 | Minimal pipeline identity | Dynamic and specialized state no longer multiplies pipeline keys. |
| 8 | GPU-generated root data | Supported devices can generate work and its root data together. |
| 9 | Strict rasterization and presentation | Graphics and presentation use the strict memory and hazard model. |
| 10 | Consumer migration | Strict samples, documentation, tooling, and benchmarks are canonical. |
| 11 | Transitional removal | Strict aliases and obsolete ABI forms are gone. |

Implementation gates are blocking. Compatibility extraction does not begin while current contracts are still changing, and strict implementation does not begin while compatibility types or initialization remain ambiguous. Vulkan 1.2 compatibility is deferrable and does not block the strict sequence.

Positive hardware conformance is a release gate, not a prerequisite for later implementation. CI software drivers must cover compilation, deterministic feature rejection, injected capability tests, and every supported positive path they expose. Hardware runs record the adapter, driver, API version, and extensions used for positive strict paths. A path that no available runner exposes remains pending hardware evidence and cannot be advertised as release-ready, but it does not stall unrelated implementation.

## Target module layout

```text
gpu/
├── gpu.c3i                 strict public entry point
├── *.c3                    strict public types and operations
├── vk/                     strict Vulkan backend
└── compat/
    ├── compat.c3i          compatibility public entry point
    ├── *.c3                stabilized current API
    └── vk/                 Vulkan 1.3 and 1.2+extension backend
```

The manifest continues to provide the `gpu` package. `gpu::compat` is a submodule, not a second package.

C3 imports submodules recursively. `import gpu;` can therefore make compatibility declarations visible. This is acceptable because declarations and imports perform no profile initialization. Only `gpu::compat::create_device` may create compatibility runtime state.

Strict and compatibility devices, resources, commands, pipelines, barriers, descriptors, and faults are declared in their owning modules. Plain value types may be shared only when their meaning and validation rules are identical.

## Stabilization baseline

The existing source, not stale repository guidance, defines the starting implementation. Stabilization reconciles:

- public signatures and generated API documentation;
- backend behavior and ownership;
- command, frame, descriptor, swapchain, and readback lifecycles;
- deterministic hardware-limit validation;
- architecture, API, testing, limitations, and sample documentation;
- pure CPU, Vulkan, windowed sample, and benchmark coverage.

Stabilization fixes current correctness defects but does not introduce strict placement, descriptors, or synchronization. The resulting behavior is the compatibility contract.

## Compatibility extraction

The current public and Vulkan files move into `gpu::compat` and `gpu::compat::vk` as one coherent migration. Do not retain aliases in strict `gpu`; aliases would make the profile types interchangeable and allow compatibility semantics to leak back into the core.

All current tests and samples first move to the compatibility module. This provides a direct proof that the relocation preserved behavior. Strict tests and samples are then added independently.

The compatibility backend preserves the working Vulkan 1.3 path before adding Vulkan 1.2 support. Vulkan 1.2 support is a parallel follow-up expressed as core-or-extension feature loading inside `gpu::compat::vk`; it is not a mode bit on strict device creation.

## Strict runtime model

### Device

The strict device is opaque. Device creation probes semantic requirements rather than exposing backend identity or extension names. Required semantics fault device creation when absent. Optional capabilities are immutable after creation.

### Memory

An allocation owns memory. It exposes size, alignment, memory properties, GPU address, and an optional CPU mapping. Buffer data is represented as checked address ranges. A texture is created over explicit placement after a requirements query and does not free that placement when destroyed.

Frame, persistent, staging, and readback arenas are utilities over the same allocation primitives. An allocation whose GPU address has escaped cannot move without an explicit relocation operation and reference repair.

### Descriptors

The CPU owns generation-checked descriptor allocations. Shaders receive raw fixed-width indices. A range allocation reserves contiguous entries and returns its shader-visible base. Retirement delays reuse until all referencing submissions complete.

The strict Vulkan path uses one resource heap and one sampler heap. Descriptor-set and descriptor-indexing behavior stays in compatibility.

### Synchronization

Strict barriers contain source and destination stages plus memory hazards. They do not contain resources, ranges, image layouts, or queue families. Copy, clear, rasterization, presentation, and host operations provide the resource intent needed for backend-private image and compression handling.

Compatibility retains explicit resource transitions and layout tracking.

### Commands and pipelines

Strict command records are transient and one-shot. Draw and dispatch commands consume root GPU addresses. Pipeline identity contains compiled GPU state only. Specialization data and dynamic state are represented separately and have deterministic cache identity.

Indirect work may use GPU-generated root records only when the semantic capability is present. Strict operations do not emulate this through a draw identifier or CPU loop.

### Rasterization and presentation

Textures remain explicit objects where the CPU must identify render targets, copy operands, and presentation images. Attachment selection does not imply synchronization. Swapchain state and image layouts stay backend-private.

## Vulkan implementation map

| Strict semantic | Backend facility |
|---|---|
| Address-based commands | `VK_KHR_device_address_commands` |
| Resource and sampler heaps | `VK_EXT_descriptor_heap` |
| Layout-free image use | `VK_KHR_unified_image_layouts` |
| Pre-creation texture requirements | Vulkan 1.3 or `VK_KHR_maintenance4` |
| Dynamic pipeline state | `VK_EXT_extended_dynamic_state3` and related support |
| GPU-generated roots and state | `VK_EXT_device_generated_commands` |

These names remain in the backend and backend documentation. Public capabilities use semantic names.

The vendored Vulkan binding does not yet expose every required strict extension. Binding work is incremental and limited to the declarations used by the backend. C names remain only in `@cname` strings; C3 declarations follow the `vk` module conventions and require layout checks for exposed structs.

## Failure policy

- Missing required strict semantics return `UNSUPPORTED_FEATURE` during strict device creation.
- Invalid placement, alignment, range, ownership, state, or lifetime returns a specific public fault before backend mutation.
- Descriptor exhaustion and fragmentation fault without partial publication.
- Failed command recording leaves the command record in its prior valid state.
- Submission success consumes every command alias; pre-submit failure preserves retryable state where documented.
- Profile mismatch is rejected by the C3 type system, not a runtime backend check.
- Compatibility never starts as a side effect of a strict call.

## Migration policy

- Current consumers migrate mechanically from `gpu` to `gpu::compat` after stabilization.
- Compatibility behavior changes only for correctness and must be documented.
- New strict samples do not depend on compatibility helpers or types.
- No transitional strict alias survives the final gate.
- Intermediate development revisions may expose only compatibility functionality, but a release requires at least one complete strict compute path.

## Pitfalls

- Recursive C3 imports make namespace visibility unavoidable; runtime initialization must remain explicit.
- Copying the current API instead of moving it would create two drifting implementations.
- Sharing resource types would erase the profile boundary.
- VMA create-and-allocate helpers must not define strict ownership.
- Raw descriptor indices require separate CPU lifetime tokens and delayed reuse.
- Exposed GPU addresses make transparent defragmentation unsafe.
- Strict image synchronization depends on unified-layout support; compatibility keeps layout transitions.
- Lavapipe is the baseline and negative-path environment. Positive paths for strict extensions run only on CI or recorded hardware that enumerates them; final release evidence names each adapter and driver.
- Existing portability fallbacks belong to compatibility even when they are convenient for strict callers.
- Milestone identifiers remain in planning documents only, never in source names, tests, or comments.

## Verification plan

- Generator drift checks and pure CPU tests run before native setup.
- Compatibility and strict test projects build independently.
- Compile-fail fixtures prove profile types cannot cross.
- Headless Vulkan tests cover device, memory, descriptors, barriers, pipelines, indirect work, readback, and cleanup.
- Windowed samples cover acquire, render, present, resize, surface loss, and teardown.
- Release evidence distinguishes CI-positive, hardware-positive, and pending-hardware strict paths; pending paths are not advertised as supported.
- Generated API documentation is scanned for backend leakage.
- Shader ABI tests pin all CPU/GPU shared layouts.
- Benchmarks compare the stabilized baseline with each strict replacement.
- Linux and Windows CI remain blocking; extension-specific hardware runs are recorded separately when software ICDs cannot exercise them.
