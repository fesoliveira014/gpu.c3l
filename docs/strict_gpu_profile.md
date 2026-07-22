# Strict GPU Architecture

This document summarizes the strict architecture contract for `gpu`. It follows
the pointer-first, bindless, explicit-synchronization model described in
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api).

The contract includes separately designed extensions that are not present in the
current source; it is not a current API inventory. See [Public API](api.md) and
[Architecture](architecture.md) for the implemented surface. Detailed requirements
and design are in [requirements](specs/strict-gpu-profile/requirements.md) and
[design](specs/strict-gpu-profile/design.md). The
[task list](specs/strict-gpu-profile/tasks.md) records implementation status and
verification commands.

## Public topology

- `gpu` owns the canonical runtime, adapter, device, queue, memory, texture, command, synchronization, rendering, presentation, and strict pipeline APIs.
- `gpu::compat` adds explicit descriptor-set capabilities and compatibility pipelines. It does not preserve or wrap the current API.
- `gpu::surface::<platform>` creates platform surfaces and returns the shared `gpu::Surface` type.
- `gpu::internal::vk` is the single private Vulkan backend module. Compatibility implementation code, if added, remains private within that module.
- Importing any module performs no runtime initialization.

Vulkan types, feature names, queue families, layouts, descriptor mechanisms, and result codes do not appear in the GPU-shaped API.

## Runtime and devices

`gpu::Runtime` owns backend discovery, adapters, surfaces, and diagnostics. Applications enumerate adapters before device creation, query semantic support, build a `DeviceRequest`, and create a device from one adapter.

Capability groups are explicit and immutable. A strict request enables only the strict contract. `gpu::compat` can add descriptor-set requirements to that same request. A capable device may enable both groups, but importing `gpu::compat` or detecting descriptor-set support enables nothing by itself.

The library supports multiple live devices. Devices, queues, and resources use generational ownership. Each device owns its backend state, capability state, dispatch tables, completion state, and resource tables. Queue requests and resource access domains use semantic roles rather than backend queue-family indices.

Device destruction never waits. It faults while children, incomplete queue work, or active operations remain, and changes the device generation only after successful destruction.

Backend API and driver versions are diagnostic information. Applications select semantic capabilities, not Vulkan versions.

## Memory and resources

- An owning GPU allocation exposes its size, memory class, GPU address, and optional CPU mapping.
- A `GpuSpan` is a checked, non-owning slice of an allocation.
- Mapped-span flush and invalidate operations define CPU/GPU visibility and are no-ops for coherent memory.
- Generic GPU data uses addresses and spans rather than public buffer objects.
- Copies, fills, index data, indirect arguments, uploads, and readback operate on spans or GPU addresses.
- Texture requirements are queried before creation and report whether dedicated backing is required.
- Placed texture creation validates caller-provided memory before mutation.
- Dedicated texture creation transactionally publishes separate texture and allocation tokens.
- Sampler descriptions intern directly to stable device-lifetime shader indices and require no individual destruction. Compatibility-only devices have no strict sampler heap and reject interning before backend mutation.
- VMA remains private.
- Non-WSI resource destruction is immediate. No live recording command list, executable command token, or incomplete submission may reference the resource. Strict presentation integration applies the same no-hidden-wait rule to swapchain destruction and resize.
- Readback uses a caller-owned `CPU_READ` allocation and span, copy, completion
  point, mapped-span invalidation, and direct CPU access.

Explicit transfers use caller-owned `CPU_WRITE` and `CPU_READ` allocations,
mapped visibility operations, commands, and completion points. Short-lived and
long-lived data use the same ownership contract: retain each allocation through
the completion that covers its last use. Readback waits or polls before
invalidation and CPU access.

The explicit transfer path requires no root-level application work boundary,
public semaphore, or readback-ticket API.

## Shader data and pipelines

Strict shaders receive root GPU addresses and access textures and samplers through device-wide shader-visible heaps. The public API exposes heap semantics and fixed-width shader indices, never the backend descriptor mechanism.

Shader IR is supplied through reusable CPU-side `ShaderCode` values. The library computes their content identity. Pipeline creation may be batched to deduplicate shared IR. There is no public shader-module handle.

Pipeline creation performs compilation explicitly. Compute pipelines contain compute code. Graphics pipelines contain shader code and the raster state that affects compilation or attachment compatibility. Depth/stencil state is separate; baseline blend state remains part of the graphics pipeline. Viewport and scissor are dynamic.

Pipeline binding is separate from draw and dispatch. Draw and dispatch commands carry root addresses and execution arguments, not pipeline handles.

## Commands and synchronization

- Command lists are transient and one-shot.
- Recording storage is device-managed and safe for concurrent recording.
- Successful submission consumes command tokens and returns a compact queue-owned `CompletionPoint`.
- Completion points support host poll/wait and stage-scoped cross-queue waits;
  same-queue order is inherent after point and stage validation.
- Failed submission publishes no point and preserves retryable command tokens.
- Creating a completion point allocates no public synchronization object.
- Allocations and textures declare admitted queue roles; cross-family resources use backend-managed concurrent sharing rather than inferred ownership transfers.
- Buffer and pointer-visible memory hazards use global execution and memory barriers.
- Texture representation changes use explicit semantic transitions.
- No barrier, transition, or render-pass dependency is inferred.
- Render passes name attachments, load/store operations, and clear values
  directly.
- The backend requires Vulkan 1.3 plus `VK_EXT_extended_dynamic_state3` and
  `dynamicPrimitiveTopologyUnrestricted == VK_TRUE`; it does not synthesize a
  lower-version render-pass fallback.
- Swapchain acquisition uses one-shot readiness; presentation consumes the
  acquired image and accepts its render completion point. Native
  synchronization remains private.

## Compatibility extension

`gpu::compat` exposes only semantic differences that applications must author explicitly:

- descriptor layouts;
- descriptor arenas;
- descriptor sets;
- batched descriptor writes;
- compatibility pipeline types;
- descriptor-set binding commands;
- descriptor-set shader interfaces.

Compatibility pipelines use the shared device, queues, command lists, memory, textures, samplers, completion points, lifetime rules, render passes, and presentation APIs. Strict and compatibility pipelines may alternate in one command list when both capability groups were requested.

The library does not translate shaders, emulate root pointers through
descriptors, silently change binding models, or provide a Vulkan 1.2 fallback.

## Required invariants

- `gpu` evolves in place; the current API is not moved wholesale into `gpu::compat`.
- Strict operations never fall back to compatibility behavior.
- Compatibility state exists only on devices that explicitly requested it.
- Device creation either enables the complete request or publishes no device.
- Native pipeline compilation never occurs implicitly during draw or dispatch.
- Public synchronization does not require resource lists for generic GPU memory.
- Non-WSI resource and device destruction never hide waits or deferred release; strict presentation extends that rule to swapchain destruction and resize.
- Strict completion and readback require no root-level work lifecycle or public synchronization objects.
- Public documentation and generated API references remain backend-neutral.
