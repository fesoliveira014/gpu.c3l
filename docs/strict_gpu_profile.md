# Strict GPU Architecture

This document defines the target architecture for `gpu`. It follows the pointer-first, bindless, explicit-synchronization model described in [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api).

This is a target design, not a description of the current implementation. Detailed requirements and design are in [requirements](specs/strict-gpu-profile/requirements.md) and [design](specs/strict-gpu-profile/design.md). The [task list](specs/strict-gpu-profile/tasks.md) remains withdrawn until it is regenerated from this architecture.

## Public topology

- `gpu` owns the canonical runtime, adapter, device, queue, memory, texture, command, synchronization, rendering, presentation, and strict pipeline APIs.
- `gpu::compat` adds explicit descriptor-set capabilities and compatibility pipelines. It does not preserve or wrap the current API.
- `gpu::surface::<platform>` creates platform surfaces and returns the shared `gpu::Surface` type.
- `gpu::vk` is a private shared backend. Compatibility implementation code may live under a private `gpu::vk::compat` submodule.
- Importing any module performs no runtime initialization.

Vulkan types, feature names, queue families, layouts, descriptor mechanisms, and result codes do not appear in the GPU-shaped API.

## Runtime and devices

`gpu::Runtime` owns backend discovery, adapters, surfaces, and diagnostics. Applications enumerate adapters before device creation, query semantic support, build a `DeviceRequest`, and create a device from one adapter.

Capability groups are explicit and immutable. A strict request enables only the strict contract. `gpu::compat` can add descriptor-set requirements to that same request. A capable device may enable both groups, but importing `gpu::compat` or detecting descriptor-set support enables nothing by itself.

The library supports multiple live devices. Devices, queues, and resources use generational ownership. Active operations pin device state so concurrent destruction cannot reclaim it. Each device owns its backend state, capability state, dispatch tables, and resource tables. Queue requests and resource access domains use semantic roles rather than backend queue-family indices.

Backend API and driver versions are diagnostic information. Applications select semantic capabilities, not Vulkan versions.

## Memory and resources

- An owning GPU allocation exposes its size, memory class, GPU address, and optional CPU mapping.
- A `GpuSpan` is a checked, non-owning slice of an allocation.
- Mapped-span flush and invalidate operations define CPU/GPU visibility and are no-ops for coherent memory.
- Generic GPU data uses addresses and spans rather than public buffer objects.
- Copies, fills, index data, indirect arguments, uploads, and readback operate on spans or GPU addresses.
- Textures remain explicit objects and use caller-provided placement.
- Texture requirements are queried before creation.
- Samplers are immutable device-interned values and require no individual destruction. Strict sampler-heap publication returns a separate shader index; compatibility-only devices retain sampler identity without creating the strict heap.
- VMA remains private.

Allocation-owning arenas and policies are outside the strict core. A future `gpu::alloc` module may provide frame, persistent, staging, and readback allocators over the placement primitives.

## Shader data and pipelines

Strict shaders receive root GPU addresses and access textures and samplers through device-wide shader-visible heaps. The public API exposes heap semantics and fixed-width shader indices, never the backend descriptor mechanism.

Shader IR is supplied through reusable CPU-side `ShaderCode` values. The library computes their content identity. Pipeline creation may be batched to deduplicate shared IR. There is no public shader-module handle.

Pipeline creation performs compilation explicitly. Compute pipelines contain compute code. Graphics pipelines contain shader code and the raster state that affects compilation or attachment compatibility. Depth/stencil state is separate; baseline blend state remains part of the graphics pipeline. Viewport and scissor are dynamic.

Pipeline binding is separate from draw and dispatch. Draw and dispatch commands carry root addresses and execution arguments, not pipeline handles.

## Commands and synchronization

- Command lists are transient and one-shot.
- Recording storage is device-managed and safe for concurrent recording.
- Submission consumes successfully submitted command tokens.
- Allocations and textures declare admitted queue roles; cross-family resources use backend-managed concurrent sharing rather than inferred ownership transfers.
- Buffer and pointer-visible memory hazards use global execution and memory barriers.
- Texture representation changes use explicit semantic transitions.
- No barrier, transition, or render-pass dependency is inferred.
- Render passes name attachments, load/store operations, and clear values directly.
- Vulkan 1.2 render-pass and framebuffer objects may be synthesized privately without changing public semantics.

## Compatibility extension

`gpu::compat` exposes only semantic differences that applications must author explicitly:

- descriptor layouts;
- descriptor arenas;
- descriptor sets;
- batched descriptor writes;
- compatibility pipeline types;
- descriptor-set binding commands;
- descriptor-set shader interfaces.

Compatibility pipelines use the shared device, queues, command lists, memory, textures, samplers, synchronization, render passes, and presentation APIs. Strict and compatibility pipelines may alternate in one command list when both capability groups were requested.

The library does not translate shaders, emulate root pointers through descriptors, or silently change binding models. Equivalent Vulkan 1.2 fallbacks remain private backend code.

## Required invariants

- `gpu` evolves in place; the current API is not moved wholesale into `gpu::compat`.
- Strict operations never fall back to compatibility behavior.
- Compatibility state exists only on devices that explicitly requested it.
- Device creation either enables the complete request or publishes no device.
- Native pipeline compilation never occurs implicitly during draw or dispatch.
- Public synchronization does not require resource lists for generic GPU memory.
- Public documentation and generated API references remain backend-neutral.
