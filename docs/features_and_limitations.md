# Features and limitations

What the library does, what it deliberately does not do, and the fixed
limits an application must design around.

## Features

- Vulkan 1.3 backend behind backend-neutral `gpu` types.
- Runtime and adapter discovery, semantic queue selection, and surfaces for
  Wayland, X11, and Win32.
- VMA-backed allocations, checked subspans, stable GPU addresses, host
  mapping with explicit flush and invalidate, and memory statistics.
- 2D and 3D textures, multisample attachments, sparse color textures,
  bindless 2D, 3D, and sampled cube views, interned samplers, and
  block-compressed BC1, BC3, BC4, BC5, BC6H, and BC7 sampled textures.
- Compute and graphics pipelines from SPIR-V, with deduplication and
  driver pipeline-cache import and export.
- Command allocators with fixed reusable units; transfer, compute, dynamic
  rendering, direct, indirect, and generated work; barriers, timestamps,
  and debug labels.
- Timeline-based completion points, cross-queue waits, swapchain acquire
  and present, and explicit resize recovery.
- Optional full contract validation, structured diagnostics, object naming,
  and leak reporting.
- A schema generator that emits matching C3 and GLSL std430 declarations.
- Opt-in ray queries: triangle and AABB BLAS, TLAS instances, in-place
  updates, device clones, GPU-written build ranges, and a bindless TLAS heap.
- Opt-in ray-tracing pipelines: all six shader roles, triangle and
  procedural hit groups, caller-owned SBTs, direct and indirect traces, and
  per-group stack queries with optional dynamic stack sizing.

## Not provided

- Public Vulkan types or handles.
- Descriptor sets, push descriptors, or binding layouts.
- A render graph, frame allocator, upload ring, readback queue, residency
  manager, or streamer.
- Hidden texture-state tracking, barriers, queue ownership transfers, waits,
  deferred destruction, or memory relocation.
- Render-pass or framebuffer objects; rendering is dynamic.
- A compatibility path for Vulkan versions below 1.3.
- Package-registry distribution; consumers vendor the library.
- Ray-tracing pipeline libraries, capture replay, deferred creation,
  automatic stack sizing, or batched dispatch.
- Acceleration-structure compaction, serialization, host builds, or
  updates into a different destination.

## Required device profile

Vulkan 1.3 with synchronization2, dynamic rendering, timeline semaphores,
buffer device address, descriptor indexing, extended dynamic state, and the
dynamic raster and color state used by `GraphicsState`. An adapter that
lacks any of these fails `create_device` with `UNSUPPORTED_FEATURE`.

Optional capabilities are reported in `DeviceCaps`: asynchronous compute,
indirect-count draws, generated work, line polygon mode, sparse textures,
anisotropy, timestamps, and workload limits. Ray queries and ray-tracing
pipelines are requested in `DeviceDesc` and also need a nonzero
`RuntimeDesc.acceleration_structure_heap_capacity`. A request the adapter
cannot satisfy fails atomically.

Indirect acceleration-structure builds are reported separately by
`AccelerationStructureCaps.indirect_build`. The recording path is covered
by CPU tests but has not run on hardware that reports the capability.

## Ownership and ordering rules

- `GpuAddress`, `TextureIndex`, `SamplerIndex`, and
  `AccelerationStructureIndex` are raw shader values with no owner. Keep the
  allocation, view, or device alive until every use completes.
- A `TextureView` or `AccelerationStructureView` owns a heap slot. Destroying
  it recycles the slot immediately. Sampler indices last until the device
  is destroyed.
- Allocations are never moved. An address is valid until `free_allocation`.
- A `CompletionPoint` orders work. It does not keep anything alive.
- Under `ContractValidation.FULL`, a command list retains resources it names
  by handle until it retires. Under `TRUSTED` nothing is retained.
- Creation is transactional; destruction never waits and may return
  `RESOURCE_IN_USE` or `DEVICE_BUSY`.
- Texture layout history is application state. Each transition names the
  exact prior state.
- Timestamp slot reset, write, and read history is application state.
  Values from different native queues are not calibrated.
- A recording `CommandList` and its aliases are confined to the recording
  thread. Submit, present, and sparse-bind operations on the same native queue
  require application synchronization, including when queue roles alias.
  Completion polling and waiting are thread-safe. See
  [Threading](architecture.md#threading) for the complete contract.

## Resource limits

- Textures are 2D or 3D. A cube view covers six layers of a
  cube-compatible 2D texture and is sampled only. No 1D, array, or
  cube-array views, and no format reinterpretation.
- Multisample textures are 2D attachments with one mip. They are resolved,
  not sampled or copied.
- Depth format is `D32_FLOAT`. There is no stencil.
- Block-compressed textures are sampled-only, single-sampled 2D images.
  They are not storage images, attachments, or sparse textures, and the
  library never encodes, decodes, or generates mips.
- Sparse textures are single-layer, single-sample color 2D or 3D images.
- A texture is either sampled or storage within one layout interval.
- `GraphicsState` has no default. Set a complete state before drawing.
- The ABI schema has no matrix or fixed-array type.
- One BLAS holds only triangles or only AABBs.

## Capacities

| Resource | Default | Maximum |
|---|---:|---:|
| texture heap | 4,096 | 65,536 |
| sampler heap | 256 | 65,536 |
| acceleration-structure heap | 0 (opt-in) | device descriptor limit |
| live textures | 1,024 | 65,536 |
| live acceleration structures | — | 4,096 |
| live pipelines | 256 | configurable |
| swapchains | — | 8 |
| color attachments | — | min(8, device limit) |
| command allocators per device | — | 256 |
| command units per allocator | 8 | 4,096 |
| retained references per list | 64 | 4,096 |
| generated-work reservations per allocator | — | 64 × command units |

Exhaustion returns `DESCRIPTOR_HEAP_FULL`, `SLOT_TABLE_FULL`, or
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`. Nothing is silently clamped.

## Validation and callbacks

`ContractValidation.FULL` checks ownership, generations, call order,
resource references, and semantic limits. It does not prove sparse
residency or the validity of data reached through a `GpuAddress`.

Debug callbacks are synchronous, may run on any thread, and must not call
back into the library. Message pointers are valid only during the call.

## Known environment behavior

| Symptom | Cause | Action |
|---|---|---|
| `create_runtime` returns `UNSUPPORTED_FEATURE` with Vulkan validation on | Khronos validation layers not installed | Install them or set `enable_vulkan_validation = false`. |
| No Vulkan driver in an elevated Windows shell | Elevated processes ignore loader environment variables | Register the ICD in the system registry. |
| Pipeline cache blob is a few bytes on lavapipe | The software driver does not persist shaders | Expected. Measure on the production driver. |
| Multithreaded recording does not scale with Vulkan validation on | The validation layer serializes commands | Benchmark with the layer off. |
| FIFO does not throttle under Xvfb | No real vblank | Do not use Xvfb for pacing measurements. |
| Generated GLSL field fails to compile | The name is a GLSL keyword | Rename the schema field. |
| `SHADER_INVALID` from a root push block | The block is not the exact contract | Declare only the flat address fields, in order. |
