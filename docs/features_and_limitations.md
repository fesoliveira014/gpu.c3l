# Features and limitations

This page describes the shipped capability profile and the constraints an
application must design around.

## Features

- Vulkan 1.3 backend behind backend-neutral `gpu` handles and descriptors.
- Runtime and adapter discovery, semantic queue selection, and platform
  surfaces for Wayland, X11, and Win32.
- Independent VMA-backed allocations, checked subspans, stable GPU addresses,
  host mapping, explicit flush/invalidate, and memory statistics.
- 2D and 3D textures, multisample attachments, sparse color textures, views,
  bindless texture indices, and interned samplers.
- Compute and graphics pipelines with direct SPIR-V input, deterministic
  identity, deduplication, and driver pipeline-cache import/export.
- Explicit command allocators and one-shot command lists; transfer, compute,
  dynamic rendering, direct/indirect/generated work, barriers, timestamps, and
  debug labels.
- Timeline-based completion points, cross-queue waits, swapchain acquisition
  and presentation, and resize recovery.
- Optional full contract validation, structured diagnostics, object naming,
  and leak reporting.
- Shader ABI schemas that generate matching C3 and GLSL std430 declarations.
- Explicitly opted-in compute/graphics ray queries with triangle and procedural
  AABB BLAS values, TLAS instances, in-place updates, singular device clones,
  and a bindless TLAS heap.
- Independently opted-in KHR ray-tracing pipelines with all six shader roles,
  triangle/procedural hit groups, caller-owned SBTs, direct traces, and
  capability-gated basic indirect dimensions or GPU-authored SBT/dimension
  packets. Per-group stack queries are available for every ray-tracing pipeline;
  pipelines may explicitly opt in to caller-recorded dynamic stack sizing.

## Deliberate exclusions

- No public Vulkan types or handles.
- No descriptor-set, push-descriptor, or traditional binding-layout API.
- No render graph, frame graph, frame allocator, upload ring, readback queue,
  residency manager, or resource streamer.
- No hidden texture-state tracking, barriers, queue ownership transfers,
  waits, deferred destruction, or automatic memory relocation.
- No traditional render-pass/framebuffer objects or subpasses; rendering is
  dynamic.
- No public compatibility profile for older Vulkan versions.
- No package-registry distribution; consumers vendor the repository and its
  submodules.
- No ray-tracing pipeline libraries/linking, capture replay, deferred creation,
  automatic stack-size derivation, multi/count/batched ray dispatch, or
  GPU-authored root/SBT helpers.
- No acceleration-structure compaction, serialization/deserialization, host
  builds/copies, updates into a distinct destination, handle-preserving
  relocation, or automatic address/view repair. Singular device clone into a
  distinct matching destination is the only copy form.

## Required capability profile

The backend requires Vulkan 1.3 plus the features used by synchronization2,
dynamic rendering, timeline semaphores, buffer device addresses, descriptor
indexing, and extended dynamic state. Device creation also requires the
dynamic raster and color state needed by the public `GraphicsState` model.
Unsupported adapters fail with `UNSUPPORTED_FEATURE`; the library does not
emulate missing capabilities with hidden pipeline variants.

The library targets `linux-x64` and `windows-x64`. A Vulkan loader and the
vendored VMA static library are required. SDL3 is only an application-side
dependency used by the samples. `glslangValidator` or another SPIR-V compiler
is required to build GLSL examples.

Optional capabilities are reported in `DeviceCaps`, including asynchronous
compute, indirect-count draws, generated work, wireframe polygon mode, sparse
textures, anisotropy limits, timestamp support, and workload limits.
Ray queries and ray-tracing pipelines are also optional, but unlike passive
capability discovery they must be requested independently with their
`DeviceDesc` flags and a nonzero runtime acceleration-structure heap capacity.
Either opt-in enables the shared acceleration-structure foundation.
Unsupported requests fail atomically.

Indirect acceleration-structure builds are an independently reported optional
part of that foundation. The current API records one BLAS or TLAS per command
with fixed 16-byte range records. It does not expose batched builds, variable
strides, GPU-authored build descriptions or handles, host builds, compaction,
copy modes beyond device clone, serialization, or capture/replay.

The indirect-build recording path is covered by CPU and compile-time tests but
has not yet been submitted on capable hardware. Tested adapters reported the
capability as unavailable, so hardware testing exercised the explicit direct
fallback.

## Ownership and ordering constraints

- `GpuAddress`, `TextureIndex`, `SamplerIndex`, and
  `AccelerationStructureIndex` are raw shader-visible values. They carry no
  device owner or generation. Keep the owner-bearing allocation, view, or
  device alive until every use completes.
- A `TextureView` owns a recyclable texture-heap slot. Destroying the view
  immediately makes its index reusable. Sampler indices remain stable until
  device destruction.
- An `AccelerationStructureView` owns a recyclable TLAS-heap slot. Wait for
  query completion before destroying it; the view blocks destruction of its
  TLAS while live.
- Allocations are not relocated. A `GpuAddress` remains numerically stable
  while its allocation is alive, but it becomes invalid immediately when that
  allocation is freed.
- `CompletionPoint` orders work; it does not own arbitrary application memory
  reached through GPU pointers or indices.
- Under `ContractValidation.FULL`, command lists retain explicitly named
  resources. For Indirect2, this retains the packet span and pipeline, not the
  raw SBT addresses in the packet. Raw GPU pointers and shader indices remain
  caller-owned. Under `TRUSTED`, all resource lifetime is caller-owned.
- Creation is transactional. A failed create call leaves no live public
  object. Destruction does not insert a hidden device wait and may return
  `RESOURCE_IN_USE` or `DEVICE_BUSY`.
- Texture history is caller-owned. Every layout change requires an explicit
  texture barrier whose `before` state matches prior ordered use.
- Timestamp slots have caller-owned reset/write/read history. Values from
  distinct native queues are not calibrated.
- Command recording is confined to the thread that owns the allocator.
  Submission and completion operations are thread-safe as documented in the
  API reference.

## Resource and rendering limits

- Textures are 2D or ordinary 3D. There are no 1D, cube, 3D-array,
  z-slice-view, or format-reinterpreting views.
- Multisample textures are 2D color/depth attachments with one mip and are not
  sampled, stored, or transferred directly.
- Depth attachments use `D32_FLOAT`; no public stencil attachment state is
  exposed.
- Sparse textures are limited to single-layer, single-sample color 2D/3D
  images. Residency and backing-memory lifetime are entirely caller-owned.
- A texture cannot be sampled and storage-accessed in one layout interval.
  Split the uses and transition explicitly.
- Graphics state has no implicit default. Before a draw, bind a compatible
  pipeline and apply a complete `GraphicsState` whose color packet matches the
  pipeline.
- Generated root records are optional. Shared-root direct and indirect work
  remains available when `DeviceCaps.generated_work` is false.
- The ABI schema has no matrix type. Represent a matrix as vector columns.
- One BLAS contains only triangles or only AABBs. Mixed scenes use separate
  BLAS instances beneath one TLAS. AABB traversal supplies candidates; shaders
  calculate and explicitly confirm procedural intersections.

## Default and maximum capacities

| Resource | Default | Maximum or selected-device bound |
|---|---:|---:|
| texture heap | 4,096 | 65,536 |
| sampler heap | 256 | 65,536 |
| acceleration-structure heap | configured explicitly | selected-device descriptor limits |
| live textures | 1,024 | 65,536 |
| live acceleration structures | — | 4,096 |
| live pipelines | 256 | configurable capacity |
| swapchains | — | 8 |
| color attachments | — | min(8, `DeviceCaps.max_color_attachments`) |
| command allocators per device | — | 256 |
| command buffers per allocator | 8 | 4,096 |
| retained references per command list | 64 | 4,096 |
| generated-work reservations per command list | — | 64 |
| generated-work reservations per allocator | — | 64 x `command_buffer_capacity` |

Heap exhaustion and fixed table exhaustion return explicit faults such as
`DESCRIPTOR_HEAP_FULL`, `SLOT_TABLE_FULL`, or
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`. Device-dependent bounds are exposed by
`DeviceCaps`; calls never silently clamp requested work.

## Validation and callbacks

`ContractValidation.FULL` checks public ownership, generations, call order,
resource references, and many semantic limits. It does not prove residency for
sparse texture regions or ownership of arbitrary data reached through a GPU
address.

Debug callbacks are borrowed, synchronous, and potentially concurrent.
Payload pointers are valid only during the callback. The callback must
synchronize its userdata, return promptly, and must not call back into
`gpu.c3l`, because an internal lock may be held.

## Known environment behavior

| Symptom | Cause | Action |
|---|---|---|
| Runtime creation with Vulkan validation returns `UNSUPPORTED_FEATURE` | Khronos validation layers are unavailable | Install the layers or disable Vulkan-layer validation; full contract validation remains available. |
| Windows Vulkan driver is missing only in elevated shells | Elevated processes may ignore loader environment variables | Register the ICD in the system Vulkan driver registry. |
| Pipeline-cache data is only a small header on lavapipe | The software driver may not persist compiled shaders | Treat it as expected driver behavior; measure on the production driver. |
| Multithreaded recording scales poorly with Vulkan validation enabled | The validation layer serializes command calls | Benchmark without the layer; run correctness gates with it enabled. |
| FIFO presentation does not throttle under Xvfb | A virtual display has no real vblank | Do not use Xvfb presentation timing as a pacing result. |
| A generated GLSL field name fails to compile | The ABI generator does not rewrite GLSL keywords | Rename the schema field. |
| Pipeline creation reports `SHADER_INVALID` for a root block | Reflection requires the generated push-block member shape | Declare the generated fields directly and in schema order. |

For exact constants, descriptors, and faults, see the
[public API reference](api/index.md).
