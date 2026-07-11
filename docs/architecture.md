# gpu.c3l Architecture

## 1. Purpose

`gpu.c3l` is a C3 library that exposes a direct GPU programming model suitable for modern explicit rendering and compute workloads. It is not a renderer, render graph, material system, asset system, or platform abstraction layer.

It is a concrete take on Sebastian Aaltonen's "No Graphics API" proposal (<https://www.sebastianaaltonen.com/blog/no-graphics-api>): expose the modern GPU directly — root pointers, bindless heap indices, explicit barriers — instead of the descriptor/binding abstractions designed for ~2012 hardware.

The API centers on four ideas:

```text
GpuAddress      -> shader-visible buffer pointers
GpuSpan         -> CPU/GPU range metadata
TextureIndex    -> shader-visible texture heap index
SamplerIndex    -> shader-visible sampler heap index
```

Draw and dispatch commands pass root GPU addresses. Shaders follow pointers to structured data and use texture/sampler indices for image access. Barriers are explicit. Resource lifetimes are explicit.

## 2. Layer model

```text
application / engine / sample
        |
        v
gpu public API, module gpu
        |
        v
backend dispatch layer
        |
        v
gpu::vk Vulkan backend
        |
        +--> vk.c3l         -> Vulkan API calls
        +--> vma.c3l        -> Vulkan memory allocation
        +--> spvreflect.c3l -> SPIR-V shader reflection
        +--> sdl3.c3l       -> gpu.c3l-samples repository only, not backend public API
```

The public API does not expose backend handles. A Vulkan backend can be replaced or supplemented later without changing shader data structures or most user code.

## 3. Package structure

`gpu.c3l` uses a C3 library package layout. Library source files live at package root and in submodule directories.

```text
gpu.c3l/
├── manifest.json
├── gpu.c3i
├── gpu.c3
├── types.c3
├── faults.c3
├── caps.c3
├── device.c3
├── queue.c3
├── memory.c3
├── buffer.c3
├── texture.c3
├── descriptor_heap.c3
├── shader_abi.c3
├── pipeline.c3
├── command.c3
├── sync.c3
├── swapchain.c3
├── vk/
│   └── *.c3
├── include/
│   └── shaders/        published shader-side ABI includes only (no application shaders)
├── test/
├── tools/
└── docs/
```

### Library files

Files at package root declare:

```c3
module gpu;
```

Backend files under `vk/` declare:

```c3
module gpu::vk;
```

Samples are standalone consumers and may declare their own sample modules.

### Shader ownership

The library ships **no application shaders**. Shader entry points are written and owned by the consuming project. The only shader-side artifacts the library publishes are ABI includes under `include/shaders/` (descriptor-heap helpers and generated ABI structs/offsets) that a consumer's shaders `#include`. Samples and tests own their shaders inside their own trees, because they are consumers like any other.

## 4. Public object model

### Device

`Device` owns all backend resources.

Responsibilities:

```text
backend lifetime
queue ownership
resource slot tables
VMA allocator through backend state
descriptor heaps
frame upload arenas
persistent arenas
readback/staging arenas
pipeline cache
debug and stats state
```

Public shape:

```text
Device
    BackendKind backend
    DeviceCaps caps
    BackendVTable* vtable
    void* backend_state
```

The pointer is opaque. Public code should not inspect it.

gpu.c3l currently supports at most one live `Device` per process. Multiple
devices require ownership information that the present resource handles and
descriptor indices do not encode, so multi-device operation is deferred.

All handles, indices, addresses, spans, command tokens, and synchronization
values are scoped to this device and its runtime lifetime. Passing them to
another device is unsupported. Table- and index-backed values without owner
metadata may resolve a coincident resource rather than returning a fault.
The owner-bearing `CommandList` token has a defensive cross-device rejection,
but that isolated check does not make multi-device operation supported.

### Queues

The API exposes queue kinds rather than raw queue handles:

```text
QueueKind.GRAPHICS
QueueKind.COMPUTE
QueueKind.TRANSFER
```

The backend maps those kinds to Vulkan queue families and queue handles.

### Command lists

A command list is a transient, owner-bearing token for a device-owned command
record. The public token contains only its `Device*` and generation-checked handle;
the Vulkan command buffer, bind cache, pending layouts, context, queue, frame slot,
and lifecycle state remain backend-owned. Copies therefore alias one record.

State transitions:

```text
RECORDING -> RECORDING_RENDER_PASS -> RECORDING -> EXECUTABLE -> SUBMITTING -> consumed
```

`begin_commands` creates a record in `RECORDING`. Render passes nest into
`RECORDING_RENDER_PASS` and return to `RECORDING` on end. `end_commands` closes
the record to `EXECUTABLE`. `submit` atomically preflights and claims the whole
batch as `SUBMITTING`; a pre-queue fault restores it, while success invalidates
every alias. Frame-slot pool reset also invalidates abandoned records. Invalid
transitions return faults, and render-pass command constraints remain enforced.

### Buffers

Buffers are backend-owned resources with optional CPU mapping and optional shader-visible GPU address.

Public uses:

```text
copy source/destination
shader-readable/writable storage
indirect command buffers
index buffers
fixed vertex buffers when needed
arena backing buffers
readback buffers
```

Backend slot:

```text
BufferSlot
    vk::Buffer buffer
    vma::Allocation allocation
    vma::AllocationInfo allocation_info
    vk::DeviceAddress gpu_base
    void* cpu_base
    usz size
    BufferUsage usage
    MemoryKind memory_kind
    ushort generation
    bool used
```

### Textures

Textures represent Vulkan images and their default views.

Public uses:

```text
sampled textures
storage textures
color attachments
depth/stencil attachments
transfer sources/destinations
```

Backend slot:

```text
TextureSlot
    vk::Image image
    vma::Allocation allocation
    vma::AllocationInfo allocation_info
    vk::ImageView default_view
    vk::ImageLayout layout
    Format format
    TextureUsage usage
    uint width, height, depth
    uint mip_levels
    ushort generation
    bool used
```

### Texture and sampler descriptors

`TextureHandle` owns the image. `TextureIndex` is the shader-visible descriptor heap index.

This separation matters:

```text
TextureHandle -> lifetime and commands
TextureIndex  -> shader-visible sampled/storage reference
```

Destroying a texture must invalidate or reject descriptors pointing at it according to the final debug policy. The safer initial policy is to require descriptor destruction before texture destruction in debug builds and report a fault otherwise.

### Pipelines

Pipelines are immutable shader execution objects. Creation is split by kind:

```text
create_compute_pipeline(device, ComputePipelineDesc)   -> PipelineHandle?
create_graphics_pipeline(device, GraphicsPipelineDesc) -> PipelineHandle?
```

Graphics pipelines include the minimum Vulkan-required immutable state. Dynamic viewport/scissor should be used. Blend/depth/raster state should be deduplicated through the pipeline cache. The cache also fronts a serializable driver cache: `get_pipeline_cache_size` / `get_pipeline_cache_data` export the driver blob, and `DeviceDesc.pipeline_cache_data` warm-starts it at device creation.

### Semaphores

Timeline semaphores are the default synchronization primitive.

Public:

```text
SemaphoreHandle
SemaphoreValue
```

Binary semaphores are backend-internal swapchain details unless a public need appears.

### Swapchains

Swapchains are optional. Headless compute and offscreen graphics must work without a swapchain.

A swapchain depends on platform surface creation. Samples use SDL3 to create windows and provide platform handles, but the `gpu` public API should not require `sdl::Window` in signatures.

## 5. Backend dispatch

Preferred backend connection:

```text
Device
    BackendVTable* vtable
    void* backend_state
```

The vtable groups operations by resource type:

```text
create/destroy device
create/destroy buffer
create/destroy texture
create/destroy descriptors
create/destroy pipeline
create/destroy recording contexts
begin/end/submit commands
record commands
upload/readback helpers
create/destroy swapchain
query present-mode support
query stats
```

The public functions perform handle validation and call the backend implementation.

## 6. Resource lifetime

### Creation

Creation functions return fallible values:

```text
Device?
BufferHandle?
TextureHandle?
PipelineHandle?
GpuSpan?
```

Failures return specific faults.

### Destruction

Destruction functions should be explicit and should validate handles.

Initial policy:

```text
invalid handle              -> INVALID_HANDLE
resource still referenced   -> RESOURCE_IN_USE or INVALID_RESOURCE_STATE
valid destruction           -> retire slot and increment generation
```

### Deferred destruction

Vulkan resources cannot be destroyed while in use by the GPU. The backend should maintain per-frame deferred destruction queues:

```text
retire_frame(frame_index)
    destroy resources whose retire_timeline <= completed_timeline
```

Calling `destroy_buffer` removes the public handle immediately, but backend destruction may be deferred.

## 7. Frame model

Each frame-in-flight has:

```text
frame upload arena
command pool(s)
deferred destruction list
last submit timeline value
```

Frame lifecycle and flow:

```text
IDLE --begin_frame--> ACTIVE --end_frame--> IDLE

begin_frame(device)              // valid only in IDLE
    wait if frame slot is still in flight
    reset command pools
    reset frame upload arena
    set VMA current frame index

alloc_frame_span(device, ...)    // valid only in ACTIVE
record work
submit work

end_frame(device)                // valid only in ACTIVE
    record frame timeline value
```

Invalid lifecycle transitions fault `INVALID_RESOURCE_STATE` before changing
the frame slot, arena, pools, retirement state, or queue submissions.

Headless tests may skip swapchain-specific acquire/present steps.

## 8. Command model

`begin_commands` takes an optional `RecordingContextHandle`. One context per worker thread (`create_recording_context` / `destroy_recording_context`) enables concurrent recording; see `docs/threading.md`.

### Compute

```text
cmd_dispatch(command_list, pipeline, root_gpu, groups)
```

The command binds the compute pipeline, pushes the root pointer, and dispatches.

### Graphics

```text
cmd_begin_render_pass(command_list, render_pass_desc)
cmd_set_viewport(command_list, viewport)
cmd_set_scissor(command_list, scissor)
cmd_draw(command_list, pipeline, vertex_root, fragment_root, vertex_count, instance_count)
cmd_draw_indexed(command_list, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count, index_type = IndexType.U32)
cmd_end_render_pass(command_list)
```

Pass begin records full-pass viewport/scissor defaults. The public setters
record one portable pass-bounded rectangle each; their state persists across
pipeline binds until another setter or the next pass begin. Viewport/scissor
remain outside pipeline keys and pipeline-state replay, so handle aliasing
cannot overwrite caller-selected rectangles.

Vertex data can be shader-loaded through GPU addresses. Fixed-function vertex input is allowed for simple paths and compatibility, but it is not the preferred data model.

### Transfer

```text
cmd_copy_buffer
cmd_copy_buffer_to_texture
cmd_copy_texture_to_buffer
cmd_fill_buffer
```

Transfer helpers do not imply next-use barriers.

### Barriers

Barriers are explicit:

```text
cmd_buffer_barrier(command_list, BufferBarrier)
cmd_texture_barrier(command_list, TextureBarrier)
cmd_global_barrier(command_list, GlobalBarrier)
```

Render pass boundaries do not imply shader-read or transfer-read readiness.

## 9. Descriptor heap model

The public API exposes descriptor allocation and updates:

```text
create_texture_descriptor(device, texture, view_desc) -> TextureIndex?
destroy_texture_descriptor(device, index) -> void?
create_sampler(device, desc) -> SamplerIndex?
destroy_sampler(device, index) -> void?
```

Backend implementations:

```text
default (AUTO): descriptor indexing + large descriptor arrays
opt-in: VK_EXT_descriptor_buffer (DescriptorHeapMode.DESCRIPTOR_BUFFER)
```

Neither path changes shader material records.

## 10. Swapchain model

Swapchain operations:

```text
create_swapchain(device, SurfaceDesc, SwapchainDesc) -> SwapchainHandle?
get_swapchain_info(device, swapchain) -> SwapchainInfo?
acquire_next_image(device, swapchain) -> AcquiredImage?
present(device, PresentDesc) -> void?
get_present_mode_support(device, swapchain) -> PresentModeSupport?
retry unchanged on WAIT_TIMEOUT
resize on SWAPCHAIN_OUT_OF_DATE
replace the surface and swapchain on SURFACE_LOST
```

`PresentModeSupport` is a bitstruct reporting fifo/immediate/mailbox availability for the swapchain's surface.

`SwapchainInfo` is the coherent runtime snapshot: selected format, clamped
extent, driver-returned image count, selected present mode, and dormant state.
Successful creation/recreation publishes all fields together. A zero extent
or failed rebuild publishes a queryable dormant sentinel with `UNDEFINED`
format, zero extent/count, and FIFO as the inactive mode value. Consumers
re-query after resize and rebuild format-dependent pipelines if needed.

`AcquiredImage.prior_layout` comes from committed texture-layout state. New
swapchain images report `UNDEFINED`; images that completed the normal
transition/present cycle report `PRESENT`.

Surface creation is platform-specific. The sample harness may provide helper functions that take `sdl::Window*`, but those helpers should live outside the core public API.

## 11. Debug model

Debug builds should track:

```text
resource names
live resource counts
allocation names
allocation user data
slot generation errors
resource state errors
outstanding frame allocations
leaked descriptors
leaked backend objects
```

`destroy_device` should report live resources before destroying the backend.

## 12. Release architecture gate

The architecture is complete enough for a first release when the library supports:

```text
headless root-pointer compute
bindless texture compute
offscreen graphics readback
SDL3 windowed triangle sample
GPU-driven indirect draw sample
VMA memory budget reporting
debug resource leak reporting
shader ABI docs and generated layout checks
```
