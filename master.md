# gpu.c3l Master Architecture

**Library name:** `gpu.c3l`  
**Public module:** `gpu`  
**Primary backend:** Vulkan 1.3  
**Language target:** C3 0.8.0  
**Required backend bindings:** `vk.c3l`, `vma.c3l`  
**Windowed sample/test binding:** `sdl3.c3l` (`module sdl`, dependency name `sdl3`)  
**Memory allocator:** Vulkan Memory Allocator through `vma.c3l`  
**Document role:** master architecture and document map

---

## 1. Executive summary

`gpu.c3l` is a C3 library that exposes a compact, explicit, modern GPU programming model. It is designed for engines, renderers, research projects, tools, and sample applications that want direct GPU control without carrying a traditional graphics API abstraction into application code.

The design follows the "No Graphics API" direction argued by Sebastian Aaltonen (<https://www.sebastianaaltonen.com/blog/no-graphics-api>): modern bindless GPUs with 64-bit addressing and coherent caches no longer need the descriptor-set, root-signature, and resource-binding machinery that DX12/Vulkan/Metal carry. `gpu.c3l` applies that direction concretely — a single root GPU pointer per draw/dispatch, buffers by GPU address, textures/samplers by heap index, transient command lists, and explicit stage-based synchronization — on top of a Vulkan 1.3 backend.

The public API is intentionally small. User code writes root data into GPU-visible memory, passes a single GPU address to draw or dispatch commands, accesses buffer data by GPU address, accesses textures and samplers by compact heap indices, and records explicit synchronization. Descriptor sets, descriptor pools, descriptor set layouts, Vulkan image memory requirements, Vulkan pipeline layouts, VMA allocations, and platform window details stay behind backend or sample boundaries.

The first backend is Vulkan. The Vulkan backend lives under `module gpu::vk` and uses:

```text
vk.c3l     -> Vulkan 1.3 C3 binding, imported as module vk
vma.c3l    -> Vulkan Memory Allocator C3 binding, imported as module vma
sdl3.c3l   -> SDL3 binding for windowed samples/tests only, imported as module sdl
```

The core shader ABI is:

```text
push constants:
    root_gpu_address : uint64

root data in GPU memory:
    std430-compatible structs
    GpuAddress fields for buffers and tables
    TextureIndex fields for sampled/storage textures
    SamplerIndex fields for samplers
```

The library should prove the architecture through the following vertical slice before building a full graphics feature set:

```text
1. Create a Vulkan device.
2. Create a VMA allocator.
3. Create addressable VMA-backed buffers.
4. Allocate a root struct from a mapped frame arena.
5. Write input/output GPU addresses into the root struct.
6. Dispatch a compute shader with one root GPU pointer.
7. Read back the result.
```

That slice proves the hard parts first: Vulkan bootstrap, VMA integration, buffer device address, command submission, barriers, root-pointer shader ABI, mapped memory, and readback.

---

## 2. Document set

This master document is supported by topic documents. The documents should be kept in `docs/` in the repository and updated when public API, memory policy, shader ABI, backend behavior, or milestones change.

| Document | Purpose |
|---|---|
| `docs/document_index.md` | Reading order, ownership, document status, and maintenance rules. |
| `docs/architecture.md` | Layering, module boundaries, object model, command model, resource lifetimes. |
| `docs/api.md` | Public API surface, type taxonomy, descriptors, functions, and usage examples. |
| `docs/memory.md` | VMA-backed memory model, memory kinds, arenas, virtual allocator use, budgets, stats. |
| `docs/shader_abi.md` | Root pointer ABI, data layout rules, texture indices, generated structs, GLSL conventions. |
| `docs/vulkan_backend.md` | Vulkan feature requirements, backend mapping, VMA integration, synchronization, swapchain. |
| `docs/testing.md` | Pure CPU tests, headless Vulkan tests, SDL3 windowed tests, validation policy, CI matrix. |
| `docs/style.md` | C3 project conventions: modules, naming, construction/destruction, faults, formatting. |
| `docs/milestones.md` | Milestone plan, deliverables, acceptance criteria, and verification checks. |
| `docs/platforms_and_dependencies.md` | Dependency setup, library manifest policy, sample/test dependencies, platform support. |
| `docs/samples.md` | Sample applications and test harness plan, including SDL3 windowed samples. |

---

## 3. Architectural principles

### 3.1 Public API is GPU-shaped, not Vulkan-shaped

The public API should expose concepts that are stable across modern explicit GPU APIs:

```text
Device
Queue
CommandList
GpuAddress
GpuSpan
BufferHandle
TextureHandle
TextureIndex
SamplerIndex
PipelineHandle
SemaphoreHandle
SwapchainHandle
```

It should not expose backend implementation types:

```text
vk::Device
vk::Buffer
vk::Image
vk::DeviceMemory
vk::DescriptorSet
vk::PipelineLayout
vma::Allocator
vma::Allocation
vma::VirtualBlock
sdl::Window
```

`gpu.c3l` is a library, so the public module boundary is the product boundary. Backends and samples may import platform/vendor bindings; public API signatures should not.

### 3.2 Explicit resource ownership

Every object created by the public API has an explicit destruction path:

```text
create_device             -> destroy_device
create_buffer             -> destroy_buffer
create_texture            -> destroy_texture
create_texture_descriptor -> destroy_texture_descriptor
create_sampler            -> destroy_sampler
create_pipeline           -> destroy_pipeline
create_semaphore          -> destroy_semaphore
create_swapchain          -> destroy_swapchain
```

Frame allocations are reset only when the GPU timeline proves the frame is no longer in flight. Persistent spans are explicitly freed. Backend resources live in generation-checked slot tables.

### 3.3 No automatic synchronization

The library records what the caller asks it to record. It does not infer hazards, perform automatic render graph scheduling, or insert implicit barriers at render pass boundaries.

Convenience helpers may return barrier descriptors or require a declared next-use state, but synchronization remains explicit in user code.

### 3.4 One root pointer per draw or dispatch

A command receives a GPU address to root data. The root struct describes everything the shader needs:

```text
frame data address
pass data address
draw or dispatch data address
material table address
vertex/index data address
texture indices
sampler indices
```

For graphics, vertex and fragment stages may receive separate root addresses:

```text
cmd_draw_indexed(command_list, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count)
```

Cross-stage pointer passing should be avoided. Each shader stage should receive or reconstruct its own root pointer.

### 3.5 Buffer data uses GPU addresses

Shader-dereferenced buffers are created with device-address usage. The backend stores a `vk::DeviceAddress` for each addressable buffer. Public code uses `GpuAddress` or `GpuSpan`, not a raw Vulkan buffer plus offset.

### 3.6 Textures and samplers use heap indices

Textures and samplers are referenced by compact indices:

```text
TextureIndex
SamplerIndex
```

A material record contains texture/sampler indices, not bound resources. The Vulkan backend may implement the heap with descriptor buffers or descriptor indexing. Public shader data remains the same.

### 3.7 Backend compromises stay backend-local

Vulkan still has image layouts, descriptor machinery, queue ownership, swapchains, memory requirements, and pipeline layouts. Those are not reasons to expose Vulkan-shaped concepts publicly. The backend adapts the public model to Vulkan.

### 3.8 Shaders are consumer-owned

`gpu.c3l` ships no application shaders. Shader programs — the actual compute and graphics entry points — are written and owned by the project that calls the library. The library publishes only the *shader-side ABI contract*: generated ABI structs/offsets and the descriptor-heap access helpers, as include files a consumer's shaders `#include` (see `docs/shader_abi.md`). The repository's own samples and tests are consumers too, so their shaders live inside each sample/test, not in a library-owned shader tree.

---

## 4. C3 library structure

`gpu.c3l` should use a library package layout rather than an executable project layout. Source files live at the library root and in module subdirectories. Tests and samples are consumer harnesses and should not be part of the shipped library manifest.

```text
gpu.c3l/
├── manifest.json
├── README.md
├── LICENSE
├── CHANGELOG.md
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
├── render_pass.c3
├── swapchain.c3
├── vk/
│   ├── instance.c3
│   ├── device.c3
│   ├── queue.c3
│   ├── allocator.c3
│   ├── memory.c3
│   ├── buffer.c3
│   ├── texture.c3
│   ├── descriptor_heap.c3
│   ├── shader.c3
│   ├── pipeline_compute.c3
│   ├── pipeline_graphics.c3
│   ├── command.c3
│   ├── sync.c3
│   ├── render_pass.c3
│   ├── swapchain.c3
│   ├── debug.c3
│   └── helpers.c3
├── include/
│   └── shaders/              published shader-side ABI includes (consumer #includes these)
│       ├── descriptor_heap.glsl
│       └── generated/        generated ABI structs/offsets
├── tools/
│   ├── gen_shader_abi/
│   └── shader_build/
├── test/
│   ├── project.json
│   ├── README.md
│   ├── shaders/              test-owned shaders (tests are consumers)
│   └── *.c3
├── samples/
│   ├── project.json
│   ├── shared/
│   ├── hello_compute/        each sample owns its shaders/ subdirectory
│   ├── root_pointer_compute/
│   ├── bindless_texture_compute/
│   ├── offscreen_triangle/
│   ├── hello_triangle_sdl/
│   └── gpu_driven_draw_sdl/
└── docs/
    ├── document_index.md
    ├── architecture.md
    ├── api.md
    ├── memory.md
    ├── shader_abi.md
    ├── vulkan_backend.md
    ├── testing.md
    ├── style.md
    ├── milestones.md
    ├── platforms_and_dependencies.md
    └── samples.md
```

### 4.1 Library manifest

The shipped library manifest should provide `gpu` and depend on backend bindings required by the library implementation:

```json
{
  "provides": "gpu",
  "dependency-search-paths": [ "lib" ],
  "dependencies": [ "vk", "vma" ]
}
```

If C3 manifest behavior requires target-specific dependency sections, keep the same rule: the library depends on `vk` and `vma`; SDL3 is not a library dependency unless a public API begins exposing SDL types, which should be avoided.

### 4.2 Developer project files

`test/project.json` and `samples/project.json` may depend on `sdl3` for windowed paths:

```json
{
  "dependency-search-paths": [ "../lib", ".." ],
  "dependencies": [ "gpu", "sdl3" ]
}
```

Windowed sample source imports `sdl`, not `sdl3`, because the SDL3 binding's C3 module is `sdl` while its package/dependency name is `sdl3`.

---

## 5. Module boundaries

### 5.1 Public module `gpu`

All public types and functions live in `module gpu;`.

Public files:

```text
gpu.c3i                  public declarations and imports expected by consumers
gpu.c3                   root module façade and convenience wrappers
types.c3                 handles, aliases, common structs
faults.c3                public fault definitions
caps.c3                  device capability structs
device.c3                DeviceDesc, Device, create/destroy
queue.c3                 queues and submit descriptors
memory.c3                GpuAddress, GpuSpan, arenas, memory kinds
buffer.c3                BufferDesc and buffer API
texture.c3               TextureDesc, views, texture descriptors
pipeline.c3              shader and pipeline descriptors
command.c3               command list functions
sync.c3                  stages, hazards, barriers, semaphores
render_pass.c3           render target and render pass descriptors
swapchain.c3             optional WSI abstraction
```

### 5.2 Backend module `gpu::vk`

Backend files declare `module gpu::vk;` and import `gpu`, `vk`, and `vma` as needed.

Backend state is private:

```text
VkDeviceState
BufferSlot
TextureSlot
PipelineSlot
DescriptorHeap
FrameArenaState
PersistentArenaState
SwapchainState
```

Public API functions dispatch into backend-local implementation functions through a backend vtable or a `BackendKind` switch. The vtable approach is preferred if additional backends are planned.

### 5.3 SDL sample boundary

SDL3 belongs to samples and windowed tests. It should not appear in public `gpu` signatures.

SDL sample code owns:

```text
sdl::Window
sdl event loop
surface creation inputs
sample-specific resize handling
```

The Vulkan backend should expose a neutral surface creation descriptor rather than taking `sdl::Window*` directly.

---

## 6. Public object model

### 6.1 Device

`Device` owns backend state, queues, resource tables, descriptor heaps, frame arenas, and debug configuration.

Public fields should be minimal. Prefer opaque storage or a pointer to backend state.

```text
Device
    BackendKind backend
    DeviceCaps caps
    void* backend_state
```

Creation descriptor:

```text
DeviceDesc
    BackendKind backend
    bool enable_validation
    bool enable_debug_names
    bool require_descriptor_buffer
    bool allow_descriptor_indexing_fallback
    uint frames_in_flight
    ZString application_name
```

### 6.2 Queues

Queues are selected at device creation.

```text
QueueKind.GRAPHICS
QueueKind.COMPUTE
QueueKind.TRANSFER
```

A queue owns or references:

```text
vk::Queue backend handle
family index
queue index
current timeline value
submit statistics
```

### 6.3 Command lists

Command lists are explicitly begun, recorded, ended, and submitted.

```text
begin_commands(device, QueueKind) -> CommandList?
end_commands(device, CommandList) -> void?
submit(device, SubmitDesc) -> void?
```

Command list states:

```text
EMPTY
RECORDING
CLOSED
SUBMITTED
RETIRED
```

Invalid state transitions return faults.

### 6.4 Handles

Use typed handles backed by `ulong`.

```text
BufferHandle
TextureHandle
PipelineHandle
SamplerHandle
SemaphoreHandle
SwapchainHandle
```

Handles pack:

```text
slot index
generation
optional type tag/debug bits
```

Validation checks:

```text
index in range
slot used
slot generation matches
slot type matches where applicable
```

### 6.5 GPU addresses and spans

`GpuAddress` is a shader-visible 64-bit address. `GpuSpan` is a host-side range object.

```text
GpuAddress
    ulong value

GpuSpan
    GpuAddress gpu
    void* cpu
    usz size
    BufferHandle buffer
    usz offset
    MemoryKind kind
```

`cpu` may be null for non-mappable spans. `gpu` must be non-zero for shader-visible spans.

---

## 7. Memory architecture summary

The public memory taxonomy is simple:

```text
MemoryKind.FRAME_UPLOAD
MemoryKind.PERSISTENT_UPLOAD
MemoryKind.DEVICE
MemoryKind.READBACK
MemoryKind.STAGING
```

The Vulkan backend maps those kinds to VMA allocation policies.

### 7.1 Frame upload memory

Used for:

```text
root structs
small per-frame tables
draw packets
dispatch packets
small constants
```

Properties:

```text
VMA-backed buffer
host-visible
persistently mapped
shader-device-address capable
linear bump allocation
reset after frame timeline retires
```

### 7.2 Persistent upload memory

Used for:

```text
material records
small persistent shader tables
CPU-updated GPU-visible data
```

Properties:

```text
VMA-backed large buffer
optional VMA virtual block for suballocation
mapped when host-visible
addressable
explicit allocate/free
```

### 7.3 Device memory

Used for:

```text
large static buffers
GPU-written buffers
textures
render targets
storage images
```

Properties:

```text
VMA-backed
prefer device-local memory
usually not mapped
updated through staging/copy commands
```

### 7.4 Readback memory

Used for:

```text
compute test results
screenshot/readback buffers
GPU diagnostics
statistics dumps copied from GPU
```

Properties:

```text
VMA-backed buffer
host-visible
prefer host-cached when possible
mapped or map-on-demand
requires invalidate before CPU read when non-coherent
```

### 7.5 Staging memory

Used for:

```text
large buffer uploads
texture uploads
one-shot transfer data
```

Properties:

```text
VMA-backed buffer
host-visible
sequential CPU write
TRANSFER_SRC usage
recycled after transfer queue timeline retires
```

---

## 8. Shader ABI summary

### 8.1 Root pointer

Each dispatch receives one root pointer:

```text
cmd_dispatch(command_list, pipeline, root_gpu, groups)
```

Graphics may use two:

```text
cmd_draw_indexed(command_list, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count)
```

Root pointers are passed through push constants as `uint64` values.

### 8.2 Root struct

Root structs are `std430`-compatible and should be generated once the ABI generator exists.

Manual ABI structs must obey:

```text
no vec3 fields
use Vec4f/Vec4u for packed data
align uint64 values explicitly
add C3 $assert size checks
mirror shader-side constants and field order
```

### 8.3 Buffers

Buffers are referenced by `GpuAddress`, not descriptor binding.

```text
RootArgs
    input  : GpuAddress
    output : GpuAddress
    count  : uint
```

### 8.4 Textures and samplers

Textures and samplers are referenced by indices:

```text
Material
    albedo_texture : TextureIndex
    normal_texture : TextureIndex
    sampler        : SamplerIndex
```

The backend makes the heap visible to shaders.

---

## 9. Vulkan backend summary

The Vulkan backend must enable and use:

```text
Vulkan 1.3
buffer device address
synchronization2
dynamic rendering
timeline semaphores
descriptor indexing fallback
optional descriptor buffer fast path
VMA allocator with buffer-device-address support
```

Backend mapping:

| Public concept | Vulkan/VMA implementation |
|---|---|
| `Device` | `vk::Instance`, `vk::PhysicalDevice`, `vk::Device`, queues, `vma::Allocator` |
| `GpuAddress` | `vk::DeviceAddress` |
| `BufferHandle` | slot containing `vk::Buffer`, `vma::Allocation`, `vma::AllocationInfo`, address |
| `TextureHandle` | slot containing `vk::Image`, `vma::Allocation`, `vk::ImageView`, layout |
| `TextureIndex` | descriptor buffer entry or descriptor indexing array index |
| `CommandList` | `vk::CommandBuffer` |
| `PipelineHandle` | `vk::Pipeline`, `vk::PipelineLayout`, shader metadata |
| `SemaphoreHandle` | timeline `vk::Semaphore` |
| `SwapchainHandle` | `vk::SwapchainKHR`, images, views, surface format |

---

## 10. Testing summary

Use three test layers:

```text
pure CPU tests
headless Vulkan tests
SDL3 windowed samples/tests
```

Pure CPU tests must not require a Vulkan ICD. Headless Vulkan tests require Vulkan but no window. SDL3 tests/samples validate WSI, resize, event handling, and presentation.

SDL3 is a sample/test dependency only:

```text
project dependency: sdl3
C3 import:          import sdl;
```

---

## 11. Milestone summary

The milestone plan is detailed in `docs/milestones.md`. The intended implementation order is:

```text
M0  documentation and architecture freeze
M1  C3 library scaffold
M2  public type system and handles
M3  Vulkan bootstrap
M4  VMA allocator integration
M5  addressable VMA-backed buffers
M6  frame upload arena
M7  persistent arena using VMA virtual allocator
M8  command lists and synchronization
M9  compute root-pointer shader ABI
M10 descriptor heap and bindless texture indices
M11 VMA-backed images and texture upload
M12 offscreen render targets and graphics pipeline
M13 SDL3 windowed swapchain sample
M14 upload/readback helpers
M15 pipeline cache and state policy
M16 GPU-driven indirect execution
M17 debug names, stats, leak reporting
M18 shader ABI generator
M19 cross-platform packaging
M20 first release hardening
```

---

## 12. First implementation target

The first complete proof should be `samples/root_pointer_compute` and its matching headless test:

```text
create_device
create_buffer(input, addressable)
create_buffer(output, addressable + readback path)
alloc_frame_span(RootArgs::size, RootArgs::alignment)
write RootArgs { input_gpu, output_gpu, count }
begin command list
cmd_barrier host write -> compute shader read
cmd_dispatch(root_gpu)
cmd_barrier compute shader write -> transfer/readback
submit and wait/readback
verify output
```

Do not start with presentation. Presentation is valuable, but it does not validate the core architecture as thoroughly as the root-pointer compute slice.
