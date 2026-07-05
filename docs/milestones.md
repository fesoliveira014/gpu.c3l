# gpu.c3l Milestones

Milestones are planning units. Milestone labels are allowed in this document, issue trackers, plans, and commit messages. They must not appear in code identifiers, file names, test names, source comments, or shader symbols.

## M0 — Architecture and documentation baseline

### Goal

Establish the architecture, public API direction, memory policy, shader ABI, backend rules, and project conventions.

### Deliverables

```text
master.md
docs/document_index.md
docs/architecture.md
docs/api.md
docs/memory.md
docs/shader_abi.md
docs/vulkan_backend.md
docs/testing.md
docs/style.md
docs/platforms_and_dependencies.md
docs/samples.md
docs/milestones.md
```

### Acceptance criteria

```text
library is named gpu.c3l
public module is gpu
Vulkan backend module is gpu::vk
VMA is the required memory allocator for the Vulkan backend
SDL3 is documented as sample/windowed-test dependency only
root-pointer shader ABI is documented
explicit synchronization policy is documented
C3 library package layout is documented
```

## M1 — C3 library scaffold

### Goal

Create a buildable `gpu.c3l` library package.

### Deliverables

```text
manifest.json
gpu.c3i
gpu.c3
types.c3
faults.c3
README.md
lib/vk.c3l
lib/vma.c3l
```

### Tasks

```text
create library manifest with provides "gpu"
vendor/pin vk.c3l and vma.c3l
declare dependencies vk and vma
create empty public module
create empty backend module directory
create smoke test harness separate from shipped manifest
```

### Acceptance criteria

```text
library builds as gpu.c3l
consumer can import gpu
no SDL3 dependency is required for core library build
```

## M2 — Public type system and handle discipline

### Goal

Define stable public vocabulary and generation-checked handles.

### Deliverables

```text
types.c3
faults.c3
caps.c3
device.c3
memory.c3
buffer.c3
texture.c3
pipeline.c3
command.c3
sync.c3
test/test_handles.c3
test/test_ranges.c3
```

### Tasks

```text
define BackendKind, DeviceDesc, DeviceCaps
define typed handles
define handle pack/unpack helpers
define GpuAddress and GpuSpan
define MemoryKind
define BufferUsage and TextureUsage
define Format
define public faults
```

### Acceptance criteria

```text
pure CPU tests cover handle round trip
invalid handles are rejected
generation mismatch is rejected
GpuSpan range math is tested
no backend imports appear in public files
```

## M3 — Vulkan bootstrap

### Goal

Create and destroy a Vulkan device with required features.

### Deliverables

```text
vk/instance.c3
vk/device.c3
vk/queue.c3
vk/debug.c3
vk/helpers.c3
test/test_vk_bootstrap.c3
```

### Tasks

```text
create Vulkan instance
enable validation when requested
install debug messenger when available
select physical device
query required features
create logical device
select queues
load extension entry points
populate DeviceCaps
```

### Acceptance criteria

```text
headless test creates and destroys device
required feature absence returns UNSUPPORTED_FEATURE
validation-enabled bootstrap is clean
```

## M4 — VMA allocator integration

### Goal

Create the VMA allocator and route memory stats through it.

### Deliverables

```text
vk/allocator.c3
vk/memory.c3
docs/memory.md update
test/test_vk_vma_allocator.c3
```

### Tasks

```text
create vma::Allocator from instance/physical_device/device
enable buffer-device-address allocator support
enable memory budget support when available
query heap budgets
query stats string
set current frame index during begin_frame
map VMA faults to gpu faults
```

### Acceptance criteria

```text
VMA allocator creates/destroys cleanly
heap budget query works
stats string is non-empty
no buffers/images use raw Vulkan allocation paths
```

## M5 — VMA-backed addressable buffers

### Goal

Create buffers through VMA and retrieve GPU addresses.

### Deliverables

```text
vk/buffer.c3
vk/memory.c3 updates
test/test_vk_buffer_address.c3
```

### Tasks

```text
translate BufferDesc to vk::BufferCreateInfo
translate MemoryKind to vma::AllocationCreateInfo
create buffers with allocator.try_create_buffer
store vma::Allocation and vma::AllocationInfo
map CPU-visible buffers
query vk::DeviceAddress for addressable buffers
set debug names
```

### Acceptance criteria

```text
create device-local buffer
create mapped upload buffer
create addressable buffer with non-zero GpuAddress
destroy buffers through VMA
invalid handle paths return INVALID_HANDLE
```

## M6 — Frame upload arena

### Goal

Allocate transient root data from mapped addressable buffers.

### Deliverables

```text
memory.c3 updates
vk/memory.c3 updates
shader_abi.c3
test/test_vk_frame_arena.c3
```

### Tasks

```text
create one frame upload buffer per frame-in-flight
make each buffer mapped and addressable
implement aligned bump allocator
return GpuSpan
track frame timeline value
reset only after timeline retires
```

### Acceptance criteria

```text
GpuSpan.gpu and GpuSpan.cpu are correct
alignment is honored
overflow returns ARENA_FULL
reset before timeline retire is rejected
```

## M7 — Persistent arena using VMA virtual allocator

### Goal

Support explicit allocate/free of long-lived GPU-addressable spans.

### Deliverables

```text
vk/memory.c3 updates
memory.c3 updates
test/test_vk_persistent_arena.c3
```

### Tasks

```text
create large VMA-backed addressable buffer
create vma::VirtualBlock
allocate aligned virtual ranges
return GpuSpan with backing buffer and offset
free virtual ranges
track fragmentation/stats
```

### Acceptance criteria

```text
persistent spans allocate/free/reuse
alignment is honored
GPU address is stable while allocation is live
virtual block stats update correctly
```

## M8 — Command lists and synchronization

### Goal

Record, submit, and synchronize work explicitly.

### Deliverables

```text
queue.c3
command.c3
sync.c3
vk/queue.c3
vk/command.c3
vk/sync.c3
test/test_vk_command_submit.c3
```

### Tasks

```text
create command pools per queue/frame
begin/end command lists
submit command lists
create timeline semaphore path
translate buffer barriers
translate texture barriers
record copy commands
validate command states
```

### Acceptance criteria

```text
empty command list submits
copy upload -> readback works
timeline semaphore advances
invalid command state returns specific fault
barrier translation is validation-clean
```

## M9 — Root-pointer compute pipeline

### Goal

Prove the core shader ABI.

### Deliverables

```text
vk/shader.c3
vk/pipeline_compute.c3
samples/root_pointer_compute/shaders/root_pointer.comp.glsl
test/test_vk_root_pointer_compute.c3
samples/root_pointer_compute/
```

### Tasks

```text
load SPIR-V
create compute pipeline
create push constant layout for root pointer
allocate RootArgs from frame arena
shader reads/writes buffers through GpuAddress
read back output
```

### Acceptance criteria

```text
compute shader uses one root pointer
no public descriptor-set binding is required
readback matches expected output
validation output is clean
```

## M10 — Descriptor heap and bindless texture indices

### Goal

Expose TextureIndex and SamplerIndex to shaders.

### Deliverables

```text
descriptor_heap.c3
vk/descriptor_heap.c3
include/shaders/descriptor_heap.glsl
test/test_vk_texture_heap.c3
samples/bindless_texture_compute/
```

### Tasks

```text
implement descriptor buffer path if available
implement descriptor indexing fallback
allocate/free texture descriptor slots
allocate/free sampler descriptor slots
write sampled/storage texture descriptors
make heaps visible to shaders
```

### Acceptance criteria

```text
shader samples texture by TextureIndex
descriptor slots are reused safely
descriptor heap full returns DESCRIPTOR_HEAP_FULL
public API exposes no descriptor sets
```

## M11 — VMA-backed images and texture uploads

### Goal

Create, upload, transition, and sample textures.

### Deliverables

```text
texture.c3
vk/texture.c3
vk/command.c3 updates
test/test_vk_texture_upload.c3
```

### Tasks

```text
translate TextureDesc to vk::ImageCreateInfo
create images with allocator.try_create_image
create image views
track image layouts
record buffer-to-image copy
record image barriers
sample uploaded texture in compute shader
```

### Acceptance criteria

```text
sampled texture upload works
storage image write/readback works
image destruction uses VMA
layout transitions are explicit and validation-clean
```

## M12 — Offscreen graphics

### Goal

Render without a window or swapchain.

### Deliverables

```text
render_pass.c3
vk/render_pass.c3
vk/pipeline_graphics.c3
samples/offscreen_triangle/shaders/*.glsl
test/test_vk_offscreen_triangle.c3
samples/offscreen_triangle/
```

### Tasks

```text
create color render target texture
create graphics pipeline
implement dynamic rendering begin/end
push vertex/fragment root pointers
draw triangle
copy render target to readback
```

### Acceptance criteria

```text
offscreen clear works
offscreen triangle pixels validate
barrier color write -> transfer read is explicit
headless test is validation-clean
```

## M13 — SDL3 windowed swapchain sample

### Goal

Add WSI and presentation through an SDL3 sample.

### Deliverables

```text
swapchain.c3
vk/swapchain.c3
samples/shared/sample_window_sdl.c3
samples/hello_triangle_sdl/
```

### Tasks

```text
create SDL window in sample code
create platform surface descriptor
create Vulkan surface and swapchain
acquire swapchain image
render/blit to swapchain image
present
handle resize/out-of-date
```

### Acceptance criteria

```text
sample imports sdl and gpu
sample depends on sdl3
gpu public API does not expose sdl::Window
triangle presents
resize recreates swapchain without leaks
```

## M14 — Upload and readback convenience helpers

### Goal

Make resource setup ergonomic without hiding synchronization.

### Deliverables

```text
memory.c3 updates
vk/memory.c3 updates
vk/command.c3 updates
test/test_vk_upload_readback.c3
```

### Tasks

```text
upload_buffer_data
upload_texture_data
readback_buffer_data
readback_texture_data
staging allocator recycling
explicit next-use or returned barrier policy
```

### Acceptance criteria

```text
helpers do not hide next-use barriers
large buffer upload works
texture upload works
readback works
non-coherent flush/invalidate handled correctly
```

## M15 — Pipeline cache and graphics state policy

### Goal

Avoid unnecessary pipeline duplication.

### Deliverables

```text
vk/pipeline_cache.c3
vk/pipeline_graphics.c3 updates
test/test_vk_pipeline_cache.c3
```

### Tasks

```text
hash pipeline descriptors
cache identical pipelines
use dynamic viewport/scissor
separate dynamic state from immutable pipeline state where possible
set debug names
```

### Acceptance criteria

```text
identical descriptors reuse cache entries
different formats produce distinct pipelines
viewport/scissor changes do not create new pipelines
```

## M16 — GPU-driven indirect execution

### Goal

Support compute-generated draws and dispatches.

### Deliverables

```text
command.c3 updates
vk/command.c3 updates
samples/gpu_driven_draw_sdl/shaders/build_draws.comp.glsl
samples/gpu_driven_draw_sdl/
test/test_vk_indirect_draw.c3
```

### Tasks

```text
create indirect argument buffers
compute writes draw args
barrier shader write -> indirect read
cmd_draw_indexed_indirect
cmd_dispatch_indirect
optional draw count buffer
async readback tickets over the readback arena (cmd_readback_buffer/texture -> ReadbackTicket { GpuSpan, SemaphoreValue }; poll/resolve)
```

### Acceptance criteria

```text
compute-generated draw renders correctly
barrier is explicit
draw args can be read back for validation
async readback resolves without blocking the frame that recorded it
```

## M17 — Depth attachments

### Goal

Enable depth-tested rendering.

### Deliverables

```text
texture.c3 updates
render_pass handling in vk/render_pass.c3
vk/pipeline_graphics.c3 and vk/pipeline_cache.c3 updates
test/test_vk_depth_attachment.c3
```

### Tasks

```text
depth formats in create_texture (D32_FLOAT first)
depth_stencil usage flag and aspect handling
depth layouts and barriers (DEPTH_ATTACHMENT, DEPTH_READ)
DepthTargetDesc plumbing in cmd_begin_render_pass
accept non-UNDEFINED depth_format in graphics pipelines
depth format participates in the pipeline cache key
end-to-end verification of dynamic depth state (validation-only since the pipeline cache landed)
```

### Acceptance criteria

```text
depth-tested overlapping triangles render correctly
depth write/test toggles behave per descriptor
depth-state pipeline variants still share one cache entry
zero validation messages
```

## M18 — Threading model

### Goal

Define and implement the library's thread-safety policy.

### Deliverables

```text
docs/threading.md
synchronization in backend state where the policy requires it
test/test_vk_threading.c3
```

### Tasks

```text
explore ownership models (externally-synchronized device vs internal locks vs per-thread contexts)
document guarantees per API family (resource creation, command recording, submit, frame lifecycle)
implement the chosen synchronization for slot tables, pipeline cache, arenas, and queues
concurrent command recording across threads
concurrent resource creation policy
```

### Acceptance criteria

```text
documented thread-safety table covers every public entry point
sanctioned concurrent usage runs clean under validation
unsanctioned usage is documented as such, not undefined by omission
```

## M19 — Debug names, stats, and leak reporting

### Goal

Make backend/resource problems visible.

### Deliverables

```text
debug.c3
vk/debug.c3
vk/allocator.c3 updates
samples/memory_report/
```

### Tasks

```text
set Vulkan object names
set VMA allocation names
track live slots
report leaks on destroy_device
expose memory stats
expose VMA stats string
add command labels where useful
```

### Acceptance criteria

```text
leaked buffer reports name and handle
leaked texture reports name and allocation size
memory stats include VMA budget
clean samples report zero leaks
```

## M20 — Shader ABI generator

### Goal

Remove manual CPU/shader layout drift.

### Deliverables

```text
tools/gen_shader_abi/
shader_abi.c3
include/shaders/generated/shader_abi.glsl
test/test_shader_abi_layout.c3
```

### Tasks

```text
define schema format
generate C3 structs
generate shader structs
generate constants
generate sizeof/offset checks
migrate root/material/draw structs to generated path
```

### Acceptance criteria

```text
generated C3 and GLSL agree on sizes and offsets
manual ABI structs are removed or minimized
shader tests use generated structs
```

## M21 — Cross-platform packaging

### Goal

Make the library consumable on supported platforms.

### Deliverables

```text
docs/platforms_and_dependencies.md updates
CI scripts
windows-x64 setup docs
VMA static library artifact policy
SDL3 setup docs (gpu.c3l-samples repository)
```

### Tasks

```text
document linux-x64 setup
document windows-x64 setup
verify dependency paths
automate pure CPU tests
automate headless Vulkan smoke tests where available
```

### Acceptance criteria

```text
fresh linux-x64 checkout builds from docs
windows-x64 build path is documented
the gpu.c3l-samples repository documents SDL3 native library requirements
```

## M22 — First release hardening

### Goal

Prepare a stable first release.

### Deliverables

```text
README.md final pass
CHANGELOG.md
version tag
release docs
sample verification logs
```

### Tasks

```text
audit public API names
audit ownership docstrings
audit fault specificity
audit backend type leakage
audit docs against source
run test matrix
run sample matrix
run validation-enabled samples
```

### Acceptance criteria

```text
root-pointer compute works
bindless texture compute works
offscreen graphics works
SDL3 triangle sample works
GPU-driven draw sample works
all release-gate tests pass
no public vk::/vma::/sdl:: leakage
no known validation errors in release gates
```
