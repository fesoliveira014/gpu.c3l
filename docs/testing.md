# gpu.c3l Testing and Verification

## 1. Test layers

`gpu.c3l` uses three test layers:

```text
pure CPU tests
headless Vulkan tests
SDL3 windowed samples/tests
```

Pure CPU tests must run without Vulkan, VMA static library loading beyond compile/link, SDL3, or a window system. Headless Vulkan tests require a Vulkan ICD but no window. SDL3 windowed tests/samples require SDL3 and platform WSI support.

## 2. Pure CPU tests

Location:

```text
test/*.c3
```

Examples:

```text
test_handles.c3
test_ranges.c3
test_memory_policy.c3
test_shader_abi_layout.c3
test_descriptor_heap_slots.c3
test_barrier_desc_validation.c3
```

Coverage:

```text
handle pack/unpack
generation mismatch
invalid handle values
range alignment
GpuSpan offset math
arena bump allocation logic without Vulkan
MemoryKind policy table completeness
Format translation table completeness through pure tables if separated
BarrierDesc validation
shader ABI sizeof/offset checks
DescriptorHeap free-list reuse
```

Pure CPU tests should be exhaustive where practical.

## 3. Headless Vulkan tests

Headless tests validate backend behavior without SDL3 or swapchains.

Examples:

```text
test_vk_bootstrap.c3
test_vk_vma_allocator.c3
test_vk_buffer_address.c3
test_vk_frame_arena.c3
test_vk_persistent_arena.c3
test_vk_command_submit.c3
test_vk_compute.c3
test_vk_texture_heap.c3
test_vk_texture_upload.c3
test_vk_offscreen_graphics.c3
```

Coverage:

```text
create/destroy Vulkan device
create/destroy VMA allocator
query memory budget and stats
create addressable VMA-backed buffers
retrieve non-zero GPU address
map/flush/invalidate paths
frame arena allocation and reset safety
persistent arena suballocation/free
command list begin/end/submit
timeline semaphore signaling
copy upload -> readback
root-pointer compute shader
TextureIndex sampling in compute
offscreen render target clear/draw/readback
```

## 4. SDL3 windowed tests and samples

SDL3 belongs to sample/test harnesses. The binding package dependency is `sdl3`; the import module is `sdl`.

Windowed samples:

```text
samples/hello_triangle_sdl
samples/gpu_driven_draw_sdl
samples/texture_viewer_sdl
samples/swapchain_resize_sdl
```

Coverage:

```text
SDL init/shutdown
window creation/destruction
Vulkan surface creation path
swapchain creation
image acquire/present
resize and out-of-date recovery
present mode selection
frame pacing sanity
```

Windowed tests may be manual at first. Automated windowed tests can be added only when CI/window-system support is stable.

## 5. Sample project dependencies

Core library manifest dependencies:

```text
vk
vma
```

Sample/test project dependencies:

```text
gpu
sdl3
```

Sample source imports:

```c3
import gpu;
import sdl;
```

Do not make SDL3 a required dependency of the shipped library unless a public helper module explicitly becomes part of the library.

## 6. Validation policy

All Vulkan tests should support validation-enabled runs.

Validation requirements:

```text
zero Vulkan validation errors for release-gate tests
zero leaked Vulkan objects at device destruction
zero leaked VMA allocations at device destruction
zero leaked public resource handles at device destruction
```

Validation warnings should be triaged. Some warnings may be documented as driver/ICD quirks, but release gates should prefer clean output.

## 7. Test naming

Test functions:

```text
test_handle_pack_round_trip
test_generation_mismatch_rejected
test_frame_arena_alignment
test_vk_create_device
test_vk_create_addressable_buffer
test_vk_root_pointer_compute
```

Do not include milestone labels in test names.

## 8. Required test categories by milestone

| Milestone area | Required tests |
|---|---|
| Handles | pack/unpack, invalid handle, generation mismatch. |
| Memory policy | memory kind translation, alignment, range validation. |
| Vulkan bootstrap | create/destroy device, required feature checks. |
| VMA allocator | allocator create/destroy, heap budget query, stats string. |
| Buffers | mapped buffer, device buffer, addressable buffer. |
| Frame arena | allocation, alignment, overflow, reset safety. |
| Persistent arena | allocate/free/reuse, virtual allocator stats. |
| Commands | begin/end/submit, timeline signal/wait, invalid state. |
| Compute | root pointer shader read/write, readback. |
| Texture heap | descriptor allocation, sampling by TextureIndex. |
| Graphics | offscreen clear/draw/readback. |
| Swapchain | SDL windowed present and resize. |

## 9. Build commands

Recommended developer commands:

```sh
c3c build
c3c test
```

If separate project files exist:

```sh
cd test && c3c test
cd samples && c3c build
```

Exact commands should be pinned after `project.json` structure is finalized for C3 0.8.0.

## 10. CI matrix

Minimum CI:

```text
linux-x64 pure CPU tests
linux-x64 headless Vulkan tests when a Vulkan software ICD is available
format/style static checks implemented as scripts, not c3fmt
```

Second-stage CI:

```text
windows-x64 build
windows-x64 pure CPU tests
windows-x64 headless Vulkan smoke test where environment permits
```

Windowed SDL3 tests are manual unless CI has a reliable display server.

## 11. Test data and shaders

Test shaders should live in:

```text
resources/shaders/compute/
resources/shaders/graphics/
```

Generated SPIR-V should either:

```text
be built as part of the test/sample build step
or be committed only if shader toolchain availability is a problem
```

The shader build policy must be documented in `docs/platforms_and_dependencies.md`.

## 12. Readback verification

Compute and graphics tests should prefer deterministic readback.

Compute:

```text
write known input
run shader
read output buffer
compare exact values
```

Graphics:

```text
clear to known color
render primitive
copy render target to readback buffer
sample a small set of pixels
compare with tolerance for floating formats
```

## 13. Leak verification

Every backend test should end with:

```text
destroy all public resources
drain deferred destruction
wait device idle if needed
destroy device
assert no live resources
```

Debug leak output should list:

```text
handle type
slot index
generation
debug name
allocation size
creation frame if tracked
```

## 14. Failure path tests

Tests should cover specific faults:

```text
invalid handle -> INVALID_HANDLE
arena overflow -> ARENA_FULL
unsupported required feature -> UNSUPPORTED_FEATURE
invalid command state -> COMMAND_RECORDING_ERROR
invalid descriptor index -> INVALID_HANDLE or DESCRIPTOR_HEAP_FULL as appropriate
```

Tests should assert specific faults, not merely that some fault occurred.

## 15. Release verification checklist

Before first release:

```text
pure CPU tests pass
headless Vulkan tests pass validation-clean
root-pointer compute sample works
bindless texture compute sample works
offscreen graphics sample readback matches expected output
SDL3 triangle sample presents and resizes
GPU-driven indirect draw sample works
memory stats report plausible budgets
leak reports are clean after all samples
public API docs match signatures
no public API signature exposes vk::, vma::, or sdl:: types
```
