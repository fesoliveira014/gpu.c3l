# gpu.c3l Testing and Verification

## 1. Test layers

`gpu.c3l` uses three test layers:

```text
pure CPU tests
headless Vulkan tests
SDL3 windowed samples (gpu.c3l-samples repository)
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
GpuSpan checked/unchecked offset math
immediate-parent exact-fit, nested, zero-size, and out-of-parent slicing
GPU-address, CPU-pointer, backing-offset, and requested-size overflow rejection
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
test_vk_buffer.c3
test_vk_frame_arena.c3
test_vk_persistent_arena.c3
test_vk_command_submit.c3
test_vk_root_pointer_compute.c3
test_vk_texture_heap.c3
test_vk_texture_upload.c3
test_vk_offscreen_triangle.c3
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

Windowed samples live in the `gpu.c3l-samples` repository (this library repo
carries no SDL3 dependency). Ten windowed samples ship today —
`hello_triangle_sdl`, `textured_cube`, `shadow_mapping`, `gpu_driven_draw_sdl`
among them; see that repository and `docs/samples.md` for the full list.

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

The local `vk_swapchain` target pure-tests result translation for transient
acquire, out-of-date, surface-lost, suboptimal, and delegated device-loss
outcomes. Real surface loss and acquire starvation are not portable to force;
exercise their caller recovery manually in the windowed sample repository.

Windowed tests may be manual at first. Automated windowed tests can be added only when CI/window-system support is stable.

## 5. Sample project dependencies

Core library manifest dependencies:

```text
vk
vma
spvreflect
```

The test harness compiles the library sources directly (whitebox — see
`test/project.json`); its dependencies are `vk`, `vma`, `spvreflect` only.
Samples in `gpu.c3l-samples` depend on `gpu` (vendored submodule) plus `sdl3`
and import `gpu` / `sdl`.

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
| Commands | begin/end/submit, timeline signal/wait, invalid state, transactional context-pool rollback. |
| Compute | root pointer shader read/write, readback. |
| Texture heap | descriptor allocation, sampling by TextureIndex. |
| Graphics | offscreen clear/draw/readback. |
| Swapchain | pure WSI result mapping; SDL windowed present, resize, and surface-loss recovery. |
| Pipeline cache | cache create/reuse, blob save/load, warm start. |
| Threading | per-thread recording contexts, parallel record, identical submit. |
| Debug report | leak report contents, debug names, command labels. |
| Depth | depth attachment creation, depth-tested draw, readback. |
| Indirect draw | compute-written draw args, indirect draw, readback. |

## 9. Build commands

The shipped library is a `manifest.json` package (module `gpu`); it has no
project of its own. The test harness (`test/project.json`) is whitebox: it
lists the library sources directly (mirroring `manifest.json`) and declares
`vk`, `vma`, and `spvreflect` as dependencies, resolved via
`"dependency-search-paths": ["../lib"]` — vendored bindings by real directory
name, no symlink directory. Consumer-style resolution of `gpu` is exercised by
the `gpu.c3l-samples` repository.

Smoke target, as CI runs it:

```sh
# from the repo root; --path runs as if standing in test/
c3c build smoke --path test
./test/build/smoke
```

Prerequisites on `linux-x64`: a Vulkan loader (`libvulkan.so.1`) on the system
and the `libVulkanMemoryAllocator.a` static lib shipped under
`lib/vma.c3l/linked-libs/linux-x64/`. After cloning, init submodules:

```sh
git submodule update --init --recursive
```

Prerequisites on `windows-x64`:

- MSVC (VS Build Tools) with the vcvars64 environment loaded — `cl`/`lib` for
  the VMA build, and the CRT link paths for `c3c`. If `c3c` reports "Failed to
  find the C runtime at link time" outside a developer shell, run
  `c3c fetch-sdk windows` once.
- A Vulkan SDK for `glslc` (the `vulkan-1.lib` import library ships with the
  vendored `vk.c3l`; a system Vulkan loader/driver is needed to run).
- The VMA static lib is not vendored for Windows — build it once into
  `lib/vma.c3l/linked-libs/windows-x64/` with the pinned headers (SDK-bundled
  VMA headers can predate the binding; the script's size probe gates this):

```sh
# from a git-bash shell with the vcvars64 environment
git clone --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers.git /tmp/vh
git clone --depth 1 --branch v3.3.0 https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git /tmp/vma
mkdir -p /tmp/vma-inc/vma && cp /tmp/vma/include/vk_mem_alloc.h /tmp/vma-inc/vma/
VULKAN_HEADERS=$(cygpath -w /tmp/vh) VMA_INCLUDE=$(cygpath -w /tmp/vma-inc) \
    sh lib/vma.c3l/scripts/build-vma-windows.sh
```

Windows-specific notes:

- `test/project.json` pins `"wincrt": "dynamic"`: `VulkanMemoryAllocator.lib`
  is compiled `/MD`, and linking it against c3c's dynamic-debug default mixes
  release and debug STL DLLs, which crashes in VMA's `std::mutex` when the two
  runtimes drift apart. Do not override it back to a debug CRT.
- Build the test SPIR-V once before any `vk_*` target:
  `python scripts/build_shaders.py` (`.spv` files are not committed).
- On machines with overlay layers registered (Steam, RTSS, Epic, RenderDoc),
  run tests and benches with `VK_LOADER_LAYERS_DISABLE=~implicit~` for
  deterministic results.

Notes pinned during scaffolding (C3 0.8.0):

- Library `manifest.json` does **not** accept `dependency-search-paths` (that is
  a `project.json` key); dependencies are declared per-target and resolved by
  the consumer's search path.
- `manifest.json` `sources` must list files explicitly — all 17 root source
  files plus `vk/**`; a glob like `*.c3` is rejected and the default does not
  recurse into `vk/`.

## 10. CI matrix

CI is shipped: `.github/workflows/ci.yml`, one workflow, three jobs.

```text
linux (blocking): generator tests, ABI drift gate, shader build, full
    lavapipe test sweep, then a c3c docgen API reference uploaded as the
    api-reference artifact
windows (blocking except the advisory sweep): same suite via mesa-dist-win
    lavapipe, registered in the HKLM Vulkan driver registry
docs-walkthrough (blocking): executes docs/getting_started.md verbatim on a
    bare runner
```

Windowed SDL3 samples run in the `gpu.c3l-samples` repository CI under
xvfb/lavapipe.

## 11. Test data and shaders

Tests are consumers of the library, so test shaders are test-owned and live with the tests, not in the shipped library:

```text
test/shaders/compute/
test/shaders/graphics/
```

They `#include` the library's published shader-side ABI includes from `include/shaders/`.

Pure-CPU tests (handles, ranges, ABI layout) live in the `test/cpu` project:
the full public module plus a stub backend, linking no native libraries — a
clean checkout runs them with no Vulkan/VMA installed, and CI runs them
before any native setup.

Shared CPU/shader structs come from `.abi` schemas (see `docs/shader_abi.md` §12). Generated outputs are committed; `scripts/gen_abi.py --check` is the drift gate — run it as part of any full test sweep, and rerun `scripts/gen_abi.py` (then `scripts/build_shaders.py`) after editing a schema.

CI tiers (`.github/workflows/ci.yml`):

| Tier | Platform | Blocking |
|---|---|---|
| Generator tests, drift gate, shader build | linux + windows | yes |
| Full lavapipe test sweep + api-reference docgen artifact | linux | yes |
| Getting-started walkthrough (docs-walkthrough job, `scripts/run_doc.py`) | linux | yes |
| Link proof (smoke) + pure-CPU targets | windows | yes |
| lavapipe (mesa-dist-win) Vulkan sweep | windows | no — advisory |

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
arena exact-end fit -> success
arena one-byte, alignment, or extent overflow -> ARENA_FULL
zero allocation size or malformed alignment -> INVALID_ARGUMENT
unsupported required feature -> UNSUPPORTED_FEATURE
invalid command state -> COMMAND_RECORDING_ERROR
invalid descriptor index -> INVALID_HANDLE or DESCRIPTOR_HEAP_FULL as appropriate
```

Pure Vulkan result-taxonomy tests cover generic fallback behavior,
operation-specific mappings, fatal pipeline/cache fault preservation, partial
enumeration, and bounded-wait timeout behavior without requiring a GPU.

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
