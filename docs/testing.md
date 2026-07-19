# gpu.c3l Testing and Verification

## 1. Test layers

`gpu.c3l` uses three test layers:

```text
pure CPU tests
headless Vulkan tests
SDL3 windowed samples (gpu.c3l-samples repository)
```

Pure CPU tests require no Vulkan loader, VMA static library, SDL3, or window system. Headless Vulkan tests require a Vulkan ICD but no window. SDL3 windowed tests and samples require SDL3 and platform WSI support.

The test matrix covers simultaneous devices, synchronized registry mutation,
closing-slot overlap, generation retirement, and stale-owner rejection. Native
command coverage destroys one device before using another. Multiple discovery
runtimes may coexist.

## 2. Pure CPU tests

Project and sources:

```text
test/cpu/project.json
test/cpu/import_gpu.c3
test/src/test_*.c3
```

Examples:

```text
test_runtime.c3
test_handles.c3
test_ranges.c3
test_texture_support.c3
test_memory_policy.c3
test_allocation_contract.c3
test_shader_abi_layout.c3
test_descriptor_heap_slots.c3
test_diagnostics.c3
test_barrier_desc_validation.c3
```

Coverage:

```text
importing gpu performs no runtime initialization
runtime and adapter token packing and stale-owner rejection
multiple runtimes, stable enumeration, no-adapter discovery, and publication rollback
handle pack/unpack
generation mismatch
invalid handle values
range alignment
GpuSpan checked/unchecked identity-preserving offset math
allocation descriptor validation and alignment normalization
allocation ownership, token consumption, failed-free preservation, and child lifetime
full/nested span mapping and address queries, unavailable mapping, and stale/cross-device rejection
device registry concurrency, generation exhaustion, closing-state rejection,
and active-operation destruction retry
immediate-parent exact-fit, nested, zero-size, out-of-parent, and offset-overflow slicing
texture descriptor normalization and backend-profile validation
arena bump allocation logic without Vulkan
FrameToken size, copy/consumption, scoped-worker fault propagation, end-fault
precedence, retry-token retention, and begin-fault short circuit
MemoryKind policy table completeness
Format translation table completeness through pure tables if separated
BarrierDesc validation
shader ABI sizeof/offset checks
DescriptorHeap free-list reuse
null-safe, exactly-once structured debug dispatch and userdata preservation
invalid-backend callback delivery with callback-enabled/disabled fault parity
borrowed field and explicit absent-fault representation
synthetic frame-arena graphics/compute sharing plans and exact buffer create-info mode/indices
```

Pure CPU tests should be exhaustive where practical.

## 3. Headless Vulkan tests

Headless tests validate backend behavior without SDL3 or swapchains.

Examples:

```text
test_vk_runtime.c3
test_vk_device_request.c3
test_vk_bootstrap.c3
test_vk_vma_allocator.c3
test_vk_allocation.c3
test_vk_span_resolver.c3
test_vk_buffer.c3
test_vk_frame_arena.c3
test_vk_persistent_arena.c3
test_vk_command_submit.c3
test_vk_texture.c3 (adapter capability query/create consistency)
test_vk_root_pointer_compute.c3
test_vk_texture_heap.c3
test_vk_texture_upload.c3
test_vk_offscreen_triangle.c3
```

Coverage:

```text
two independent Vulkan runtimes and borrowed-adapter invalidation
exact-adapter request creation with runtime-instance reuse and retention
surface-aware queue selection and presentation-request gating
create/destroy Vulkan device
create/destroy VMA allocator
query memory budget and stats
create addressable VMA-backed allocation storage
retrieve non-zero GPU address
map/flush/invalidate paths
independent allocation identity, capacity, generation reuse, and owner-domain checks
atomic allocation rollback and CPU_WRITE/GPU_PRIVATE/CPU_READ policy
mapped-span owner, liveness, mapping, coherence, atom-range, and backend-fault checks
native allocation info, mapping/address queries, mapped visibility round trip, and immediate free
allocation-span stale/foreign/range rejection, stats, leak identity, and device-loss teardown
dedicated texture validation, transaction-step rollback, publication, and independent ownership
frame-token generation, stale-alias rejection, allocation, end retry, and reset safety
scoped helper with one observed end attempt after successful and faulting named workers
owner-derived command finalization
format feature queries agree with adapter-backed texture creation
persistent arena suballocation/free
command list begin/end/submit
completion-point signaling and waits
copy upload -> readback
root-pointer compute shader
TextureIndex sampling in compute
offscreen render target clear/draw/readback
dynamic viewport/scissor validation, clipping pixels, pass reset, and pipeline-alias persistence
```

Frame-arena queue-family regressions always pin aliased and distinct creation
plans with synthetic topology tests. Validation-enabled same-range and retired-
slot reuse cases run only when the selected graphics and compute family indices
differ; single-family adapters report the cases as not applicable and do not
count them as cross-family coverage.

Persistent-arena queue-family regressions deterministically pin aliased, two-
family, and three-family creation plans, including exact ordered family metadata
and the exclusive no-list path. Validation-enabled tests use one live `Device`
to consume one persistent span across compute and graphics, then retire, free,
and reallocate the same buffer, offset, and GPU address before cross-family
reuse. Distinct-family runtime cases report PASS when exercised; same-family
hardware reports N/A and is not counted as cross-family evidence. This case
isolates queue-family behavior within one device.

The C3 test harness captures passing-test output and has no skipped-test result,
so the required topology audit enables output explicitly:

```text
c3c test vk_indirect --path test --test-filter frame_span_ --test-show-output
```

A `[PASS]` for a `when_available` test on a same-family adapter proves only that
the topology guard completed. The accompanying `not applicable` message must be
recorded separately; only a run without that message is cross-family evidence.

## 4. SDL3 windowed tests and samples

SDL3 belongs to sample/test harnesses. The binding package dependency is `sdl3`; the import module is `sdl`.

Windowed samples live in the `gpu.c3l-samples` repository (this library repo
carries no SDL3 dependency). Ten windowed samples ship today —
`hello_triangle_sdl`, `textured_cube`, `shadow_mapping`, `gpu_driven_draw_sdl`
among them; see the samples repository README for the full list.

Coverage:

```text
SDL init/shutdown
window creation/destruction
typed Win32, Wayland, and X11 surface-module imports
runtime-owned surface creation and destruction
adapter/surface presentation support and request composition
swapchain creation
runtime info: selected format/mode, clamped extent, actual image count
image acquire/present
resize and out-of-date recovery
coherent info refresh and dormant sentinel after zero/failed resize
UNDEFINED/PRESENT acquired-image prior layout without a seen table
present mode selection
frame pacing sanity
```

The local `vk_swapchain` target covers result mapping, readiness identity and
replay guards, acquire-semaphore retirement, present OOM retry, exact render
completion waits, diagnostics, and dormant publication. `vk_queue` covers
native submit rollback, full-submission bridge scopes, and all selected graphics
families. Surface loss and acquire starvation remain manual recovery cases in
the windowed samples.

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

Validation warnings should be triaged. Some warnings may be documented as
driver/ICD quirks, but release gates should prefer clean output.

Structured-debug tests verify Vulkan severity/category mapping, validation ID
name/number, backend text, named and null objects, `VK_FALSE`, and stderr
fallback without deliberately introducing validation errors into clean targets.
Public-contract diagnostics cover resource creation, recording, submission,
memory, descriptors, pipelines, queue progress, and WSI failures. They assert
exact operation and field context, unchanged faults, exactly-once delivery, and
no public resource identity before handle resolution. Backend-result coverage
fabricates an unmapped `wait_completion` result and verifies callback and
stderr-fallback parity.

Frame/persistent diagnostic tests inject retirement-query and end-signal
backend failures, exercise real virtual-arena exhaustion, and cover double
begin, boundary quiescence, alignment-before-size precedence, stale tokens,
wrong backing buffers, and double free. They assert unchanged faults,
exactly-once delivery, raw backend results, absent public identity, and
mutation-free retry state. Pure range and direct reset helpers remain
fault-only when no public operation context is supplied.

Descriptor/cache completeness coverage adds batch rollback after cached-view
creation, stale and exhausted descriptor identities, immediate slot reuse,
image-view backend results, descriptor bootstrap failures, pipeline-cache
INCOMPLETE, a compatible-header corrupt-blob integration, and a deterministic
cache-create seam whose retryable first attempt and successful empty second
attempt emit no terminal diagnostic. Pure range and lookup helpers remain
fault-only; operation-aware result helpers own specialized backend mapping and
emit exactly once with stable operation context.
Queue tests cover compact completion packing, monotonicity, exhaustion, stale and foreign ownership, unpublished values, native poll/wait, timeout retry, and no public child allocation. Submission coverage includes deterministic empty-work targeting, contiguous publication, same-queue elision, distinct transfer/compute/graphics waits, foreign and later-sequence rejection, timeline-distance backpressure, sequence exhaustion, native failure rollback, device-loss discard, token consumption, and destruction readiness.

Leak tests verify structured `resource_lifetime` delivery, including
`GpuAllocation` identity/name metadata, stderr fallback without a callback,
and callback-active reporting when `enable_validation = false`.
Concurrency coverage uses a synchronized application-owned sink; callbacks are
not assumed serialized or reentrant. Device-registry coverage includes
simultaneous publication, destruction, closing overlap, and shared-runtime
retain counts.

## 7. Test naming

Test functions:

```text
test_handle_pack_round_trip
test_generation_mismatch_rejected
test_frame_arena_alignment
test_vk_create_device
test_vk_create_addressable_allocation
test_vk_root_pointer_compute
```

Test names describe behavior, not roadmap or ticket labels.

## 8. Required coverage

| Area | Required tests |
|---|---|
| Handles | pack/unpack, invalid handle, generation mismatch. |
| Memory policy | memory kind/class translation, alignment, range validation. |
| Independent allocations | descriptor normalization; identity/capacity/reuse; all memory classes; info, mapping, address, checked subspans; rollback; immediate free; placement/lifetime rejection; leaks/stats. |
| Vulkan bootstrap | create/destroy device, required feature checks. |
| VMA allocator | allocator create/destroy, heap budget query, stats string. |
| Private allocation backing | mapped, GPU-private, and addressable native paths. |
| Frame arena | allocation, alignment, overflow, reset safety. |
| Persistent arena | allocate/free/reuse, virtual allocator stats. |
| Queue access | invalid domains stop before backend work; commands enforce semantic roles before mutation; spans cannot widen backing access; native sharing stays exact. |
| Commands | begin/end/submit, timeline signal/wait, invalid state, transactional context-pool rollback. |
| Compute | root pointer shader read/write, readback. |
| Texture heap | descriptor allocation, sampling by TextureIndex. |
| Graphics | offscreen clear/draw/readback; dynamic viewport/scissor state, validation, clipping, pass reset, and pipeline-alias persistence. |
| Swapchain | Runtime-info selection, dormant sentinel, acquired prior layout; pure WSI result mapping; SDL windowed present, resize, and surface-loss recovery. |
| Pipeline cache | cache create/reuse, blob save/load, warm start. |
| Threading | automatic per-worker recording pools, parallel record, identical submit. |
| Debug report | callback dispatch/translation, unchanged faults, leak report contents, debug names, command labels. |
| Depth | depth attachment creation, depth-tested draw, readback. |
| Indirect draw | compute-written draw args, indirect draw, readback. |

Descriptor-buffer topology has deterministic create-info coverage for the
same-family exclusive path and the distinct graphics/compute concurrent path,
including ordered family values and transfer exclusion. The gated
`GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1` regression uses one live `Device`, a
texture admitting graphics and compute, and one descriptor across compute
storage writes and fragment sampling with a timeline dependency. It reports the selected family
indices; the gate remains necessary on drivers with unreliable descriptor-buffer image access.

## 9. Build commands

The shipped library is a `manifest.json` package (module `gpu`); it has no
project of its own. The test harness (`test/project.json`) is whitebox: it
lists the library sources directly (mirroring `manifest.json`) and declares
`vk`, `vma`, and `spvreflect` as dependencies, resolved via
`"dependency-search-paths": ["../lib"]` — vendored bindings by real directory
name, no symlink directory. Consumer-style resolution of `gpu` is exercised by
the `gpu.c3l-samples` repository.

CPU targets:

```sh
c3c run import_gpu --path test/cpu
c3c build import_surface_win32 --path test/cpu
c3c build import_surface_wayland --path test/cpu
c3c build import_surface_x11 --path test/cpu
c3c build span_data_operations --path test/cpu
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
python -m unittest scripts.test_check_public_api scripts.test_check_backend_dispatch
python scripts/check_public_api.py
python scripts/check_backend_dispatch.py
python scripts/check_retired_api.py
```

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

The blocking headless matrix is shared by Linux and Windows:

```text
vk_bootstrap vk_allocation vk_command vk_texture vk_descriptor_heap vk_root_pointer
vk_texture_heap vk_shader_reflection vk_offscreen vk_swapchain
vk_pipeline_cache vk_indirect vk_indexed_draw vk_depth vk_threading
vk_queue vk_debug vk_device_request
```

The workflow stores this list once as `HEADLESS_TEST_TARGETS` and both jobs iterate it.
Distinct-adapter ownership is gated deterministically by the CPU stub suite.
`vk_device_request` also uses two physical adapters when both support the strict
profile and reports `distinct-adapter=N/A` otherwise.

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

C3 0.8.0 constraints:

- Library `manifest.json` does **not** accept `dependency-search-paths` (that is
  a `project.json` key); dependencies are declared per-target and resolved by
  the consumer's search path.
- `manifest.json` `sources` must list files explicitly — all 19 public source
  files under `gpu/` plus `gpu/vk/**`; a glob like `*.c3` is rejected and the default does not
  recurse into `gpu/vk/`.

## 10. CI matrix

CI is shipped: `.github/workflows/ci.yml`, one workflow, three jobs.

```text
linux (blocking): generator tests, ABI drift gate, shader build, full
    lavapipe test sweep, then a c3c docgen API reference uploaded as the
    api-reference artifact
windows (blocking): same suite via mesa-dist-win lavapipe, registered in the
    HKLM Vulkan driver registry; descriptor-buffer device/heap coverage reports
    its status, while shader E2E awaits a fixed driver or real hardware
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
| Smoke linkage + full lavapipe test sweep + api-reference artifact | linux | yes |
| Getting-started walkthrough (docs-walkthrough job, `scripts/run_doc.py`) | linux | yes |
| Link proof (smoke) + pure-CPU targets | windows | yes |
| lavapipe (mesa-dist-win) Vulkan sweep | windows | yes |
| Descriptor-buffer device/heap | windows | yes when exposed; otherwise reported not exercised |
| Descriptor-buffer shader E2E | real hardware | pending; software ICD is reported not exercised |

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
read the output allocation span
compare exact values
```

Graphics:

```text
clear to known color
render primitive
copy the render target to readback storage
sample a small set of pixels
compare with tolerance for floating formats
```

## 13. Leak verification

Every backend test should end with:

```text
wait every submitted completion point
destroy all public resources
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

The callback form carries the applicable subset as a borrowed
`resource_lifetime` message; the stderr form remains the no-callback fallback.

## 14. Failure path tests

Tests should cover specific faults:

```text
invalid handle -> INVALID_HANDLE
arena exact-end fit -> success
arena one-byte, alignment, or extent overflow -> ARENA_FULL
zero allocation size, malformed alignment, or unavailable span capability -> INVALID_ARGUMENT
stale or foreign allocation/span identity -> INVALID_HANDLE
live placement on free_allocation -> RESOURCE_IN_USE and unchanged token
allocation table capacity -> SLOT_TABLE_FULL with no published token
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
SDL3 triangle sample queries the selected format/count, presents, and resizes without fixed image-state tables
GPU-driven indirect draw sample works
memory stats report plausible budgets
leak reports are clean after all samples
public API docs match signatures
no public API signature exposes vk::, vma::, or sdl:: types
```
