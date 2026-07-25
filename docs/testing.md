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

The library source inventory is intentional: `gpu/gpu.c3i` supplies public
non-callables, `gpu/gpu.c3` supplies public callables, each platform surface is
a local `.c3i`/`.c3` pair, `gpu/internal/*.c3` is backend-independent private
code, and `gpu/internal/vk/*.c3` is the private Vulkan backend. CPU projects
list private files explicitly so their `gpu::internal::vk` stub never compiles
the real backend. The stub implements the same typed private Vulkan functions
at compile time; it is not a runtime polymorphism layer.

Examples:

```text
test_runtime.c3
test_handles.c3
test_ranges.c3
test_texture_support.c3
test_resource_access.c3
test_allocation_contract.c3
test_shader_abi_layout.c3
test_texture_transition.c3
test_diagnostics.c3
test_workload_limits.c3
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
texture-view normalization and backend-profile validation
allocation range, alignment, access, and class validation without Vulkan
caller-owned allocation and completion lifetime contracts
MemoryClass policy table completeness
Format translation table completeness through pure tables if separated
global barrier stage/hazard/queue validation and exact Vulkan scope mapping
single-profile texture-layout mapping completeness and exact Vulkan lowering
shader ABI scalar widths plus generated-work record sizes, offsets, and strides
TextureView owner/generation validation and descriptor-heap free-list reuse
null-safe, exactly-once structured debug dispatch and userdata preservation
runtime-bootstrap callback delivery with callback-enabled/disabled fault parity
borrowed field and explicit absent-fault representation
synthetic allocation sharing plans and exact buffer create-info mode/indices
sampler exact anisotropy boundaries, inactive-field canonicalization and byte
hashing, forced collision equality, stable-index deduplication, pre-dispatch
transactionality, table exhaustion, and concurrent convergence
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
required `VK_EXT_extended_dynamic_state3` absence and
`dynamicPrimitiveTopologyUnrestricted` property rejection
required EDS3 color blend-enable, blend-equation, and write-mask feature and
dispatch rejection
generated-work extension, feature, stage, layout-limit, and public-cap coherence
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
caller-owned CPU_WRITE roots retained through covering completion
explicit wait-before-reuse and wait-before-free behavior
owner-derived command finalization
format feature queries agree with adapter-backed texture creation
command list begin/end/submit
completion-point signaling, retired-prefix publication, cached polls/waits,
and deterministic publication/concurrent-retirement races
exact ordinary/generated completion-wait stage lowering and distinct-family
compute-producer to graphics generated-work execution when topology permits
global semantic barrier hazard matrix submission
copy upload -> readback
root-pointer compute shader
TextureView publication and raw TextureIndex sampling in compute
offscreen render target clear/draw/readback
dynamic viewport/scissor device-bound validation, exact native lowering,
validation-clean negative-height and off-pass cases, clipping pixels,
command-buffer packet persistence across pass boundaries, explicit replacement,
partial updates, and pipeline-alias persistence
attachment-view capacity/generation/ownership, creation rollback, texture
retention, in-flight destruction rejection, and allocation-free pass begin
borrowed swapchain-view build/acquire/render guards and resize invalidation
texture-view capacity, batch rollback, immediate index reuse, stale and
cross-device release rejection, and concurrent publication/release
sampler exact-cap acceptance and over-cap rejection, exact native lowering,
hashed-bucket interning, collision-safe stable indices, heap/table/native-create
rollback, index consistency, concurrent convergence, and device-owned teardown
```

Allocation queue-family regressions pin aliased and distinct creation plans
with synthetic topology tests. Live Vulkan coverage verifies that an
allocation's immutable access roles select the exact native sharing families;
barriers and completion waits remain explicit.

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
exact empty UNDEFINED/PRESENT acquired-image prior state
independent PRESENT source and first-consumer readiness scopes
present mode selection
application render-loop pacing sanity
```

The local `vk_swapchain` target covers exact zero/finite/infinite timeout
propagation, timeout and error rollback, same-semaphore retry, invalid native
image indices, readiness identity and replay guards, acquire-semaphore
retirement, present OOM retry, per-image
presentation-fence attachment/reuse, immediate destroy/resize rejection, shared
texture-reference detection, diagnostics, and dormant publication. A busy
acquire ring is proven to skip the native acquisition call. `vk_queue`
covers native submit rollback, full-submission bridge scopes, and all selected
graphics families. Windowed samples remain the manual WSI recovery and pacing
coverage.

Texture synchronization coverage pins the compositional matrix: vertex,
fragment, compute, and combined sampled stages; storage read, write, and
read-write access; transfer and attachment requirements; swapchain-only
presentation; exact partial mip/layer ranges; and color/depth native masks.
Side-aware presentation cases prove transitions to `PRESENT` use destination
`NONE`/`NONE`, transitions from `PRESENT` use no source access and the paired
first-consumer stage, and trusted/full command tables emit identical native
fields. Submit seams separately pin the acquire readiness stage and the
full-submission presentation signal.
The injected native-emission seam also proves one accepted barrier performs one
helper call, handle resolution, recording-access validation, range resolution,
native assembly, and emission, plus exactly two state validations/lowerings.
Post-retain failures must emit nothing, preserve reference counts, and report
the exact nested `TextureBarrier` field. No test assumes backend-tracked layout
history.

Texture synchronization coverage pins the sole native layout mapping across
texture barriers, sampled/storage descriptors, color/resolve/depth attachments,
and both buffer-image copy directions. Tests cover
`VK_IMAGE_LAYOUT_UNDEFINED`, both transfer-optimal layouts, color and
depth/stencil sampled-read layouts, `VK_IMAGE_LAYOUT_GENERAL`, both
attachment-optimal layouts, and `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
Invalid-input rollback must occur before native emission, and no test may
depend on global backend layout history or an automatically inserted repair
transition.

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

Policy tests cover exactly two contract modes through the sole direct-record
command-token representation:

| Contract | Vulkan layers | Detailed semantic work | Reference work | Teardown leak scans |
|---|---:|---:|---:|---:|
| `TRUSTED` | off | zero | zero | off |
| `FULL` | off | nonzero | nonzero | on |

Separate tests exercise both contract modes with Vulkan layers on and off and
with callback and debug-name delivery enabled and disabled. `FULL` must return
the same library fault and structured diagnostic with Vulkan layers off or on.
Callback- or name-only TRUSTED configurations must not request the Khronos
layer or enable command checks, tracking, or teardown leak scans.

The contract assertions use the same four categories as `docs/api.md`:
preconditions, always-checked behavior, `FULL` diagnostics, and runtime
failures. Every mode must preserve the always-checked floor: null/slice/range
and overflow protection needed before host access or safe lowering,
authoritative command phase and internal-table integrity, public device
ownership, lifecycle/cold-path checks, Vulkan result/device-loss handling, and
transactional creation rollback. Trusted-mode cases distinguish that floor
from semantic misuse that is intentionally a caller precondition. Every policy
checks static device-slot liveness, record generation, and command phase before
native mutation. Lifecycle and submit coverage additionally validates
device ownership.

Zero-root execution evidence covers direct compute, direct graphics, indirect
work, and generated draw, indexed-draw, and dispatch records. Each family
asserts unchanged forwarding for zero and nonzero values. The policy matrix
also proves that `TRUSTED` and `FULL` do not classify zero
itself as misuse; shader fixtures branch before any zero dereference.

Run the dedicated matrix on a pinned headless ICD:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  c3c test vk_validation_policy --path test
```

It covers exact table selection, zero tracking storage under TRUSTED, exact
retain/release behavior under FULL, early destruction,
recording/executable/submitted/device-loss cleanup, callback and layer
independence, teardown predicates, mandatory safety, and partial-create
rollback. Tracking-list coverage additionally pins exact owner/index/generation
identity, linear duplicate detection without another mutex/retain, canonical
retained-counter publication before append, direct release, compound-operation
capacity preflight, and suffix-only rollback through the 4,096-reference
ceiling.
Focused behavioral tests invoke every command family through TRUSTED and FULL.
C3 compilation verifies operation-table function existence and types; runtime
faults, ownership/state transitions, allocator observations, and exact
reference/native-emission evidence establish behavior without a source-policy
scanner.

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

Allocation diagnostic tests cover invalid alignment and size, unavailable
mapping/address capabilities, stale spans, visibility failures, and detected
resource references. They assert unchanged faults, exactly-once delivery, raw
backend results, and stable public allocation identity where available. Pure
range helpers remain fault-only when no public operation context is supplied.

Descriptor/cache completeness coverage adds batch rollback after cached-view
creation, stale and exhausted descriptor identities, immediate slot reuse,
image-view backend results, descriptor bootstrap failures, pipeline-cache
INCOMPLETE, a compatible-header corrupt-blob integration, and a deterministic
cache-create seam whose retryable first attempt and successful empty second
attempt emit no terminal diagnostic. Pure range and lookup helpers remain
fault-only; operation-aware result helpers own specialized backend mapping and
emit exactly once with stable operation context.
Texture-view accounting coverage also asserts one generation-checked lookup per
successful destruction, exact single and repeated-owner batch counts,
transactional overflow/underflow diagnostics, and zero through one ownership
work unit for texture destruction at descriptor high-water marks 16, 4,096,
and 65,536. Swapchain tests allow zero through one work unit per wrapped image
examined while retaining texture and attachment tracked-reference guards under
`FULL`.
Queue tests cover compact completion packing, monotonicity, exhaustion, stale
and foreign ownership, unpublished values, native poll/wait, timeout retry,
and no public child allocation. Deterministic seams pause after native
acceptance and batch publication, rendezvous concurrent first polls, and prove
the published-prefix cap, exact wait retirement, timeout preservation, and
100,000 cached polls with zero native queries or retirement-lock acquisitions.
Submission coverage also includes one direct-token validation and duplicate
visit per inspected token, exact record-queue validation, mixed
allocators on one queue, deterministic empty-work targeting, intrusive
queue-owned pending records, contiguous publication, and exact
completion/readiness destination masks. Invalid and queue-unsupported scopes,
wrong queues, duplicates, post-claim preparation faults, and native failures
must publish no pending link or point and leave tokens and readiness retryable.
The suite also covers same-queue wait validation and elision, distinct
transfer/compute/graphics waits, foreign and later-sequence rejection,
timeline-distance backpressure, sequence exhaustion, device-loss discard,
token consumption, and destruction readiness. Publication-gap seams prove an
earlier completed point can retire through the short retirement boundary while
a later same-queue submit remains paused inside the long submission boundary.

Leak tests verify structured `resource_lifetime` delivery under `FULL`,
including `GpuAllocation` identity/name metadata, stderr fallback without a
callback, and callback-active reporting with Vulkan layers disabled.
Concurrency coverage uses a synchronized application-owned sink; callbacks are
not assumed serialized or reentrant. Device-registry coverage includes
simultaneous publication, destruction, closing overlap, and shared-runtime
retain counts.

## 7. Test naming

Test functions:

```text
test_handle_pack_round_trip
test_generation_mismatch_rejected
test_allocation_alignment
test_vk_create_device
test_vk_create_addressable_allocation
test_vk_root_pointer_compute
```

Test names describe behavior, not roadmap or ticket labels.

## 8. Required coverage

| Area | Required tests |
|---|---|
| Handles | pack/unpack, invalid handle, generation mismatch. |
| Memory policy | memory-class translation, alignment, range validation. |
| Independent allocations | descriptor normalization; identity/capacity/reuse; all memory classes; info, mapping, address, checked subspans; rollback; immediate free; placement/lifetime rejection; leaks/stats. |
| Vulkan bootstrap | create/destroy device, required feature checks. |
| VMA allocator | allocator create/destroy, heap budget query, stats string. |
| Private allocation backing | mapped, GPU-private, and addressable native paths. |
| Queue access | invalid domains stop before backend work; commands enforce semantic roles before mutation; spans cannot widen backing access; native sharing stays exact. |
| Command allocators | exact device/queue binding; default, ceiling, and overflow validation; transactional pool/buffer/host rollback; recyclable generations; begin/reference/generated capacity faults; address-stable authoritative records; fixed per-scratch linear reference lists; exact-identity deduplication; direct retained-counter release; suffix-only rollback; no submitted-unit reuse before retirement; non-waiting destroy in recording/executable/in-flight states; device-child accounting; exact-family pools; and TRUSTED zero reference storage. |
| Commands | exact 16-byte x64 direct token size and deterministic stale/wrong-phase/consumed diagnostics, including copied tokens after device-slot reuse; submit-only foreign-device rejection; one authoritative state through begin/end/submit/rollback/discard/retirement; one backend claim with exact linear direct-token visits and duplicate visits; authoritative record-queue validation without allocator reproof; retryable failure; intrusive exact-queue retirement; timeline signal/wait; invalid state; completion-safe exact-allocator command-buffer reset/reuse; explicit generated-scratch capacity faults; FULL regular/generated draw initialization; complete and partial state updates before or during passes; retired queue-based signatures; and zero warm allocation or execution-time pipeline creation. |
| Compute | root pointer shader read/write, readback, active-pipeline kind validation, and exact zero/nonzero root push behavior. |
| Texture heap | owner-bearing view publication/release, raw-index reuse, stale/foreign rejection, and sampling by TextureIndex. |
| Graphics | offscreen clear/draw/readback for opaque, alpha, premultiplied, additive, and masked writes; explicit attachment-view lifecycle and in-flight retention; one-command minimal begin; transactional convenience rejection without native or command-state mutation; exact ten-command viewport/raster/depth prefix and three-command color packets; state reuse across compatible pipeline switches and pass boundaries and reset on command-buffer reuse; optional partial updates; FULL regular/generated missing-initialization rejection; exact zero/nonzero stage roots; per-target blend/write masks; dynamic raster validation; selected-device viewport bounds; exact negative-height, negative-coordinate, reversed-depth, off-pass scissor, and empty-scissor lowering; validation-layer-clean accepted cases; clipping; packet replacement; incompatible color-domain invalidation; and pipeline-alias persistence. |
| Swapchain | Runtime-info selection, dormant sentinel, acquired prior state; pure WSI result mapping; SDL windowed present, resize, and surface-loss recovery. |
| Pipeline cache | cache create/reuse, blob save/load, warm start, raster- and color-state aliasing across five command color packets, ordered color-format separation, transactional batches, and singleton compute/generated-dispatch layouts. |
| Threading | one explicit allocator per concurrent worker, same-allocator bounded-full rejection, synchronized allocator migration and executable handoff, one-thread-at-a-time alias confinement, no device-wide recording lock, no temp-pool setup, historical worker churn, private command-buffer/generated-scratch reuse, parallel record, same-queue submit/present serialization, distinct-queue submission concurrency, and prior-point retirement across a later publication gap. |
| Upload benchmark observations | stable device-type and lavapipe classification; scaling against one worker. |
| Debug report | callback dispatch/translation, unchanged faults, leak report contents, debug names, command labels. |
| Depth | depth attachment creation, depth-tested draw, exact nonzero mip/layer selection, neighboring-subresource isolation, and readback. |
| Indirect draw | compute-written draw args, direct/indirect/generated zero-root execution, indirect draw, and readback. |

Descriptor heap eligibility has pure coverage for every required indexing
feature and sampled-image, storage-image, sampler, per-stage aggregate, and
all-pools update-after-bind limit. Adapter support and device creation consume
the same cached support record. Native-limit failures return
`UNSUPPORTED_FEATURE` before logical-device or heap mutation, while values
above the library ceiling remain `INVALID_ARGUMENT`. No test creates a
temporary logical device or layout to probe heap support.

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
c3c build canonical_strict_surface --path test/cpu
c3c build import_surface_win32 --path test/cpu
c3c build import_surface_wayland --path test/cpu
c3c build import_surface_x11 --path test/cpu
c3c build span_data_operations --path test/cpu
c3c build sampler_operations --path test/cpu
c3c build texture_view_operations --path test/cpu
c3c build command_wrapper_bench --path test/cpu -O1
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
python -m unittest scripts.test_check_docs scripts.test_check_public_api scripts.test_check_backend_dispatch scripts.test_check_retired_api scripts.test_check_swapchain_acquire_policy scripts.test_check_sampler_policy
python scripts/check_docs.py
python scripts/check_public_api.py
python scripts/check_backend_dispatch.py
python scripts/check_swapchain_acquire_policy.py
python scripts/check_sampler_policy.py
python scripts/check_retired_api.py
python -B -m unittest scripts.test_run_benchmarks
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
upload_bench_observation vk_device_request vk_bootstrap vk_allocation
vk_command vk_texture vk_descriptor_heap vk_root_pointer vk_texture_heap
vk_shader_reflection vk_offscreen vk_performance vk_allocator_observation
vk_swapchain vk_pipeline_cache vk_indirect vk_indexed_draw vk_depth
vk_threading vk_queue vk_debug vk_validation_policy
```

The workflow stores these exact 22 targets once as `HEADLESS_TEST_TARGETS` and
both jobs iterate it.
`vk_allocator_observation` installs the test-only recording allocator at the
thread allocator seam and proves command recording reports real acquire/resize
activity without source-graph inference.
`vk_depth` covers validation-clean offscreen color/depth rendering, including a
supported multisample source, average color resolve, explicit attachment
transitions, exact nonzero mip/layer isolation, depth testing, and pixel readback.
`vk_indirect` conditionally runs
a native generated-work workload when the adapter exposes the semantic
capability: a compute producer writes draw, indexed-draw, dispatch, and count
records; all three generated commands then verify observable output and safe
zero-root execution. The reservation query seam proves each maximum-count
reservation performs one native requirements query and smaller generated
recordings perform none. The two-mode policy matrix uses portable zero-count
generated calls; nonzero emission magnitude remains the responsibility of this
`vk_indirect` workload. `vk_command` also injects a zero-byte preprocess
requirement and proves the command still consumes explicit reservation capacity.
Command
validation separately covers unsupported capability, count bounds, alignment,
short spans, zero work, index formats, and generated-preprocess barrier masks.
`vk_pipeline_cache` covers collision-safe device-owned shader interning,
cache-hit reference stability, last-alias release, unique free-list churn,
full-capacity preflight, and exact cleanup after partial shader preparation,
reflection, native shader, native pipeline, cache insertion, and mid-batch
faults. It uses exact interning and byte-comparison counters for collision and
distinct-storage cases, while cache hits require compact-key probes without
shader-byte comparison. Live tests exercise 1 KiB, 64 KiB, and 1 MiB synthetic
identities and require zero shader probes, byte comparisons, and clones after
interning at every size. A test-only synchronization hook immediately before a
generated-dispatch layout read, creates another compute pipeline on a second
thread, and verifies that recording observes the device's unchanged singleton
layout through the bound pipeline slot. Validation-mode lifetime coverage also
proves a recording command blocks public pipeline destruction, then continues
from its cached native snapshot before discard releases ownership.

Injected native pipeline and EDS3 color-command arrays are the direct
native-call oracle: they verify the declared dynamic-state set and exact emitted
array contents without relying on driver interpretation. `PipelineKey` is
process-local and is never serialized. Format-only identity equality and
inequality tests cover its exact bytes and hash contract, while opaque
driver-cache export/import smoke tests cover the only persistent cache
representation.

`vk_performance` runs complete warm
begin/bind/dispatch/end and render-pass operations against existing pipelines,
shaders, descriptor state, texture views, and allocations. Command-buffer reset
is allowed; allocator creation occurs before the measured interval, and
host/VMA allocation, command-buffer allocation/free, command-pool creation,
image-view creation, and pipeline/shader creation are required to remain zero. Resolution
snapshots begin before pipeline binding: binding the opaque pipeline handle
performs exactly one pipeline-table and one pipeline-cache lookup and records
one `pipeline_bind_commands` increment for the selected bind point, while
retained device-operation resolution, retained-pin borrow, and command-table
lookup remain zero. Each public recording call checks the static
device slot before loading the stable record.
Exact heap counters require one descriptor-set bind per used bind point;
later compatible pipeline changes add none. Compute and graphics cache state
is independent, and command-buffer reset requires the next use to bind again.
Dispatch and draw add no further
resolution and each emits exactly one root push plus its native execution
command. The dispatch test also replaces the bound handle's backing slot with
invalid, different pipeline state after binding and requires the recorded
layout snapshot to remain unchanged. No compile-time exact member-shape guard
is used for private pipeline or command-record state; semantic, resolution,
and native-emission outcomes remain the authority as those representations
change.

Command color-state work coverage runs on every created strict device. One
packet and an identical replacement each require exactly three native calls and
zero host/VMA allocation, resource lookup/lock, native pipeline creation, or
pipeline/cache lookup.

The benchmark runner builds twelve executable targets with `-O1`:
`allocation_bench`, `resource_create_bench`, `descriptor_churn_bench`,
`upload_throughput_bench`, `command_wrapper_bench`,
`command_path_baseline_bench`, `command_record_bench`, `lifecycle_bench`,
`submit_batch_bench`,
`pipeline_cache_bench`, `async_overlap_bench`, and
`completion_wait_scope_bench`.
`command_wrapper_bench` routes through `test/cpu/project.json`; it links only
the CPU project and libc, so it can run without a Vulkan loader, ICD, VMA, or
native Vulkan library. `command_path_baseline_bench` routes through
`test/project.json` and requires the normal headless Vulkan 1.3 environment.
Run them directly with:

```sh
c3c build command_wrapper_bench --path test/cpu -O1
./test/cpu/build/command_wrapper_bench
python3 scripts/build_shaders.py
c3c build command_path_baseline_bench --path test -O1
VK_DRIVER_FILES=/path/to/icd.json ./test/build/command_path_baseline_bench
```

The CPU target reports five alternating direct/public wrapper pairs with exact
per-repetition observation. The Vulkan target reports five equivalent native
operation pairs, zero forbidden work deltas for warmed recording,
non-zero dispatch and buffer-copy readback equivalence, and full
begin-bind-record-end-submit-wait lifecycle cases for 0, 1, 16, and 256
commands under the same two contract modes as the command-recording
benchmark. Lifecycle timing ends only after successful completion wait; its
deterministic work gate permits only the exact command-list allocation/reset
work and rejects unrelated allocation, creation, and registry-lock work. All
elapsed times and ratios are advisory, while schema, work, and equivalence
failures are blocking.
The submit-batch target enables `SUBMIT_WORK_STATS` and submits real batches of
1, 8, 32, 128, and 1,024 executable lists. For every ordinary TRUSTED
batch it requires `resolutions` and `duplicate_visits` equal to the batch
length; one command mutex, one queue submission mutex, and one native
submission; and zero epoch-reset cells, rollback mutexes, and warm host
allocations. Token consumption and pending-list drainage are asserted directly.
The runner rejects missing, duplicate, reordered, malformed, or nonzero
forbidden fields. Duplicate-position tests stop resolution and visits at the
first repeat. Post-claim failure tests require one rollback lock, retryable
records, and no pending link or point. `ns/submit` and `ns/list` remain
advisory. The pipeline-cache target requests 200
topology/cull/front-face/depth-bias permutations through
`cmd_set_raster_state`, reports the requested count, native graphics creates,
cache entries/aliases, and recording/create timings; all permutations share
one immutable pipeline. It additionally reports exact interning and clone/free
outcomes plus zero-through-one compact-key probes for 1 KiB, 64 KiB, and 1 MiB identities. Separate
lookup-side counters require zero shader probes, byte comparisons, and clones
after interning, mirroring the blocking `vk_pipeline_cache` scale test; elapsed
boundary time remains advisory. Command recording covers ordinary and
semantic-hazard barriers, indirect dispatch, and capability-gated generated
dispatch with the sole direct representation. The target runs the two contract
modes and reports direct recording/executable token sizes,
authoritative-record, cell, and total fixed-storage sizes plus lists containing
1, 16, 256, and 4,096 commands. The timing workload
measures five 64-record lists after an untimed 64-record warmup. Before warmup,
the caller reserves 64 preprocess buffers on the measured explicit allocator
for the declared generated workload and exact public pipeline
handle. Alias handles for one native pipeline require independent reservations. Each execute
receives a distinct reserved address, and reuse occurs only after a list is
discarded.
The command target enables test-only resolution counters and measures complete
recording sequences. Minimal pass begin reports one native begin-rendering
command and no dynamic-state commands. Convenience begin reports that command
followed by exactly ten viewport/raster/depth commands and three color-array
commands for a nonempty color domain. A complete `cmd_set_graphics_state`
replacement reports the same state commands, while additional compatible
passes and draws add only their begin/end/draw commands unless another setter
is recorded. The `{2, 16, 256}` matrix rejects hidden pass-boundary replay,
default, or state-diff work. Every warm command performs one static
device-slot liveness load and requires zero retained device-operation
resolution, retained-pin borrow, command-table lookup, pipeline-table/cache
lookup. It also requires
zero encoder-cell computation, packed-lease comparison, frontend phase
transition, and warm allocation. Source helper names, call topology, and
proof-note placement are not blocking contracts.
These process-wide counters use relaxed atomics and are compared only across
externally synchronized benchmark intervals. The native count covers every
Vulkan command emitted by recording paths.
The same target reports cold and warm recording-work snapshots. Warm host
allocations, command-buffer allocations/frees, image-view creations, VMA
allocations, and generated-scratch misses must all be zero; command-buffer
resets demonstrate reuse.
Cold allocator counters separately prove one exact-family pool create, one
complete native command-buffer allocation call, fixed host-scratch allocation,
and the configured buffer count. Both command-policy modes reuse that
allocator outside their measured intervals and require the same warm zeros.
Lifecycle measurements cover submission, cached completed-point polling, and
immediate texture destruction. Required immediate-destruction tests use exact
injected native-destroy counts and immediate handle invalidation to prove the
release occurs before return. An unrelated queue stalled on an unsignaled
timeline must remain stalled while destruction completes, proving destruction
does not wait for queue or device idle; completion-work snapshots independently
require zero native completion queries and waits.

`descriptor_churn_bench` additionally reports texture-destruction and wrapped-
image ownership work at descriptor high-water marks 16, 4,096, and 65,536. Its
feature-gated counters are blocking upper-bound evidence: destruction permits
zero through one ownership decision per texture, and swapchain checks permit
zero through one per image examined. The accompanying elapsed times are
advisory.

The same benchmark reports sampler lookup occupancy 8, 64, 1,024, and 65,536
through production hash/bucket/link/equality helpers. The runner requires a
power-of-two bucket count at least twice occupancy, zero through eight probes for
the selected hit, and zero candidate probes for a guaranteed empty-bucket miss
at every tier. The blocking `vk_descriptor_heap` target independently requires
zero candidate probes for the 65,536-entry empty-bucket miss. Collision,
rollback, bucket consistency, concurrent publication, and teardown are covered
by Vulkan and CPU tests. Collision-chain scenarios require exact candidate-
probe counts for head, middle, tail, and miss lookups.

The lifecycle output requires `cached_poll_queries=0` and
`retirement_locks=0` across each 100,000-poll measured interval and zero native
completion queries/waits, device waits, and deferred-release enqueues in their
respective intervals. Run `python -B scripts/run_benchmarks.py`; one build of
`command_record_bench` executes all four contract/layer combinations with the
same fixed workload. The runner enforces exact policy fields, zero warm
command-table and other resolution/proof work, exact token/storage sizes, and
exact native output. Only TRUSTED command timings with
Vulkan layers disabled participate in
release threshold evaluation. `--validation` still supplies the separate
all-enabled debug run for the other benchmark devices; those timings are not
release comparisons and pinned comparison flags are rejected. Exact schemas
and work invariants hard-fail whenever the runner is executed locally or on a
self-hosted machine; timings are advisory unless runner, driver, and profile
are explicitly pinned. Hosted CI only builds benchmark executables and tests
their schemas. Its blocking live policy coverage comes from
`vk_validation_policy`. See [Performance](performance.md).

Run the behavioral performance targets directly with:

```sh
c3c test vk_performance --path test --test-show-output
c3c test vk_offscreen --path test --test-show-output
c3c test vk_indirect --path test --test-show-output
c3c test vk_pipeline_cache --path test --test-show-output
c3c test vk_queue --path test --test-show-output
```

Capability-gated tests print one terminal `EXERCISED` or `NOT EXERCISED
(<reason>)` line. Portable lanes accept declared unavailability while
preserving every other blocking assertion. A pinned lane sets
`REQUIRED_GPU_CAPABILITIES` and fails unless each named capability reports
`EXERCISED`. The version-pinned Windows mesa-dist-win lane requires
`generated-work` and `generated-scratch-reservation`; Linux distro Mesa may
report either as unavailable.

`vk_shader_reflection` covers selected-entry isolation, absent blocks, exact
compute and graphics roots, wrong stage and entry, and malformed block/member
shapes. It also verifies property-specific block/member diagnostics while
preserving `SHADER_INVALID`. Test-only counters prove a rejected deduplicated
`ShaderId` is reflected once and reaches neither native shader creation nor
cache/output publication; `vk_pipeline_cache` proves the same once-per-identity
behavior for successful batches. `scripts/build_shaders.py` compiles sorted
`.glsl` fixtures with
`glslc` and assembles sorted `.spvasm` fixtures with `spirv-as`. The latter make
multi-entry, offset, member-order, and multiple-block reflection shapes
deterministic. Run the policy and mutation gate with:

```sh
python3 -B -m unittest scripts.test_check_shader_reflection_policy
python3 -B scripts/check_shader_reflection_policy.py
```

The bounded evidence surface is behavioral: checked-in shader sources,
deterministically rebuilt SPIR-V, the reflection-policy mutation gate,
production rejection tests, and root-pointer runtime tests. The ignored SPIR-V
fixtures are regenerated from source before focused validation:

```sh
python3 -B scripts/build_shaders.py
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  c3c test vk_shader_reflection --path test
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  c3c test vk_root_pointer --path test
```

The root-pointer target exercises canonical execution and, through an internal
test-only evidence seam, signed and direct-reference compute forms with
distinctive nonzero address bits, real dereference/readback, and zero-root
dispatch. Normal pipeline preparation continues to reject those alternate
forms. HLSL and Slang are not supported or blocking audit toolchains; add one
only when the project adopts a real frontend path that emits a distinct shape.

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
- `manifest.json` `sources` use the complete final layout:
  `gpu/gpu.c3i`, `gpu/gpu.c3`, `gpu/surface/**`, and `gpu/internal/**`.
  A glob like `*.c3` is rejected, and the default does not recurse into either
  the surface or internal trees.

## 10. CI matrix

CI is shipped: `.github/workflows/ci.yml`, one workflow, three jobs.

```text
linux (blocking): documentation/source-list, API/retired-API/backend-boundary,
    command-operation-table shape, direct command-token representation, generator and ABI drift
    gates; benchmark executable builds and schema tests (no benchmark
    execution); deterministic behavioral performance targets; shader build;
    full lavapipe sweep; and a c3c docgen API reference artifact
windows (blocking): documentation/source-list, backend-boundary, generator and
    ABI drift gates; benchmark executable builds and schema tests (no benchmark
    execution); deterministic behavioral targets; and the full suite via
    mesa-dist-win lavapipe registered in the HKLM Vulkan driver registry
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
the full public module plus the typed private Vulkan stub, linking no native
libraries — a clean checkout runs them with no Vulkan/VMA installed, and CI
runs them before any native setup.

Shared CPU/shader structs come from `.abi` schemas (see `docs/shader_abi.md`
§12). Generated outputs are committed; `scripts/gen_abi.py --check` is the drift
gate — run it as part of any full test sweep, and rerun `scripts/gen_abi.py`
(then `scripts/build_shaders.py`) after editing a schema. Run the split-generator
unit coverage with `python3 -B -m unittest scripts.test_gen_abi`.

CI tiers (`.github/workflows/ci.yml`):

| Tier | Platform | Blocking |
|---|---|---|
| Generator tests, drift gate, shader build | linux + windows | yes |
| Smoke linkage + full lavapipe test sweep + api-reference artifact | linux | yes |
| Getting-started walkthrough (docs-walkthrough job, `scripts/run_doc.py`) | linux | yes |
| Link proof (smoke) + pure-CPU targets | windows | yes |
| lavapipe (mesa-dist-win) Vulkan sweep | windows | yes |
| Deterministic behavioral performance invariants | linux + windows | yes |
| Benchmark/scanner/assembly parser unit coverage | linux + windows | yes |
| Benchmark nanosecond comparisons | portable runners | no, unless runner, driver, and profile are pinned |
| Live benchmark and generated-assembly execution | local/self-hosted | advisory unless the complete comparison identity is pinned |
| Descriptor-indexing device/heap | linux + windows | yes |

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
construct a complete GraphicsState for the render area
include one ColorTargetState per pipeline color format, or none for depth-only
bind the compatible graphics pipeline
record that packet, or use transactional begin-with-state
begin the pass
clear to known color
render primitive
copy the render target to readback storage
sample a small set of pixels
compare with tolerance for floating formats
```

Separate state tests reuse a complete packet across passes, replace it
explicitly, exercise each optional partial setter before and during a pass, and
prove reset clears checked initialization. Invalid convenience input must leave
active-pass metadata, tracked references, initialization, and native command
emission unchanged. FULL regular and generated draws both reject a missing
complete packet before native execution.

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
creation context if tracked
```

The callback form carries the applicable subset as a borrowed
`resource_lifetime` message; the stderr form remains the no-callback fallback.

## 14. Failure path tests

Tests should cover specific faults:

```text
invalid handle -> INVALID_HANDLE
allocation exact-range fit -> success
zero allocation size, malformed alignment, or unavailable span capability -> INVALID_ARGUMENT
allocation extent or offset overflow -> INVALID_ARGUMENT
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
two-mode validation-policy matrix and behavioral command-family checks pass
direct command lifecycle tests pass
root-pointer compute sample works
bindless texture compute sample works
offscreen graphics sample readback matches expected output
SDL3 triangle sample queries the selected format/count, presents, and resizes without fixed image-state tables
GPU-driven indirect draw sample works
memory stats report plausible budgets
leak reports are clean after all samples
public API docs match signatures
explicit command allocators are balanced and retired queue-based begin/reservation signatures are absent
warm command recording/submission paths allocate no host/native/VMA/temp-pool storage
warm command recording performs zero command-table lookup and zero other resolution/proof work
no public API signature exposes vk::, vma::, or sdl:: types
public sources declare only gpu and the three platform surface modules
private sources declare only gpu::internal and gpu::internal::vk
documentation links resolve
manifest sources cover every shipped C3 source
```
