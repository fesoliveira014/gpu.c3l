# Strict GPU Architecture Tasks

This plan implements the approved [requirements](requirements.md) and [design](design.md). It evolves `gpu` in place. `gpu::compat` is an additive capability on the same runtime, device, resources, queues, commands, synchronization, and presentation model.

## Execution rules

- Target C3 0.8.0 and follow `AGENTS.md` and `docs/style.md`.
- Deliver one reviewer-sized change per ready-for-review pull request. Do not open draft pull requests.
- Start behavior changes with a focused failing test or compile fixture. Mark a task complete only after its verification passes and the change is merged.
- Update affected public documentation, project source lists, and benchmarks in the same task as a contract change. For each breaking public change, update affected samples in a companion `gpu.c3l-samples` pull request and advance its pinned `gpu.c3l` submodule to that exact library commit.
- Keep public APIs GPU-shaped. Vulkan and VMA types, feature names, layouts, queue families, result codes, and dispatch details remain private.
- Do not introduce a parallel strict profile or move the current implementation into `gpu::compat`.
- Do not add `gpu::compat` source before Gate C or Vulkan 1.2 fallbacks before Gate D.
- Reserve `gpu::alloc` for a possible future extension. This initiative must not implement allocation-owning convenience policy in the root module.
- Keep milestone, phase, issue, and pull-request labels out of source identifiers, test names, code comments, and user documentation.
- Remove generated build residue and temporary worktrees before handoff.

## Review gates

- **Gate A — plan approval:** approve this task sequence before source implementation begins.
- **Gate B — strict architecture:** complete Milestones 1–7 and review the canonical strict API as a coherent whole.
- **Gate C — compatibility contract:** approve separate requirements, design, and tasks for compatibility per-draw data and descriptor authoring. This gate resolves the design question tracked by issue #33 before compatibility source exists.
- **Gate D — shared-backend compatibility:** verify strict-only, compatibility-only, and combined devices on one backend before Vulkan 1.2 work begins.

## Milestones

| Milestone | Outcome | Gate |
|---|---|---|
| 0. Stabilization record | Preserve completed preparatory work without reviving the superseded profile split. | Complete |
| 1. Runtime and discovery | Explicit runtime, borrowed adapters, immutable requests, and runtime-owned surfaces. | — |
| 2. Devices and queues | Multiple independent devices, semantic queues, ownership validation, and retryable teardown. | — |
| 3. Commands and completion | Transient one-shot commands, compact completion points, explicit submission, and private native synchronization. | — |
| 4. Memory and textures | Independent allocations, checked spans, placed textures, immediate lifetimes, and application-owned allocation policy. | — |
| 5. Binding and pipelines | Strict heaps, automatic shader identity, explicit pipeline identity, and bind-before-execute commands. | — |
| 6. Synchronization and rendering | Global hazards, semantic texture transitions, dynamic render passes, and strict presentation. | — |
| 7. Strict release | Remove transitional API, migrate consumers, and collect correctness and performance evidence. | Gate B |
| 8. Compatibility extension | Add descriptor-set semantics to the shared architecture after a separate design review. | Gates C and D |
| 9. Vulkan 1.2 | Add private fallbacks that preserve the approved public semantics. | After Gate D |

## Completed stabilization

These changes remain useful inputs. They do not authorize the superseded wholesale compatibility extraction.

- [x] Establish source and documentation quality checks in PRs #204 and #205.
- [x] Stabilize handle and backend validation groundwork in PRs #206–#208.
- [x] Improve shader ABI and pipeline preparation groundwork in PRs #209–#211.
- [x] Record follow-up findings from the abandoned profile split in PR #213.
- [x] Approve the in-place strict architecture requirements and design in PR #216.

## Milestone 1 — Runtime and discovery

### 1.1 Explicit runtime and borrowed adapters

- [x] Add `Runtime`, `RuntimeDesc`, `Adapter`, `AdapterInfo`, and explicit create, enumerate, query, and destroy operations in `gpu/runtime.c3` and `gpu/adapter.c3`; move native discovery ownership from `gpu/vk/instance.c3` into per-runtime state in `gpu/vk/runtime.c3` and `gpu/vk/adapter.c3`.
  - **Depends on:** Gate A.
  - **Contract:** imports remain inert; adapters are borrowed from a runtime; runtime destruction rejects live surfaces or devices; backend and driver versions are diagnostic only.
  - **Edges:** no adapter, repeated enumeration, stale adapter after runtime destruction, multiple runtimes, and transactional runtime creation failure.
  - **Verify:** import-only fixtures for every public module; CPU handle tests; native enumeration with two runtimes; `scripts/check_public_api.py`.

### 1.2 Immutable semantic device requests

- [x] Add canonical `DeviceRequest`, strict request construction, semantic support queries, and transactional request validation in `gpu/device.c3`, `gpu/caps.c3`, `gpu/vk/adapter.c3`, and `gpu/vk/device.c3`.
  - **Depends on:** 1.1.
  - **Contract:** support, requested capabilities, and enabled capabilities are separate; request contents become immutable at creation; unsupported requirements identify the unmet semantic requirement; no API-version selector is public.
  - **Edges:** strict-only, empty, unsupported, duplicate contribution, and failure after temporary native allocation.
  - **Verify:** pure-CPU request composition tests and native successful/failing device creation tests with leak checks.

### 1.3 Runtime-owned platform surfaces

- [x] Replace the root untyped platform-handle path with `gpu::surface::win32`, `gpu::surface::wayland`, and `gpu::surface::x11` constructors backed by shared `Surface` tokens in `gpu/surface/`, `gpu/swapchain.c3`, and `gpu/vk/surface/`.
  - **Depends on:** 1.1 and 1.2.
  - **Contract:** a surface belongs to one runtime; presentation support is queried for an adapter and surface before device creation; importing any surface module is inert; SDL remains outside this repository.
  - **Edges:** wrong runtime, stale surface, unsupported presentation, invalid native handles, dormant surface, and runtime destruction with a live surface.
  - **Verify:** platform compile fixtures, lifetime tests, presentation-support tests, and a public-source scan for `PlatformKind` and untyped native-handle pairs.

## Milestone 2 — Devices and queues

### 2.1 Concurrent generational device registry

- [x] Replace the process-wide active device with a generational slot registry in `gpu/device.c3`, `gpu/types.c3`, `gpu/vk/device.c3`, and focused registry tests under `test/src/`.
  - **Depends on:** 1.2.
  - **Contract:** multiple devices coexist; steady-state resolution is a slot and generation check; registry mutation is synchronized; most public operations use short-lived pins, while command tokens retain and borrow one pin.
  - **Edges:** stale generation, concurrent create/destroy, slot reuse, generation exhaustion, and operations racing a closing device.
  - **Verify:** deterministic CPU concurrency tests, stale-handle tests, and native two-device isolation tests.

### 2.2 One backend state per device

- [x] Split global Vulkan state into runtime-owned discovery and device-owned backend state across `gpu/vk/backend.c3`, `gpu/vk/device.c3`, `gpu/vk/internal.c3`, and resource tables.
  - **Depends on:** 2.1.
  - **Contract:** every device owns its dispatch, queues, completion state, and resource tables; optional capability state is initialized only when requested; no parallel compatibility backend exists.
  - **Edges:** partial backend initialization, device loss isolated to one device, simultaneous devices on different adapters, and cleanup after publication failure.
  - **Verify:** fault-injection rollback tests and native independent create/use/destroy tests for two devices.

### 2.3 Deterministic token ownership validation

- [x] Extend all runtime, surface, device, allocation, texture, view, sampler, pipeline, swapchain, and command tokens with deterministic owner validation in their public modules and backend tables.
  - **Depends on:** 2.1 and 2.2.
  - **Contract:** stale and cross-device tokens fault before backend mutation even when an explicit cast defeats nominal typing; the private token representation may vary by resource kind.
  - **Edges:** same slot index on different devices, stale parent with live-looking child bits, cross-runtime surface use, and validation during device closing.
  - **Verify:** table-driven CPU misuse tests and backend mutation counters proving early rejection.

### 2.4 Explicit semantic queue handles

- [x] Add queue-role/count requests, `Queue` handles, queue queries, and per-resource access domains in `gpu/queue.c3`, `gpu/device.c3`, `gpu/memory.c3`, `gpu/texture.c3`, `gpu/vk/queue.c3`, and resource creation code.
  - **Depends on:** 2.2 and 2.3.
  - **Contract:** public requests use semantic roles, never family indices; resources admitted to roles on distinct native families use private concurrent sharing; explicitly named resources reject unadmitted queues.
  - **Edges:** unavailable count, one queue serving several roles, same-family multi-role access, cross-family sharing, and GPU-pointer-only access as a documented caller precondition.
  - **Verify:** queue-selection and access-domain tests covering single-role, same-family, and cross-family devices.

### 2.5 Non-blocking retryable device destruction

- [x] Implement the approved live/closing/device-destroy state machine in `gpu/device.c3` and `gpu/vk/device.c3`, resolving issues #214 and the device portion of #200.
  - **Depends on:** 2.1–2.4.
  - **Contract:** live children return `RESOURCE_IN_USE`; incomplete queue work and active pins return retryable `DEVICE_BUSY`; destruction never waits; command tokens retain pins until submit or discard; failed attempts preserve state and generation.
  - **Edges:** new pin while closing, child creation between the live-child check and closing mark, active recording/executable token, completion racing destruction, device loss, and retry after every failure branch.
  - **Verify:** CPU state-machine and concurrency tests plus validation-clean native teardown tests.

## Milestone 3 — Commands and completion

### 3.1 Compact queue-owned completion points

- [x] Add `CompletionPoint` and queue-owned monotonic submission sequences in `gpu/sync.c3`, `gpu/queue.c3`, `gpu/vk/sync.c3`, and `gpu/vk/queue.c3`.
  - **Depends on:** 2.4.
  - **Contract:** a point identifies one queue and sequence, fits within two machine words, allocates no public object, remains reusable until device destruction, and supports host poll and wait operations; timeout expiry faults retryable `WAIT_TIMEOUT` without invalidating or changing the point.
  - **Edges:** zero or reserved sequence, sequence exhaustion before native submission, stale queue/device, foreign device, already-complete points, and wait timeout expiry.
  - **Verify:** representation-size assertion; packing, monotonicity, exhaustion, stale, poll, and wait tests; public child-count instrumentation proving point creation allocates no public object.

### 3.2 Transactional submission and cross-queue waits

- [x] Change submission in `gpu/queue.c3`, `gpu/command.c3`, `gpu/vk/queue.c3`, and `gpu/vk/sync.c3` to consume executable command tokens only after successful native submission and return one `CompletionPoint`.
  - **Depends on:** 3.1.
  - **Contract:** cross-queue waits accept reusable completion points; same-queue order is inherent; successful submission publishes exactly one contiguous queue sequence; validation or retryable failure publishes no point and preserves retryable tokens; destruction readiness observes completed queue timelines without requiring `wait_queue_idle`.
  - **Edges:** empty submit, duplicate token, mixed queues/devices, stale wait point, wait on a later same-queue sequence, device loss, and partial native preparation failure.
  - **Verify:** command state-machine tests and native transfer/compute/graphics cross-queue tests, including failure injection before submit and completed-submission destruction readiness.

### 3.3 Remove public synchronization objects

- [x] Remove public semaphore, caller-managed counter, and `wait_queue_idle` contracts from `gpu/sync.c3`, `gpu/queue.c3`, `gpu/gpu.c3i`, backend adapters, docs, and samples; retain native synchronization privately.
  - **Depends on:** 3.2.
  - **Contract:** completion points are the only public queue-progress primitive; removal is in place rather than wrapped by a compatibility layer.
  - **Edges:** presentation and teardown call sites still needing native binary synchronization; ensure device destruction remains non-blocking.
  - **Verify:** compile-fail fixtures for retired symbols, public API scan, and all submission tests.

### 3.4 Transient one-shot command lifecycle

- [x] Replace public recording-context and pool policy with `begin_commands(queue)`, recording `CommandList`, successful end to an executable token, and consuming `discard_commands` in `gpu/command.c3`, `gpu/vk/command.c3`, and `gpu/vk/command_state.c3`.
  - **Depends on:** 2.5 and 3.2.
  - **Contract:** recording and executable tokens retain device pins; native pools are private and may be cached or sharded; command calls use pinned state and take no global registry lock.
  - **Edges:** abandon before end, double end/discard, validation failure leaving recording unchanged, worker-thread reuse, submit-once enforcement, and device loss.
  - **Verify:** CPU lifecycle tests, multithreaded native recording tests, and lock/allocation instrumentation on hot command calls.

### 3.5 One-shot swapchain readiness

- [x] Migrate acquisition, submission, and presentation in `gpu/swapchain.c3`, `gpu/queue.c3`, `gpu/vk/swapchain.c3`, and `gpu/vk/queue.c3` to one-shot readiness plus ordinary render completion points.
  - **Depends on:** 3.2–3.4 and 1.3.
  - **Contract:** acquire returns a borrowed texture and readiness token; submission consumes readiness; presentation consumes the acquired image and accepts a same-device completion point covering rendering; native bridge objects remain private.
  - **Edges:** unused or double-used readiness, wrong image or device, out-of-date swapchain, dormant surface, surface loss, and retryable present failure.
  - **Verify:** state-machine tests and validation-clean acquire/render/present native tests.

## Milestone 4 — Memory and textures

### 4.1 Independent allocations and checked spans

- [x] Add independent storage ownership with `GpuAllocation`, non-owning `GpuSpan`, memory classes, mapping, and address queries in `gpu/memory.c3` and `gpu/vk/allocation.c3`.
  - **Depends on:** 2.3 and 2.4.
  - **Contract:** allocations own storage and release exactly once; spans cannot release parents; span slicing checks overflow and bounds; CPU mappings and GPU addresses exist only for supported memory classes.
  - **Edges:** zero length, end-boundary span, integer overflow, wrong device, unmapped memory, unavailable address, and releasing with live placements.
  - **Verify:** exhaustive CPU range/ownership tests and native allocation/map/address tests.

### 4.2 Explicit mapped-memory visibility

- [x] Add mapped-span flush and invalidate operations in `gpu/memory.c3` and `gpu/vk/allocation.c3` with private non-coherent range alignment.
  - **Depends on:** 4.1.
  - **Contract:** flush makes CPU writes visible to the GPU; invalidate makes completed GPU writes visible to the CPU; coherent memory is a semantic no-op.
  - **Edges:** unaligned ranges, zero length, overflow, unmapped spans, non-host-visible allocations, and calls before completion.
  - **Verify:** CPU range normalization tests and coherent/non-coherent upload/readback native tests.

### 4.3 Span/address data operations

- [x] Migrate copies, fills, index data, indirect arguments, upload, and readback from `BufferHandle` to spans or addresses in `gpu/command.c3`, transfer helpers, shader ABI, tests, benchmarks, and samples.
  - **Depends on:** 4.1, 4.2, and 3.4.
  - **Contract:** no generic buffer object remains in the strict public surface; operations validate ranges and admitted queue roles before recording mutation.
  - **Edges:** overlapping copy, alignment, indirect/count bounds, address-only lifetime responsibility, and cross-device spans.
  - **Verify:** compile-fail fixtures for `BufferHandle`; copy/fill/index/indirect tests; shader ABI scan for buffer-binding objects.

### 4.4 Texture requirements and placed creation

- [x] Add pre-creation texture requirements and placed texture creation in `gpu/texture.c3`, `gpu/vk/texture.c3`, and allocation placement tracking.
  - **Depends on:** 4.1 and 2.4.
  - **Contract:** requirements precede allocation; placed creation validates compatibility, size, alignment, offset, queue access, and dedicated-only requirements before backend mutation; texture destruction never releases placement.
  - **Edges:** wrong memory class or type, overlapping live placement, out-of-range offset, alignment overflow, dedicated-required request, stale allocation, and failure after validation.
  - **Verify:** table-driven requirements/placement tests and backend mutation counters for every invalid request.

### 4.5 Transactional dedicated textures

- [x] Add dedicated texture creation returning separate `Texture` and `GpuAllocation` tokens in `gpu/texture.c3` and `gpu/vk/texture.c3`.
  - **Depends on:** 4.4.
  - **Contract:** native texture, compatible memory, and binding are one transaction; failure publishes neither token; callers destroy the texture and release the allocation separately.
  - **Edges:** allocation failure after native texture creation, binding failure, publication failure, dedicated-only requirements, and both valid destruction orders subject to placement rules.
  - **Verify:** fault injection at every transaction step, leak checks, and independent token ownership tests.

### 4.6 Immediate caller-managed resource lifetime

- [x] Remove deferred core resource release from `gpu/vk/deferred.c3` and public destruction paths, resolving issue #199 and the lifetime portion of #200.
  - **Depends on:** 3.1, 4.1, and 4.5.
  - **Contract:** destroying a non-WSI GPU-visible resource is immediate and never waits or queues deferred work; no live recording, executable token, or incomplete submission may reference it; releasing an allocation with live placed textures faults without consuming it. Swapchain destruction and resize remain under the presentation contract until 6.5.
  - **Edges:** resources referenced through raw GPU addresses or shader indices cannot always be discovered and remain a caller precondition; detectable command references must reject destruction.
  - **Verify:** every destruction-order permutation, no-wait instrumentation, placement rejection tests, and validation-clean teardown.

### 4.7 Remove root allocation policy and frame/readback wrappers

- [x] Remove `FrameToken`, root `begin_frame`/`end_frame`, `@with_frame`, frame/persistent arenas, readback tickets, and deferred-release helpers from public source, docs, tests, and samples; replace usage with allocations, mapped spans, commands, and completion points.
  - **Depends on:** 3.1–3.4 and 4.1–4.6.
  - **Contract:** readback is CPU-cached span → record copy → submit → wait or poll → invalidate → direct CPU access; frame-scoped allocation policy remains application code and a possible future `gpu::alloc` extension.
  - **Edges:** samples must demonstrate safe reuse and release from completion points without implying automatic ownership tracking.
  - **Verify:** retired-symbol API scan and compile fixtures; upload/readback tests; sample-local linear allocator example without root convenience APIs.

## Milestone 5 — Binding and pipelines

### 5.1 Interned sampler identity and strict publication

- [ ] Split immutable device-interned `Sampler` identity from capability-gated strict heap publication in `gpu/texture.c3`, `gpu/descriptor_heap.c3`, and corresponding Vulkan files.
  - **Depends on:** 2.2 and 2.3.
  - **Contract:** sampler identity works on compatibility-only devices; strict publication returns a backend-neutral shader index only when strict capability is enabled; samplers live until device destruction.
  - **Edges:** duplicate descriptors, exhaustion, unsupported strict publication, cross-device sampler, and concurrent interning.
  - **Verify:** CPU interning/concurrency tests and strict-only, compatibility-only, and combined-device publication tests.
  - **Verification deferral:** the public compatibility-only and combined-device matrix remains gated on task 8.2; the 5.1 implementation slice covers strict-only publication and a private strict-disabled backend seam.

### 5.2 Device-wide strict texture and sampler heaps

- [x] Finalize device-wide view publication and raw shader-visible indices in `gpu/descriptor_heap.c3`, `gpu/texture.c3`, `gpu/vk/descriptor_heap.c3`, and `gpu/vk/texture.c3`.
  - **Depends on:** 5.1 and 4.4.
  - **Contract:** heap implementation selection is private; index width is ABI-pinned; CPU generation metadata validates API operations but is not embedded in shader values; caller lifetime covers indices stored in GPU memory.
  - **Edges:** capacity exhaustion, stale view, index reuse after completion, unsupported view kind, and concurrent publish/release.
  - **Verify:** index-width ABI tests, exhaustion/rollback tests, stale CPU token tests, and shader sampling/storage native tests.

### 5.3 Remove backend-shaped heap configuration

- [x] Remove `DescriptorHeapMode` and descriptor-indexing or descriptor-buffer feature choices from `gpu/caps.c3`, `gpu/descriptor_heap.c3`, `gpu/gpu.c3i`, docs, and samples; select native facilities in `gpu/vk/descriptor_heap.c3`.
  - **Depends on:** 5.2.
  - **Contract:** callers request strict semantics and query semantic limits; backend strategy is diagnostic at most.
  - **Edges:** devices supporting several implementations, devices supporting none, differing capacities, and diagnostic reporting without behavior branching.
  - **Verify:** public API scan and native capability-selection tests using mocked feature matrices.

### 5.4 Automatic shader-code identity

- [x] Replace public shader-module handles with lightweight `ShaderCode` values and library-computed content identity in `gpu/pipeline.c3`, `gpu/shader_abi.c3`, `gpu/vk/shader.c3`, and pipeline creation.
  - **Depends on:** 2.3.
  - **Contract:** `prepare_shader_code` validates borrowed IR and computes an opaque process-local digest; the IR remains immutable and alive while used; collisions compare length and bytes; one-off creation may prepare raw IR internally; no public `ShaderHandle` exists.
  - **Edges:** empty or invalid IR, identical bytes from different storage, caller mutation of borrowed bytes, hash collision handling, and cross-device prepared data.
  - **Verify:** deterministic identity/deduplication tests, invalid-IR tests, and compile fixtures rejecting retired shader handles.

### 5.5 Explicit deterministic pipeline identity and batches

- [x] Redesign graphics and compute pipeline creation in `gpu/pipeline.c3`, `gpu/vk/pipeline_compute.c3`, `gpu/vk/pipeline_graphics.c3`, and `gpu/vk/pipeline_cache.c3` around explicit immutable state and batch deduplication.
  - **Depends on:** 5.4 and 5.3.
  - **Contract:** native compilation occurs during creation; shared shader IR deduplicates within a batch; depth/stencil is separate immutable state; viewport/scissor are dynamic; baseline blend state remains in graphics identity.
  - **Edges:** partial batch failure, cache failure, duplicate pipeline descriptions, shared shader stages, unsupported state, and transactional publication.
  - **Verify:** identity tests, shared-IR native creation counters, failure rollback tests, and pipeline-cache benchmarks.

### 5.6 Bind pipelines separately from execution

- [x] Add strict pipeline and depth/stencil state binding commands and remove pipeline handles from draw and dispatch argument records in `gpu/command.c3`, `gpu/pipeline.c3`, and Vulkan command code.
  - **Depends on:** 5.5 and 3.4.
  - **Contract:** draw and dispatch contain root addresses plus execution arguments; active strict pipeline is command state; no native compilation or variant synthesis occurs during execution commands.
  - **Edges:** execute without pipeline, wrong pipeline kind, stale pipeline, missing required stage root, dynamic-state omission, and validation failure leaving command state unchanged.
  - **Verify:** command state-machine tests, compile fixtures for retired signatures, and instrumentation proving zero draw or dispatch compilation.

### 5.7 Root-pointer shader ABI and generated work

- [x] Update `gpu/shader_abi.c3`, `scripts/gen_abi.py`, generated shader headers, shader tests, and strict samples for one root GPU address per participating stage and backend-neutral heap indices.
  - **Depends on:** 4.3, 5.2, and 5.6.
  - **Contract:** address and index widths are pinned; direct and indirect work use addresses; optional GPU-generated work is a separately queried semantic capability and is never CPU-emulated.
  - **Edges:** missing stage root, alignment, indirect/count overflow, unsupported generated work, and shared root-record layouts.
  - **Verify:** ABI drift check, reflection tests, direct/indirect native tests, and unsupported-capability tests.

## Milestone 6 — Synchronization and rendering

### 6.1 Global execution and memory barriers

- [x] Replace resource/range-shaped generic barriers with global semantic stage and hazard barriers in `gpu/sync.c3`, `gpu/command.c3`, `gpu/vk/sync.c3`, and tests.
  - **Depends on:** 3.4 and 4.3.
  - **Contract:** generic barriers contain no handles, addresses, ranges, layouts, or queue families; no barrier is inferred; cross-queue order still requires completion-point waits.
  - **Edges:** transfer, shader, indirect, descriptor, color, depth, host, and presentation hazards; empty and contradictory masks.
  - **Verify:** pure-CPU mapping tests and validation-clean native hazard matrix.

### 6.2 Semantic texture transitions

- [x] Replace public `TextureLayout` and backend-shaped transition fields with texture or view plus semantic previous and next uses in `gpu/texture.c3`, `gpu/sync.c3`, `gpu/vk/sync.c3`, and render/transfer call sites.
  - **Depends on:** 6.1 and 4.4.
  - **Contract:** native layouts remain private; transitions are always explicit; debug tracking may validate expected use but cannot change release behavior.
  - **Edges:** first use, presentation, depth/stencil aspects, subresources/views, mismatched expected use, and queues outside the access domain.
  - **Verify:** transition mapping tests, retired-layout API scan, and validation-layer transfer/render/present tests.

### 6.3 Dynamic render-pass commands

- [x] Express render passes as begin and end commands with attachments, load/store operations, and clear values in `gpu/command.c3`, a new `gpu/render_pass.c3`, and `gpu/vk/render_pass.c3`.
  - **Depends on:** 6.2 and 5.6.
  - **Contract:** no public render-pass or framebuffer objects; begin/end add no synchronization; Vulkan 1.3 uses an appropriate private implementation without exposing it.
  - **Edges:** attachment extent or sample mismatch, missing transitions, resolve/depth attachments, nested passes, and pipeline incompatibility detected before recording mutation.
  - **Verify:** CPU descriptor validation and validation-clean offscreen color/depth rendering tests.

### 6.4 Address-based direct, indirect, and generated work

- [x] Complete address-based draw, indexed draw, dispatch, indirect/count, and optional generated-work recording in `gpu/command.c3`, `gpu/vk/command.c3`, and workload tests.
  - **Depends on:** 5.7, 6.1, and 6.3.
  - **Contract:** execution arguments remain pipeline-free and buffer-object-free; generated work is exposed only when supported; no CPU-loop emulation.
  - **Edges:** alignment, count bounds, missing barriers, unsupported generated work, index format, and zero-work commands.
  - **Verify:** direct/indirect native workloads, negative validation tests, and indirect/generated-work benchmarks.

### 6.5 Strict presentation integration

- [x] Integrate swapchain textures with shared texture views, transitions, render-pass commands, semantic queue roles, readiness, and completion in `gpu/swapchain.c3` and Vulkan WSI code.
  - **Depends on:** 3.5, 6.2, and 6.3.
  - **Contract:** swapchain images use the ordinary resource and synchronization model; public presentation exposes no native semaphore or layout; destruction and resize never hide queue-idle waits; resize and loss faults retain distinct retry contracts.
  - **Edges:** acquire without render, present without completion, completion that does not cover rendering, destruction or resize during use, dormant surface, and device or surface loss.
  - **Verify:** state-machine tests and validation-clean resize/acquire/render/present tests.

## Milestone 7 — Strict release

### 7.1 Remove transitional root API

- [ ] Delete or replace every superseded public symbol in `gpu/gpu.c3`, `gpu/gpu.c3i`, module sources, tests, and docs; add compile fixtures that pin the canonical strict surface.
  - **Depends on:** Milestones 1–6.
  - **Contract:** no parallel profile, legacy alias, backend-shaped escape hatch, `BufferHandle`, `ShaderHandle`, `FrameToken`, public semaphore, readback ticket, `DescriptorHeapMode`, or `TextureLayout` remains.
  - **Edges:** recursive imports and generated interface output must not accidentally expose private backend declarations.
  - **Verify:** `scripts/check_public_api.py`, generated docs scan, compile-pass canonical fixture, compile-fail retired-symbol fixtures, and full test suite.

### 7.2 Canonicalize and audit strict samples

- [ ] Finish and audit the incremental `gpu.c3l-samples` migration to runtime → adapter → request → device → queue → allocation → record → submit → completion usage, keep allocation policy local to samples, and pin the samples repository to the merged 7.1 library commit.
  - **Depends on:** 7.1.
  - **Contract:** getting-started material teaches strict use first; no sample imports compatibility; readback and presentation use the approved completion/lifetime flow.
  - **Edges:** headless compute, windowed rendering, upload/readback, resize, multiple devices, and cleanup after partial setup.
  - **Verify:** build and run packaged samples on supported platforms; scan samples for retired symbols and Vulkan-shaped public choices.

### 7.3 Rewrite current-state documentation and tooling

- [ ] Update `docs/architecture.md`, `docs/api.md`, `docs/memory.md`, `docs/shader_abi.md`, `docs/threading.md`, `docs/vulkan_backend.md`, getting-started/cookbook/limitations/testing documents, public doc strings, manifest source lists, and CI scripts.
  - **Depends on:** 7.1 and 7.2.
  - **Contract:** documentation describes implemented behavior, remains lean, and contains no narration, development labels, issue or PR references, or backend details in public contracts.
  - **Edges:** distinguish target planning documents from current API until release; keep platform and VMA setup accurate without leaking implementation into usage APIs.
  - **Verify:** documentation links/walkthroughs, packaged-library builds, source-list checks, API leakage scan, and comment review.

### 7.4 Strict correctness and performance evidence

- [ ] Replace obsolete benchmarks and publish reproducible baselines for allocation, pipeline creation, command recording, submission, completion polling, barriers, indirect work, and destruction in `test/src/`, `scripts/run_benchmarks.py`, and `docs/performance.md`.
  - **Depends on:** 7.1–7.3.
  - **Contract:** hot command recording has no registry lock, hidden per-command allocation, or draw-time compilation; completion points have no per-point allocation; destruction has no wait or deferred work.
  - **Edges:** separate debug validation cost, cold/warm pipeline cache, CPU-only evidence, driver variability, and hardware metadata.
  - **Verify:** benchmark schema checks, repeated native runs with adapter/driver/API metadata, allocation/lock instrumentation, and regression thresholds. Complete Gate B review after this evidence is available.

## Milestone 8 — Compatibility extension

### 8.1 Approve compatibility per-draw and descriptor SDD

- [ ] Write and approve separate `gpu::compat` requirements, design, and tasks covering descriptor layouts, arena lifetimes, set writes, compatibility pipelines, binding commands, and per-draw data; reconcile issue #33.
  - **Depends on:** Gate B.
  - **Contract:** this is a hard design gate; it adds no source or placeholder API; compatibility extends shared types and explicitly requested device capabilities.
  - **Edges:** compatibility-only devices, combined devices, shader authorship, transient versus persistent set lifetime, per-draw frequency, and strict/compat alternation.
  - **Verify:** Gate C approval and a requirements-to-task coverage map.

### 8.2 Compatibility request and capability state

- [ ] After Gate C, add `gpu::compat` request composition, support/limit queries, and optional per-device backend state without duplicating root ownership.
  - **Depends on:** 8.1.
  - **Contract:** importing `gpu::compat` is inert; adding requirements is explicit; compatibility-only and combined requests produce ordinary `gpu::Device` values; unsupported requests publish no device.
  - **Edges:** support without request, duplicate contribution, strict unsupported but compatibility supported, and rollback after optional-state initialization failure.
  - **Verify:** import, request composition, creation matrix, transactional failure, and one-backend-state tests.

### 8.3 Descriptor layouts, arenas, sets, and writes

- [ ] Implement only the approved compatibility descriptor model under `gpu/compat/` and private `gpu/vk/compat/`, reusing root allocations, spans, textures, views, samplers, and completion points.
  - **Depends on:** 8.2.
  - **Contract:** arenas express allocation/reset lifetime without native pools; transient reset requires a completed caller point; persistent arenas release sets individually; batched writes are transactional.
  - **Edges:** exhaustion, fragmentation or rollover, reset before completion, stale set after reset, mixed-device resource writes, and partial invalid write batches.
  - **Verify:** CPU arena state-machine tests, native write/bind tests, rollback tests, and public API leakage scan.

### 8.4 Distinct compatibility pipelines and commands

- [ ] Implement compatibility graphics and compute pipelines plus binding and execution commands from the approved per-draw design, sharing root command lists, render passes, resources, synchronization, and presentation.
  - **Depends on:** 8.3.
  - **Contract:** compatibility pipeline types are nominally distinct; no shader translation occurs; binding-model mismatch faults before recording mutation; strict and compatibility pipelines may alternate in one command list.
  - **Edges:** missing layouts or sets, stale arena generation, switching models inside or outside a render pass, shared resource hazards, and compatibility-only devices rejecting strict commands.
  - **Verify:** compile-fail nominal type fixtures and native strict-only, compatibility-only, combined, and alternating-pipeline workloads.

### 8.5 Compatibility samples, documentation, and evidence

- [ ] Add focused compatibility samples and descriptor benchmarks without making compatibility the primary getting-started path.
  - **Depends on:** 8.4.
  - **Contract:** documentation explains explicit capability request and shader-interface differences; shared lifecycle and synchronization are not duplicated.
  - **Edges:** transient reset timing, persistent set release, per-draw update cost, and combined-device teardown.
  - **Verify:** packaged sample builds, descriptor allocation/write/reset benchmarks, documentation scan, and Gate D review.

## Milestone 9 — Vulkan 1.2

### 9.1 Private semantic fallback matrix

- [ ] Define and test a private feature and extension matrix in `gpu/vk/` mapping approved semantics to promoted core or extension facilities without changing public requests or capability names.
  - **Depends on:** Gate D.
  - **Contract:** Vulkan version never selects strict or compatibility mode; exact semantic support determines request validation; no public native feature booleans are added.
  - **Edges:** promoted core feature disabled, extension present without required subfeature, mixed feature sources, and diagnostic version reporting.
  - **Verify:** mocked feature-matrix tests and identical public capability results for equivalent Vulkan 1.2 and 1.3 devices.

### 9.2 Synchronization, rendering, and requirements fallbacks

- [ ] Add private Vulkan 1.2 paths for global barriers, semantic texture transitions, dynamic render-pass semantics, placed texture requirements, and other exact fallbacks in the existing backend modules.
  - **Depends on:** 9.1.
  - **Contract:** public behavior, faults, command state, and lifetime rules remain identical; legacy render-pass and framebuffer objects may be synthesized and cached privately.
  - **Edges:** cache identity/lifetime, fallback feature gaps, synchronization equivalence, requirements-query mutation, and device loss.
  - **Verify:** run the same public conformance tests against Vulkan 1.2 and 1.3 paths; validation-clean rendering and teardown.

### 9.3 Compatibility on qualifying Vulkan 1.2 devices

- [ ] Route the approved descriptor-set compatibility implementation through the shared Vulkan 1.2 backend when its semantic requirements are satisfied.
  - **Depends on:** 9.1, 9.2, and 8.5.
  - **Contract:** `gpu::compat` remains explicitly requested and uses the same device/backend; strict support is reported independently; no automatic mode selection or fallback occurs.
  - **Edges:** compatibility-only device, combined support, insufficient strict heap support, descriptor limits, and optional state rollback.
  - **Verify:** strict-only, compatibility-only, and combined creation/use matrices on qualifying and non-qualifying configurations.

### 9.4 Portability and release evidence

- [ ] Extend CI, hardware records, samples, limitations, and benchmark baselines for the supported Vulkan 1.2 and 1.3 configurations.
  - **Depends on:** 9.2 and 9.3.
  - **Contract:** claims name tested adapter, driver, API version, and enabled native features; absent hardware evidence is documented as unverified rather than supported.
  - **Edges:** platform loader differences, software adapters, unavailable validation layers, and feature-equivalent devices reporting different API versions.
  - **Verify:** CI matrix checks, packaged sample runs, semantic conformance suite, validation and leak checks, and reproducible benchmark records.

## Requirements coverage

| Requirement area | Tasks |
|---|---|
| Runtime-inert imports, discovery, diagnostics, and surfaces | 1.1–1.3 |
| Immutable requests and explicit capability groups | 1.2, 8.1–8.2, 9.1 |
| Multi-device ownership and retryable destruction | 2.1–2.5 |
| Semantic queues, access domains, and cross-queue order | 2.4, 3.1–3.2, 6.1 |
| Transient command lifecycle and compact completion | 3.1–3.4 |
| Presentation readiness and completion | 3.5, 6.5 |
| Allocations, spans, mapping, placement, and lifetime | 4.1–4.7 |
| Strict heaps, shader identity, pipelines, and root ABI | 5.1–5.7 |
| Explicit hazards, transitions, rendering, and work | 6.1–6.5 |
| Public cleanup, documentation, samples, and performance | 7.1–7.4 |
| Additive descriptor-set compatibility | 8.1–8.5 |
| Vulkan 1.2 semantic equivalence | 9.1–9.4 |

## Risk register

- **Token ownership:** compact handles must still reject cross-device misuse before backend mutation. Task 2.3 requires explicit tests before resource migration expands.
- **Import inertness:** recursive C3 imports and platform modules must not initialize native state. Tasks 1.1 and 1.3 pin this with import-only fixtures.
- **Migration ordering:** frame and readback wrappers cannot be removed safely until completion points and allocations exist. Task 4.7 is deliberately after both.
- **Texture transactions:** dedicated allocation is image-first on some backends. Task 4.5 requires fault injection at every private step before publication.
- **Raw GPU identity:** addresses and heap indices cannot carry discoverable ownership for every reference. Tasks 4.6 and 5.2 document caller lifetime while validating every observable token.
- **Compatibility ambiguity:** per-draw descriptor semantics are not approved. Task 8.1 and Gate C prevent placeholder source from becoming accidental policy.
- **Version drift:** Vulkan 1.2 implementation work can distort the strict API if started early. Gate D keeps fallback decisions private and secondary.
