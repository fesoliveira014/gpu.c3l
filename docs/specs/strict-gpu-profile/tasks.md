# Strict GPU Profile Tasks

Source documents:

- [Requirements](requirements.md)
- [Design](design.md)
- [Architecture overview](../../strict_gpu_profile.md)

Tasks stay unchecked until their verification step passes. Milestone gates are blocking except task 2.5: do not begin compatibility extraction before Milestone 1, or strict implementation before tasks 2.1–2.4 and 2.6.

## Milestones

| Milestone | Outcome | Depends on |
|---|---|---|
| 1. Current architecture | Stable, documented, tested, and benchmarked current API. | — |
| 2. Compatibility profile | Current Vulkan 1.3 behavior isolated under `gpu::compat`. | 1 |
| 3. Strict boundary | Strict types, capabilities, and device lifetime. | 2.1–2.4, 2.6 |
| 4. Memory placement | Independent allocations and placed resources. | 3 |
| 5. Descriptor heaps | Raw contiguous shader indices with separate ownership. | 4 |
| 6. Synchronization | Resource-agnostic stage and hazard barriers. | 4, 5 |
| 7. Pipelines | Minimal compiled-state identity. | 3 |
| 8. GPU-driven work | GPU-generated work and root data. | 4, 5, 7 |
| 9. Raster and presentation | Strict graphics and window presentation. | 4–8 |
| 10. Consumer migration | Strict samples, docs, tooling, and performance evidence. | 9 |
| 11. Final strict surface | Transitional APIs removed; release gates pass. | 10 |

Vulkan 1.2 compatibility (task 2.5) is a secondary, parallel follow-up. It does not block strict work after the compatibility boundary and Vulkan 1.3 contract are complete.

## Milestone 1 — Current architecture

Delivery slices:

- **1A — Guidance:** task 1.1.
- **1B — Opaque state:** task 1.2, stacked on 1A.
- **1C — Validation and contract:** tasks 1.3–1.4, stacked on 1B.
- **1D — Baselines:** tasks 1.5–1.7, stacked on 1C.

- [x] **1.1 Reconcile repository guidance with the implemented library.**
  - Targets: `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/architecture.md`, `docs/document_index.md`.
  - Replace stale stub, missing-file, and source-layout claims with the actual public/backend/test structure. Keep C3 0.8.0 and the current one-device limit explicit.
  - Verify: all documentation links resolve; searches for removed paths and stub claims return no matches.

- [x] **1.2 Make backend dispatch and readback bookkeeping private.**
  - Targets: `gpu/device.c3`, `gpu/memory.c3`, `gpu/gpu.c3`, `gpu/vk/device.c3`, backend call sites, `test/src/test_vk_*`, doc generation checks.
  - Replace public dispatch tables, backend pointers, Vulkan probe helpers, and readback retirement fields with opaque or generation-checked public tokens. Preserve command, frame, and readback alias behavior.
  - Edge cases: stale copies, failed readback resolution, teardown with outstanding tickets, invalid device tokens, and the one-live-device contract.
  - Verify: device/readback tests pass; generated `gpu` documentation contains no dispatch aliases, backend pointers, retirement fields, `vk::`, or `vma::` symbols.

- [x] **1.3 Validate workload dimensions against selected-device limits.**
  - Targets: `gpu/caps.c3`, `gpu/command.c3`, `gpu/vk/device.c3`, `gpu/vk/command.c3`, `gpu/vk/render_pass.c3`, `test/src/test_vk_command_validation.c3`, `docs/api.md`, `docs/limitations.md`.
  - Capture compute work-group count and indirect draw-count limits, expose backend-neutral values, and reject over-limit calls before recording backend commands.
  - Edge cases: exact-limit success, zero dimensions, overflow while computing required argument bytes, and count-buffer variants.
  - Verify: pure validation tests cover exact and over-limit values without requiring adapter-specific constants.

- [x] **1.4 Audit the current public contract for internal leaks and lifecycle drift.**
  - Targets: `gpu/*.c3`, `gpu/*.c3i`, `gpu/vk/*.c3`, `docs/api.md`, `docs/memory.md`, `docs/threading.md`, `docs/vulkan_backend.md`.
  - Align ownership, faults, state transitions, and documentation for devices, frames, commands, resources, descriptors, semaphores, swapchains, and tickets. Fix current correctness defects; do not introduce strict placement, raw heaps, or resource-free barriers yet.
  - Verify: public symbol scan is backend-neutral; documented state machines match tests and implementation.

- [x] **1.5 Establish the blocking library baseline.**
  - Targets: `scripts/gen_abi.py`, `scripts/build_shaders.py`, `test/project.json`, `test/cpu/project.json`, `.github/workflows/ci.yml`, `docs/testing.md`.
  - Run generator drift checks, pure CPU tests, shader ABI tests, smoke linkage, and every documented headless Vulkan target on Linux and Windows.
  - Edge cases: missing validation layers, software ICD selection, Windows VMA linkage, and optional descriptor-buffer coverage.
  - Verify: blocking CI is green; advisory hardware gaps are reported as not exercised rather than passed.

- [x] **1.6 Establish the compatibility sample baseline.**
  - Targets: `gpu.c3l-samples/project.json`, every sample directory, sample scripts, sample README files, `docs/samples.md`.
  - Build every sample, run all headless samples, and smoke windowed samples through bounded frame counts. Record required capabilities per sample.
  - Edge cases: resize, minimization, surface loss, unavailable optional features, and clean teardown.
  - Verify: the sample index matches project targets and each runnable target has a deterministic smoke command.

- [x] **1.7 Record performance baselines and close the stabilization gate.**
  - Targets: existing `test/src/*bench.c3`, a small benchmark runner under `scripts/`, `docs/performance.md`.
  - Record allocation, resource creation, descriptor churn, upload, command reset/recording, submission, pipeline creation, and async overlap using a fixed methodology. Add barrier and indirect-recording measurements where absent.
  - Verify: results identify adapter, driver, validation state, queue topology, iteration count, and units; the architecture review has no unresolved correctness blocker.

## Milestone 2 — Compatibility profile

- [x] **2.1 Move the stabilized current implementation into `gpu::compat`.**
  - Targets: `gpu/*.c3`, `gpu/*.c3i`, `gpu/vk/*.c3`, new `gpu/compat/` and `gpu/compat/vk/`, `manifest.json`.
  - Relocate rather than copy the current implementation. Change module declarations and imports while preserving public names and behavior inside `gpu::compat`.
  - Edge cases: explicit manifest source lists, recursive imports, backend-private visibility, and vendored dependency resolution.
  - Verify: the compatibility package compiles with no duplicated implementation files.

- [x] **2.2 Make compatibility initialization explicit and inert on import.**
  - Targets: `gpu/compat/compat.c3i`, compatibility device/bootstrap files, import-only test fixtures.
  - Provide `gpu::compat::create_device`; ensure no global initializer creates instances, allocators, loaders, descriptors, or threads.
  - Verify: an import-only executable exits without Vulkan loader calls or compatibility state; device creation occurs only through the explicit function.

- [x] **2.3 Enforce profile type separation at compile time.**
  - Targets: strict root declarations, `gpu/compat/types.c3`, compile-pass and compile-fail fixtures under `test/`.
  - Declare distinct compatibility devices, handles, commands, barriers, descriptors, and pipelines. Share only value types with identical semantics.
  - Verify: cross-profile calls fail compilation; same-profile calls compile; no alias bridges exist.

- [ ] **2.4 Move existing tests and samples to compatibility.**
  - Targets: new `test/compat/project.json`, existing test sources, `gpu.c3l-samples/project.json`, current sample sources and shader ABI modules.
  - Convert current imports and qualified names mechanically. Do not redesign the samples during this move.
  - Verify: Milestone 1 test, sample, and benchmark results remain behaviorally equivalent through `gpu::compat`.

- [ ] **2.5 Add Vulkan 1.2 plus required extensions to compatibility (deferrable).**
  - Targets: `gpu/compat/vk/instance.c3`, `device.c3`, command loading, capability building, `lib/vk.c3l/commands.c3`, compatibility bootstrap tests, `docs/platforms_and_dependencies.md`.
  - Accept Vulkan 1.2 only when every required 1.3 semantic is available through core or extension entry points. Load dynamic rendering, synchronization2, maintenance4, timeline semaphore, and buffer-device-address variants explicitly.
  - Edge cases: mixed core/extension support, missing suffixed commands, headless operation, presentation extension selection, and deterministic missing-feature faults.
  - Verify: Vulkan 1.3 and 1.2+extensions devices pass the same compatibility smoke contract; missing semantics fail at creation.
  - Gate: this task does not block Milestone 3 or later strict milestones.

- [x] **2.6 Document and freeze the compatibility contract.**
  - Targets: `docs/compatibility.md`, `docs/api.md`, `docs/limitations.md`, `docs/getting_started.md`, generated compatibility reference.
  - Document migration from the former root API, supported Vulkan paths, explicit layouts, descriptor modes, and portability limits.
  - Verify: the compatibility reference matches the stabilized baseline and contains no claim that applies to strict `gpu`.

## Milestone 3 — Strict boundary

- [ ] **3.1 Define strict public types, capabilities, and faults.**
  - Targets: root `gpu/types.c3`, `faults.c3`, `caps.c3`, `device.c3`, `gpu.c3i`.
  - Add opaque device ownership, semantic capabilities, strict limits, and specific faults. Exclude backend identity, descriptor modes, layouts, queue families, and compatibility objects.
  - Verify: public documentation contains only strict semantics and the zero-initialized creation contract is unambiguous.

- [ ] **3.2 Extend the Vulkan binding for strict feature probing.**
  - Targets: `lib/vk.c3l/vk.c3`, `commands.c3`, `builders_ext.c3`, binding layout/size probes.
  - Bind only the required portions of device-address commands, descriptor heaps, unified image layouts, dynamic state, and device-generated commands. Keep C prefixes only in `@cname` strings.
  - Verify: binding declarations compile on C3 0.8.0 and exposed C layouts match Vulkan headers.

- [ ] **3.3 Implement strict device creation and destruction.**
  - Targets: `gpu/vk/instance.c3`, `gpu/vk/device.c3`, `gpu/device.c3`, strict bootstrap tests.
  - Probe the complete required semantic profile before creation. Enable strict features atomically and return a deterministic fault when any requirement is absent.
  - Edge cases: presentation disabled, validation unavailable, partial extension support, cleanup after partial creation, and no compatibility fallback.
  - Verify: supported hardware creates/destroys cleanly; unsupported hardware faults without compatibility initialization.

- [ ] **3.4 Split strict and compatibility verification lanes.**
  - Targets: `test/strict/project.json`, `test/compat/project.json`, `.github/workflows/ci.yml`, docgen leakage checks.
  - Build and test each profile independently while retaining import-only and compile-fail profile-boundary fixtures.
  - Verify: one profile can fail feature probing without preventing the other profile's build and CPU tests.

## Milestone 4 — Memory placement

- [ ] **4.1 Define strict allocation and placement contracts.**
  - Targets: `gpu/memory.c3`, `gpu/buffer.c3`, `gpu/texture.c3`, `docs/memory.md`, `docs/api.md`.
  - Specify allocation ownership, CPU/GPU address availability, alignment, placed resource lifetime, mapping, flush/invalidate, and relocation restrictions.
  - Verify: each ownership transfer and destruction order has one documented outcome and fault.

- [ ] **4.2 Implement independent allocation and release.**
  - Targets: `gpu/vk/allocator.c3`, `gpu/vk/memory.c3`, allocation slot tables, strict memory tests.
  - Allocate host-visible, GPU-private, and readback memory without creating a buffer or texture. Keep VMA internal and prevent implicit resource ownership.
  - Edge cases: zero size, invalid alignment, memory-type exhaustion, non-mappable memory, stale handles, and deferred GPU use.
  - Verify: allocation lifecycle tests pass without resource creation.

- [ ] **4.3 Query texture requirements before creation.**
  - Targets: `gpu/texture.c3`, `gpu/vk/texture.c3`, maintenance4 command loading, texture capability tests.
  - Return size, alignment, and compatibility information from a descriptor without creating a persistent image object.
  - Verify: queried requirements accept valid placement and reject undersized, misaligned, or incompatible placement before mutation.

- [ ] **4.4 Implement placed buffers and textures.**
  - Targets: strict buffer/texture public files, `gpu/vk/buffer.c3`, `gpu/vk/texture.c3`, validation and deferred destruction.
  - Create resource interpretation over caller-owned placement. Destroying the resource must not release the allocation.
  - Edge cases: overlapping placement policy, multiple views, destruction while in flight, allocation release with live resources, and address overflow.
  - Verify: ownership tests cover every valid and invalid destruction order.

- [ ] **4.5 Rebuild arenas on allocation primitives.**
  - Targets: `gpu/memory.c3`, `gpu/vk/memory.c3`, `gpu/vk/transfer.c3`, frame/persistent/staging/readback tests.
  - Preserve checked ranges, frame generations, deferred release, and dedicated fallbacks while removing hidden resource allocation from the strict core.
  - Verify: existing arena semantics pass through strict primitives and allocations with escaped addresses never move transparently.

- [ ] **4.6 Add placement samples and benchmarks.**
  - Targets: strict test project, `gpu.c3l-samples` strict memory sample, resource creation and allocation benchmarks.
  - Demonstrate several buffers/textures suballocated from caller-managed memory and compare allocation spikes with the compatibility path.
  - Verify: sample teardown leaves no resource or allocation live.

## Milestone 5 — Descriptor heaps

- [ ] **5.1 Add strict descriptor-heap binding and feature probing.**
  - Targets: `lib/vk.c3l`, `gpu/vk/device.c3`, new strict descriptor backend files.
  - Enable the resource and sampler heap model and every required dependency. Do not use deprecated descriptor-buffer behavior as the strict contract.
  - Verify: incomplete heap support rejects strict device creation.

- [ ] **5.2 Separate CPU ownership from shader-visible indices.**
  - Targets: `gpu/types.c3`, `gpu/descriptor_heap.c3`, shader ABI includes, descriptor tests.
  - Use generation-checked CPU tokens for allocation/free and raw fixed-width indices in GPU data. Remove generation bits from shader-visible values.
  - Verify: stale CPU tokens fault while raw indices remain ABI-stable.

- [ ] **5.3 Implement contiguous range allocation.**
  - Targets: strict descriptor allocator, pure CPU fragmentation tests, limits documentation.
  - Reserve and free contiguous resource or sampler ranges with transactional failure and deferred reuse.
  - Edge cases: zero count, full heap, fragmented capacity, maximum range, rollback, and retirement across frames.
  - Verify: arbitrary churn never returns overlapping live ranges and base-plus-offset addresses exactly the reserved entries.

- [ ] **5.4 Implement descriptor writes and retirement.**
  - Targets: `gpu/descriptor_heap.c3`, strict Vulkan descriptor backend, submission/deferred queues.
  - Write resources into owned ranges, validate kinds and bounds, and delay reuse until all referencing submissions complete.
  - Verify: writes are all-or-nothing; destroy/recreate stress cannot expose stale descriptors.

- [ ] **5.5 Update the shader ABI and generator.**
  - Targets: `abi/`, `tools/gen_shader_abi/`, `include/shaders/`, generator tests, sample schemas.
  - Define raw index widths, range-base usage, resource/sampler separation, and generated CPU/GLSL layouts.
  - Verify: generation drift checks and CPU/shader layout assertions pass.

- [ ] **5.6 Add descriptor range samples and benchmarks.**
  - Targets: strict bindless sample, material sample, descriptor churn benchmark.
  - Demonstrate one base index addressing related textures and compare allocation/write/retire cost against compatibility.
  - Verify: strict samples contain no packed-generation masking or compatibility descriptor mode.

## Milestone 6 — Resource-agnostic synchronization

- [ ] **6.1 Finalize the strict stage and hazard vocabulary.**
  - Targets: `gpu/sync.c3`, `docs/api.md`, `docs/strict_gpu_profile.md`, pure validation tests.
  - Cover host, transfer, shader, descriptor, indirect, color, depth, and presentation hazards without resource or layout fields.
  - Verify: every strict command use has an expressible before/after hazard pair.

- [ ] **6.2 Enable unified image layouts in the strict backend.**
  - Targets: `lib/vk.c3l`, strict device feature chain, `gpu/vk/sync.c3`.
  - Require unified layout support and keep extension details private.
  - Verify: missing support rejects strict creation; compatibility continues to use explicit layouts.

- [ ] **6.3 Replace strict resource barriers with global hazards.**
  - Targets: `gpu/command.c3`, `gpu/sync.c3`, `gpu/vk/sync.c3`, strict command state.
  - Record one resource-agnostic barrier operation and remove strict pending-layout tables and resource barrier lists.
  - Edge cases: no-op hazards, write-after-write, descriptor updates, indirect arguments, and host visibility.
  - Verify: strict public barrier structures contain no handles, ranges, layouts, or queue families.

- [ ] **6.4 Keep resource-specific image state inside owning operations.**
  - Targets: strict transfer, render, swapchain, and texture backend files.
  - Make copies, clears, rendering, and presentation carry enough private information for compression and presentation handling without public transitions.
  - Verify: render-pass boundaries add no implicit hazard and presentation state never enters public types.

- [ ] **6.5 Add synchronization tests, samples, and benchmarks.**
  - Targets: strict command tests, compute/graphics samples, barrier-recording benchmark.
  - Cover transfer-to-shader, compute-to-indirect, descriptor update, color/depth reuse, host readback, and presentation.
  - Verify: validation-clean output and measurable removal of strict layout-tracking overhead.

## Milestone 7 — Minimal pipeline identity

- [ ] **7.1 Classify every pipeline field.**
  - Targets: `gpu/pipeline.c3`, strict pipeline design notes in `docs/api.md`, current pipeline cache tests.
  - Mark fields as compiled, specialized, or dynamic for compute and graphics. Compatibility classifications remain unchanged.
  - Verify: no field appears in more than one class and every cache-key field has a compiled-code rationale.

- [ ] **7.2 Enable required dynamic state.**
  - Targets: `lib/vk.c3l`, strict feature probing, `gpu/vk/pipeline_graphics.c3`, render command backend.
  - Bind and enable the dynamic-state commands used by strict graphics.
  - Verify: unsupported required state rejects strict creation rather than expanding the public key silently.

- [ ] **7.3 Add shader specialization data.**
  - Targets: `gpu/pipeline.c3`, `gpu/vk/pipeline_compute.c3`, `gpu/vk/pipeline_graphics.c3`, shader reflection/validation tests.
  - Accept deterministic typed or byte-oriented specialization input and include it in compiled pipeline identity.
  - Edge cases: duplicate IDs, size/alignment mismatch, empty data, and stable hashing.
  - Verify: identical specialization aliases; different specialization does not.

- [ ] **7.4 Reduce pipeline keys and add dynamic commands.**
  - Targets: strict pipeline cache, graphics command API, command-state tracking.
  - Remove dynamic and specialized values from public creation descriptors and backend keys; record them at command time.
  - Verify: dynamic-only differences create no new backend pipeline and command state survives pipeline binds as documented.

- [ ] **7.5 Add permutation tests and benchmarks.**
  - Targets: strict pipeline tests, pipeline cache benchmark, representative graphics sample.
  - Measure handle count, backend pipeline count, cache hits, and creation time over representative state combinations.
  - Verify: the strict path demonstrably reduces compiled pipeline permutations.

## Milestone 8 — GPU-driven work

- [ ] **8.1 Add address-command and generated-command bindings.**
  - Targets: `lib/vk.c3l`, strict device feature/property chains, command loaders.
  - Bind only the address-based copy/draw/dispatch and generated-command structures and functions used by strict `gpu`.
  - Verify: binding size/layout checks pass and feature probing distinguishes required from optional generated work.

- [ ] **8.2 Define direct and indirect root-record ABI.**
  - Targets: `abi/`, `tools/gen_shader_abi/`, `gpu/shader_abi.c3`, shader includes and ABI tests.
  - Pin root address width, per-command record stride, draw/dispatch argument layout, and alignment.
  - Verify: CPU and shader layouts match for direct, indexed, counted, and dispatch records.

- [ ] **8.3 Implement address-based indirect commands.**
  - Targets: `gpu/command.c3`, strict Vulkan command/render files, range validation tests.
  - Consume GPU addresses and checked spans without public buffer binding objects.
  - Edge cases: alignment, count overflow, out-of-range records, zero count, and unsupported indexed formats.
  - Verify: GPU-written argument streams execute without CPU patching.

- [ ] **8.4 Implement GPU-generated per-command roots.**
  - Targets: strict generated-command backend, pipeline integration, semantic caps.
  - Allow the GPU to generate work arguments and the root data consumed by each work item. Do not fall back to draw identifiers or a CPU loop.
  - Verify: unsupported devices report the capability as absent and the operation faults deterministically.

- [ ] **8.5 Add culling/compaction samples and benchmarks.**
  - Targets: strict GPU-driven sample, indirect tests, generated-command benchmark.
  - Generate, compact, reorder, and execute work plus roots entirely on the GPU.
  - Verify: shaders contain no compatibility draw-ID root lookup and readback confirms expected work.

## Milestone 9 — Rasterization and presentation

- [ ] **9.1 Use placed textures for strict attachments.**
  - Targets: strict texture and render APIs, `gpu/vk/render_pass.c3`, depth/color tests.
  - Select caller-owned placed textures as color and depth targets without transferring allocation ownership.
  - Verify: attachment destruction and allocation release obey independent lifetimes.

- [ ] **9.2 Complete strict render command state.**
  - Targets: strict render descriptors, viewport/scissor/dynamic state, command validation.
  - Keep attachment selection and draw state explicit while leaving synchronization outside pass boundaries.
  - Edge cases: attachment compatibility, extent mismatch, depth-only passes, empty passes, and invalid nesting.
  - Verify: pass validation faults before backend mutation.

- [ ] **9.3 Integrate strict swapchain images without public layouts.**
  - Targets: strict swapchain API/backend, WSI diagnostics, presentation tests.
  - Expose acquired strict textures and runtime surface information while keeping swapchain handles, image layouts, and recreation state private.
  - Verify: acquire/render/present needs no public texture transition.

- [ ] **9.4 Preserve robust WSI lifecycle behavior.**
  - Targets: swapchain recreation, resize, minimization, surface-loss, and device-loss paths.
  - Retain deterministic retry and teardown behavior without crossing profile state.
  - Verify: bounded windowed tests cover normal presentation and every recoverable WSI outcome.

- [ ] **9.5 Add strict raster and presentation samples.**
  - Targets: strict offscreen triangle, windowed triangle, depth, and resize samples.
  - Use placement, raw heaps, global hazards, minimal pipelines, and strict presentation end to end.
  - Verify: samples import no compatibility module or helper that owns compatibility state.

## Milestone 10 — Consumer migration

- [ ] **10.1 Make strict samples the canonical paths.**
  - Targets: `gpu.c3l-samples/project.json`, sample index, shared helpers, strict sample directories.
  - Provide strict compute, upload/readback, bindless, indirect, raster, and presentation samples. Keep compatibility samples clearly separated.
  - Verify: every strict subsystem has one minimal sample and one representative workload.

- [ ] **10.2 Retain focused compatibility smoke coverage.**
  - Targets: compatibility sample targets and Vulkan 1.2 test tier.
  - Keep only the samples needed to prove stabilized behavior and portability; avoid duplicating the full strict catalog.
  - Verify: compatibility samples exercise both Vulkan 1.3 and 1.2+extensions where hardware is available.

- [ ] **10.3 Rewrite user documentation around strict usage.**
  - Targets: `README.md`, `docs/getting_started.md`, `docs/cookbook.md`, `docs/api.md`, `docs/memory.md`, `docs/shader_abi.md`, `docs/limitations.md`, `docs/samples.md`.
  - Make strict setup and workflows primary. Link compatibility migration and limits without mixing profile semantics.
  - Verify: commands in the getting-started guide execute in CI and all sample links resolve.

- [ ] **10.4 Package strict tooling and documentation checks.**
  - Targets: ABI/shader tooling, CI workflows, generated API reference, backend-leak scanner.
  - Give external consumers stable generation, shader compilation, drift checking, and doc commands.
  - Verify: an external fixture uses the same supported entry points as the samples on Windows and Linux.

- [ ] **10.5 Publish the before/after performance assessment.**
  - Targets: `docs/performance.md`, benchmark runner, CI artifacts.
  - Compare stabilized compatibility with strict allocation, descriptor, barrier, command, pipeline, and indirect workloads under matched conditions.
  - Verify: claims identify the workload and hardware and distinguish CPU overhead, GPU time, and unsupported paths.

## Milestone 11 — Final strict surface

- [ ] **11.1 Remove transitional strict APIs and obsolete ABI forms.**
  - Targets: root `gpu` sources, shader includes, schemas, tests, samples, deprecation notes.
  - Delete compatibility aliases, resource barriers, layouts, descriptor modes, packed shader generations, and obsolete root records from strict `gpu`.
  - Verify: repository search finds these concepts only in compatibility or backend documentation.

- [ ] **11.2 Audit the final public API and ABI.**
  - Targets: generated strict reference, public symbol scanner, ABI tests, ownership documentation.
  - Confirm backend neutrality, type separation, fault specificity, layout pins, and one-shot lifetime rules.
  - Verify: strict documentation contains no Vulkan types, backend state, compatibility modes, or internal development identifiers.

- [ ] **11.3 Run the complete release matrix.**
  - Targets: strict and compatibility CPU/native tests, all strict samples, compatibility smokes, docs walkthrough, generator drift, leak checks, validation, and benchmarks.
  - Verify: blocking Linux and Windows lanes pass; required real-hardware extension runs have recorded results; teardown is leak-free.

- [ ] **11.4 Close the initiative gate.**
  - Targets: requirements, design, tasks, architecture overview, limitations, and external tracker.
  - Confirm every acceptance check has evidence, close or re-scope superseded compatibility requests, and leave only independently valuable follow-up work.
  - Verify: `gpu` is understandable without learning `gpu::compat`, while compatibility remains usable only through explicit initialization.
