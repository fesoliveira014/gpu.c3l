# Performance

Benchmarks are advisory measurements with hard structural guards. Compare
numbers only when the compiler, optimization level, adapter, driver, contract
policy, lifetime-tracking and Vulkan-layer state, and queue topology match.

## Run

Install the native dependencies in [Testing](testing.md), then run:

```sh
python -B scripts/run_benchmarks.py
```

The runner builds every target once with C3 `-O1`, uses trusted/no-tracking/no-
layer defaults for release evidence, validates output schemas and zero-work
fields, and writes `test/build/benchmark-report.md`. The command-recording
target is the exception: the same built executable and fixed workload run in
this required order:

| Mode | Contract | Tracking | Vulkan layers |
|---|---|---:|---:|
| 1 | `TRUSTED` | off | off |
| 2 | `OBJECT_BOUNDARIES` | off | off |
| 3 | `FULL` | on | off |
| 4 | `FULL` | on | on |

Only mode 1 participates in release timing thresholds. All four elapsed times
remain advisory. Whenever the runner is executed locally or on a self-hosted
machine, exact work-counter violations hard-fail the run. Hosted CI builds the
benchmark targets and unit-tests their schemas but does not execute this
four-mode runner; its blocking live equivalent is the `vk_validation_policy`
behavioral target. To collect the former all-enabled debug configuration for
the other benchmark devices in a separate report, run:

```sh
python -B scripts/run_benchmarks.py --validation \
  --output test/build/benchmark-report-validation.md
```

`--validation` selects `FULL`, lifetime tracking on, and Vulkan validation on
for benchmark devices that do not have an explicit policy matrix. It does not
evaluate or report their release timing thresholds, and pinned comparison flags
are rejected. Do not compare those timings with the release baseline. The
validation layer must recognize every enabled Vulkan extension; otherwise its
diagnostics invalidate the timing run. `command_record_bench` always uses its
four explicit modes, even during this separate report.
`command_path_baseline_bench` likewise keeps its fixed trusted, tracking-off,
layers-off policy so its native/public comparison stays on the release contract.

For a direct command-only run, build once and set all three required variables:

```sh
c3c build command_record_bench --path test -O1
GPU_C3L_BENCH_CONTRACT=trusted GPU_C3L_BENCH_TRACKING=false GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=object_boundaries GPU_C3L_BENCH_TRACKING=false GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=full GPU_C3L_BENCH_TRACKING=true GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=full GPU_C3L_BENCH_TRACKING=true GPU_C3L_BENCH_LAYERS=true ./test/build/command_record_bench
```

The executable rejects missing or malformed policy variables. Each output has
one exact `validation policy=...` line reporting semantic checks, tracking
calls, reference allocations/increments/releases, and layer selection. Trusted
and object-boundary modes require every policy-work counter to be zero. Both
full modes require semantic and tracking work, releases must equal increments,
and allocations cannot exceed increments. Every warm interval must report zero
device registry, command-table, pipeline-table/cache, and policy reselection.
The target creates its explicit queue-bound allocator before warmup and outside
every measured interval. Its cold counters report allocator host allocation,
one pool creation, and one complete native command-buffer allocation separately
from warm recording. All four policy modes require zero warm host/native/VMA
allocation; tracking modes use the reference slab allocated at allocator create.

### Command-path baselines

The command-path baseline separates four kinds of evidence so driver cost does
not obscure library work:

1. `command_wrapper_bench` is an ICD-free CPU floor. It compares an observable
   direct no-op with the equivalent public wrapper for dispatch, draw, barrier,
   viewport, and buffer copy.
2. `command_path_baseline_bench` records the same five operations through
   direct Vulkan and public gpu.c3l paths, using separate warmed command lists
   from one device and alternating which path is timed first.
3. Dispatch and buffer-copy equivalence executes both paths into distinct,
   pre-seeded outputs. Each output must match a non-zero expectation and its
   paired output.
4. Full lifecycle cases measure begin, bind, 0/1/16/256 commands, end, submit,
   and successful completion wait. Non-zero cases also report incremental cost
   per command relative to the zero-command case.

The paired operation timers include only warmed command recording. Handles,
native objects, command lists, and other shared prerequisites are resolved or
created before timing. Structural snapshots around every operation loop must
show zero host or pool allocation, command-buffer allocation/reset, image-view
or VMA allocation, shader-module or native-pipeline creation, and device-registry
lock acquisition. Lifecycle cases allow only the exact command-list
allocation/reset work required by their contract and reject unrelated work.

Run the CPU floor without Vulkan dependencies:

```sh
c3c build command_wrapper_bench --path test/cpu -O1
./test/cpu/build/command_wrapper_bench
```

Run the native baseline with the normal Vulkan loader, VMA, SPIRV-Reflect, and
a headless Vulkan 1.3 ICD available:

```sh
python3 scripts/build_shaders.py
c3c build command_path_baseline_bench --path test -O1
VK_DRIVER_FILES=/path/to/icd.json ./test/build/command_path_baseline_bench
```

Both targets emit exact machine-readable records. The runner rejects missing,
duplicated, malformed, out-of-range, or internally inconsistent operation,
work, equivalence, and lifecycle records. Minimum/median/maximum times and
direct/public ratios remain advisory; structural and semantic failures are
blocking.

## Evidence and regression gates

The suite covers:

| Target | Evidence |
|---|---|
| `allocation_bench` | Explicit `CPU_WRITE` allocation and free |
| `command_wrapper_bench` | ICD-free direct no-op and public-wrapper floor for five command classes, with exact volatile observation |
| `command_path_baseline_bench` | Paired direct/public Vulkan recording, zero hidden structural work, dispatch/copy readback equivalence, and 0/1/16/256 full-lifecycle cases |
| `command_record_bench` | Barrier, semantic-hazard barrier, indirect dispatch, and generated dispatch recording |
| `lifecycle_bench` | Submission, cached completed-point polling, and immediate texture destruction |
| `submit_batch_bench` | Real submit batches of 1/8/32/128/1,024 lists with exact one-visit-per-list duplicate-detection work |
| `pipeline_cache_bench` | Dynamic raster matrix aliasing, raster-state recording, cached duplicate lookup/batches, and exact 1 KiB/64 KiB/1 MiB shader-identity work |
| `resource_create_bench` | Texture, shader-code, allocation, and mixed creation across 1/2/4 workers |
| `descriptor_churn_bench` | Texture-view publication and sampler hits across 1/2/4 workers; exact sampler lookup probes at occupancy 8/64/1,024/65,536; exact texture/swapchain ownership work at descriptor high-water 16/4,096/65,536 |
| `upload_throughput_bench` | Explicit uploads at 4 KiB, 256 KiB, and 4 MiB across 1/2/4 workers |
| `async_overlap_bench` | Serialized and independent graphics/compute submissions |

Each timing target performs its own warmup and fixed repetitions. Generated
dispatch first reserves 64 allocator-local preprocess buffers, prewarms one
untimed 64-record command list, then measures five 64-record lists. Every
execute call in a live list receives a distinct reserved preprocess address.
Completed or discarded lists return compatible buffers to the same allocator.

Required performance evidence executes complete operations and compares narrow
subsystem snapshots:

| Invariant | Required observation |
|---|---|
| Cold allocator creation | Host allocations, exactly one exact-family command-pool create, one complete native command-buffer allocation call, and configured buffer count |
| Warm begin/bind/dispatch/end | `RecordingWorkCounters`, pipeline/shader creation counts, and pre-bind `CommandResolutionStats` |
| Warm render pass | `RecordingWorkCounters`, pipeline/shader creation counts, pre-bind `CommandResolutionStats`, and native command emission |
| Generated dispatch/draw/indexed draw | Per-family `RecordingWorkCounters` emissions plus `CommandRecordingStats` reservation/allocation state |
| Cached completion | `CompletionWorkCounters` across 100,000 polls, cached waits, and concurrent first observers |
| Immediate destruction | `CompletionWorkCounters`, injected native-destroy counts, and stalled-queue ordering |
| Submission ownership | Submitted-batch references, caller tokens, retained-reference counts, and ordered retirement state |
| Bound pipeline snapshot | Exact bind-time table/cache lookups, zero post-bind resolution under cache churn, and unchanged layout use after backing-slot mutation |
| Shader identity | Exact intern probes, collision-byte comparisons, owned clone/free bytes, and zero post-intern shader work at 1 KiB/64 KiB/1 MiB |
| Sampler buckets | Exact collision-chain probes and a zero-probe empty-bucket miss at 65,536 entries |

Warm command-buffer reset is expected reuse evidence. Warm host allocation is
prohibited in every policy mode; tracking modes retain into fixed reference
storage allocated with the command allocator. VMA allocation,
command-buffer allocation/free, image-view creation, pipeline/shader creation,
and registry, retained-pin, lifecycle-vtable, command-table, and policy work
remain prohibited. Binding an opaque pipeline handle performs exactly one
pipeline-table and one pipeline-cache lookup; dispatch and draw perform no
additional resolution and each emits exactly one root push plus its native
execution command. The dispatch invariant mutates the bound handle's backing
slot after binding and observes the pushed layout, proving command recording
uses the bind-time value snapshot. A compile-time exact member-shape guard makes
any hidden slot pointer or index/generation back-reference an explicit test
change.

The lifecycle benchmark first establishes full retirement for each measured
point, then resets completion-work counters before 100,000 repeated polls. The
measured interval requires zero native counter queries and zero retirement-lock
acquisitions; these counters are compiled only for tests and this benchmark
target.

The descriptor-churn benchmark separately varies descriptor high-water state at
16, 4,096, and 65,536. Texture destruction must report exactly one ownership
work unit per texture, and the swapchain seam exactly one per wrapped image
examined, at every level. These exact counters are blocking; `ns/destroy` and
`ns/check` remain advisory because runner, driver, build, and profile identity
are not fully pinned by that single observation.

The same target builds synthetic sampler tables at occupancy 8, 64, 1,024, and
65,536 with the production canonical hash, bucket, link, equality, and lookup
helpers. Every tier requires a power-of-two bucket count at least twice the
occupancy and between one and eight candidate probes for the selected hit.
Every tier also requires zero candidate probes for a guaranteed empty-bucket
miss. The 65,536-entry zero-probe observation is also enforced by the live
`vk_descriptor_heap` test target. Elapsed lookup time is advisory.

The submit-batch benchmark submits real executable command lists in batches of
1, 8, 32, 128, and 1,024. Its feature-gated counters require exactly one token
visit per list and zero epoch-reset cells for each ordinary batch. These exact
work counts are blocking; `ns/submit` remains advisory.

The pipeline-cache benchmark separately interns synthetic 1 KiB, 64 KiB, and
1 MiB identities. For every size, the live `vk_pipeline_cache` test and the
benchmark require one complete owned clone and free, one compact-key probe, and
zero shader probes, byte comparisons, and clones after interning. Collision and
distinct-storage tests require exact intern probes and compared bytes. Boundary
elapsed time is advisory.

Unpinned runs report crossings of these broad order-of-magnitude thresholds as
advisories:

| Measurement | Maximum |
|---|---:|
| Allocation / free | 5,000 ns |
| Barrier / hazard barrier | 2,000 ns/record |
| Indirect dispatch | 3,000 ns/record |
| Generated dispatch | 20,000 ns/record |
| Submission | 100,000 ns/submit |
| Completed-point poll | 1,000 ns/poll |
| Texture destruction | 10,000 ns/destroy |
| Raster-matrix pipeline alias creation | 500,000 ns/create |
| Cached duplicate / batch | 20,000 ns/create |

These thresholds flag observations for investigation; they are not
cross-machine acceptance criteria. To make them blocking, identify all three
comparison inputs explicitly:

```sh
python -B scripts/run_benchmarks.py \
  --pinned-runner windows-2022-rtx4090 \
  --pinned-driver nvidia-2417000448 \
  --comparison-profile release-o1-2026-07
```

The report records the runner, driver, profile, and whether timing is advisory
or blocking. Supplying only part of the pinned identity, or combining a pinned
identity with `--validation`, is rejected.

### Pipeline identity snapshot

Pipeline bind performs the only generation-checked pipeline-table lookup and
the only pipeline-cache resolution for a command interval. Lifetime-tracking
modes retain the pipeline through discard or completion; later direct, indirect,
generated, and render-pass commands read only the cached native snapshot and
kind/render metadata. Exact bind-time counters followed by a reset after
pipeline-table/cache churn require zero post-bind resolution during dispatch.

An advisory llvmpipe run on 2026-07-21 (Mesa 25.0.7, LLVM 15.0.7) requested
200 dynamic raster states for one immutable graphics descriptor:

| Measurement | Current result |
|---|---:|
| Requested dynamic raster states | 200 |
| Native graphics pipeline creates | 1 |
| Live cache entries / aliases | 1 / 200 |
| Matrix create time | 5,351.7 ns/create |
| Dynamic raster recording | 99.7 ns/state |
| Duplicate lookup | 3,438.5 ns/create |
| Cached batch | 6,838.8 ns/create |

The timings are observations, not acceptance thresholds. The stable contract
is the one-native-pipeline accounting, which the benchmark asserts and the
runner parses.

## Windows baseline

Three complete historical release-policy suite runs were recorded on 2026-07-20:

```text
host: Windows 11 build 26200
compiler: C3 0.8.0_2
optimization: -O1
adapter: NVIDIA GeForce RTX 4090, Vulkan api_version=4210991
driver: NVIDIA, id=4, version=2417000448
validation: contract=TRUSTED tracking=false layers=false
queues: graphics=0:0 compute=0:1 transfer=1:0
```

The table reports the median of the three target medians and their full range.

| Area | Case | Fixed method | Median | Range |
|---|---|---:|---:|---:|
| Allocation | Allocate / free | 4,000 per phase | 519.1 / 183.8 ns | 487.6–609.5 / 172.7–188.7 ns |
| Command recording | Barrier | 20,000 × 5 | 131.4 ns/record | 131.3–136.6 |
| Command recording | Hazard barrier | 20,000 × 5 | 136.8 ns/record | 135.2–144.0 |
| Command recording | Indirect dispatch | 20,000 × 5 | 180.3 ns/record | 179.6–189.4 |
| Lifecycle | Submission | 256 × 5 | 8,840.6 ns/submit | 8,102.7–11,106.2 |
| Lifecycle | Completed-point poll | 100,000 × 5 | 41.7 ns/poll | 41.7–41.9 |
| Lifecycle | Texture destruction | 300 × 5 | 241.7 ns/destroy | 240.3–247.0 |
| Pipeline | Cold creation | 200 | 49,669.0 ns/create | 49,105.5–51,123.0 |
| Pipeline | Cached duplicate | 200,000 | 1,062.8 ns/create | 1,056.5–1,063.3 |
| Pipeline | Cached batch | 64 × 2,000 | 2,051.9 ns/create | 2,044.4–2,065.9 |

Every run reported:

```text
invariants: point_allocations=0 destruction_queries=0 destruction_completion_waits=0 cached_poll_queries=0 retirement_locks=0
```

Generated-dispatch timing is reported only for runs using an explicit
64-buffer reservation on the measured command allocator. A release run is accepted only when it also reports zero
warm host allocations, command-buffer allocations/frees, image-view creations,
VMA allocations, and generated-scratch misses. Command-buffer resets are
expected reuse evidence. The runner publishes cold and warm work-counter lines
and at least 320 generated preprocess reuse events for the measured lists.

The installed local Vulkan validation layer was Vulkan 1.3.250 and did not recognize
the newer generated-command and maintenance structures used by the current
driver. It emitted compatibility diagnostics, so no layer-enabled timing is
published from that layer. The separate runner mode is the reproducible path
for collecting debug cost with a matching layer.

## Interpretation

- Reuse caller-owned upload and destination allocations only after their
  covering completion point completes.
- Cache pipelines; command-time topology, cull, front-face, and depth-bias
  permutations should reuse one immutable pipeline and change state with
  `cmd_set_raster_state`.
- Create one allocator per concurrently recording worker and exact queue before
  timing. Allocator creation is cold setup; `DEVICE_BUSY` means its fixed command
  buffer count is already live, while `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`
  means a fixed scratch or reservation ceiling must be enlarged.
- Reserve generated scratch per allocator and public pipeline handle before
  timing. Aliases of one native pipeline have distinct reservation keys.
  Generated command recording assigns a unique reserved preprocess address to
  every execute call and returns it to the same allocator only after the owning
  command list is discarded or completed.
- Queue overlap depends on topology, driver scheduling, and workload balance;
  treat it as an observation, not a guarantee. For cross-queue dependencies,
  choose the earliest real consumer in `CompletionWait.before`; an unnecessarily
  broad `.all` mask can reduce available overlap. Indirect and generated command
  argument consumption has no dedicated public wait-stage bit, so waits for
  GPU-produced draw/dispatch arguments must use `.all`.
