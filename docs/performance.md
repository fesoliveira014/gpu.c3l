# Performance

Benchmarks are advisory measurements with hard structural guards. Compare
numbers only when the compiler, optimization level, adapter, driver, validation
state, and queue topology match.

## Run

Install the native dependencies in [Testing](testing.md), then run:

```sh
python -B scripts/run_benchmarks.py
```

The runner builds every target with C3 `-O1`, disables validation and implicit
layers, validates the output schemas and release thresholds, and writes
`test/build/benchmark-report.md`. Use a separate report for validation cost:

```sh
python -B scripts/run_benchmarks.py --validation \
  --output test/build/benchmark-report-validation.md
```

Validation mode enables `RuntimeDesc.enable_validation` for every benchmark
device and skips release-performance thresholds. Do not compare its timings
with the release baseline. The validation layer must recognize every enabled
Vulkan extension; otherwise its diagnostics invalidate the timing run.

## Evidence and regression gates

The suite covers:

| Target | Evidence |
|---|---|
| `allocation_bench` | Explicit `CPU_WRITE` allocation and free |
| `command_record_bench` | Barrier, semantic-hazard barrier, indirect dispatch, and generated dispatch recording |
| `lifecycle_bench` | Submission, cached completed-point polling, and immediate texture destruction |
| `submit_batch_bench` | Real submit batches of 1/8/32/128/1,024 lists with exact one-visit-per-list duplicate-detection work |
| `pipeline_cache_bench` | Dynamic raster matrix aliasing, raster-state recording, cached duplicate lookup, and cached batches |
| `resource_create_bench` | Texture, shader-code, allocation, and mixed creation across 1/2/4 workers |
| `descriptor_churn_bench` | Texture-view publication and sampler hits across 1/2/4 workers; exact texture/swapchain ownership work at descriptor high-water 16/4,096/65,536 |
| `upload_throughput_bench` | Explicit uploads at 4 KiB, 256 KiB, and 4 MiB across 1/2/4 workers |
| `async_overlap_bench` | Serialized and independent graphics/compute submissions |

Each timing target performs its own warmup and fixed repetitions. Generated
dispatch first reserves 64 context-local preprocess buffers, prewarms one
untimed 64-record command list, then measures five 64-record lists. Every
execute call in a live list receives a distinct reserved preprocess address.
Completed or discarded lists return compatible buffers to the same recording
context.

The runner rejects nonzero hot-path invariants. The CPU-only
`scripts/check_performance_contract.py` gate also rejects registry locking or
pipeline construction in recording entry points, per-point allocation,
destruction waits or deferred releases, allocation-before-reuse in the
generated preprocess path, retired compute-layout caches, and dynamic raster
fields in immutable pipeline keys. It walks the reachable Vulkan recording
graph and rejects host allocation, native command-buffer allocation/free,
image-view creation, and VMA allocation outside named cold seams. Its mutation
tests run without a Vulkan ICD.

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

The submit-batch benchmark submits real executable command lists in batches of
1, 8, 32, 128, and 1,024. Its feature-gated counters require exactly one token
visit per list and zero epoch-reset cells for each ordinary batch. These exact
work counts are blocking; `ns/submit` remains advisory.

Release runs use deliberately broad order-of-magnitude thresholds:

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

These thresholds catch accidental algorithmic or lifecycle regressions. They
are not cross-machine performance rankings.

### Pipeline identity snapshot

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

Three complete validation-disabled suite runs were recorded on 2026-07-20:

```text
host: Windows 11 build 26200
compiler: C3 0.8.0_2
optimization: -O1
adapter: NVIDIA GeForce RTX 4090, Vulkan api_version=4210991
driver: NVIDIA, id=4, version=2417000448
validation: disabled
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
invariants: point_allocations=0 destruction_waits=0 deferred_releases=0 cached_poll_queries=0 retirement_locks=0
```

Generated-dispatch timing is reported only for runs using an explicit
64-buffer reservation. A release run is accepted only when it also reports zero
warm host allocations, command-buffer allocations/frees, image-view creations,
VMA allocations, and generated-scratch misses. Command-buffer resets are
expected reuse evidence. The runner publishes cold and warm work-counter lines
and at least 320 generated preprocess reuse events for the measured lists.

The installed local validation layer was Vulkan 1.3.250 and did not recognize
the newer generated-command and maintenance structures used by the current
driver. It emitted compatibility diagnostics, so no validation-enabled timing
is published from that layer. The separate runner mode is the reproducible
path for collecting debug cost with a matching layer.

## Interpretation

- Reuse caller-owned upload and destination allocations only after their
  covering completion point completes.
- Cache pipelines; command-time topology, cull, front-face, and depth-bias
  permutations should reuse one immutable pipeline and change state with
  `cmd_set_raster_state`.
- Reserve generated scratch per worker and public pipeline handle before
  timing. Aliases of one native pipeline have distinct reservation keys.
  Generated command recording assigns a unique reserved preprocess address to
  every execute call and returns it to the same context only after the owning
  command list is discarded or completed.
- Queue overlap depends on topology, driver scheduling, and workload balance;
  treat it as an observation, not a guarantee. For cross-queue dependencies,
  choose the earliest real consumer in `CompletionWait.before`; an unnecessarily
  broad `.all` mask can reduce available overlap. Indirect and generated command
  argument consumption has no dedicated public wait-stage bit, so waits for
  GPU-produced draw/dispatch arguments must use `.all`.
