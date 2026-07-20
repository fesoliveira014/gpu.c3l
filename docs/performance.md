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
| `lifecycle_bench` | Submission, completed-point polling, and immediate texture destruction |
| `pipeline_cache_bench` | Cold creation, cached duplicate lookup, and cached batches |
| `resource_create_bench` | Texture, shader-code, allocation, and mixed creation across 1/2/4 workers |
| `descriptor_churn_bench` | Texture-view publication and sampler hits across 1/2/4 workers |
| `upload_throughput_bench` | Explicit uploads at 4 KiB, 256 KiB, and 4 MiB across 1/2/4 workers |
| `async_overlap_bench` | Serialized and independent graphics/compute submissions |

Each timing target performs its own warmup and fixed repetitions. Generated
dispatch records one untimed command per command list, then measures 1,000
records. The warm command reuses device-pooled preprocess storage; later
records in the list reuse the same compatible buffer.

The runner rejects nonzero hot-path invariants. The CPU-only
`scripts/check_performance_contract.py` gate also rejects registry locking or
pipeline construction in recording entry points, per-point allocation,
destruction waits or deferred releases, and allocation-before-reuse in the
generated preprocess path. Its mutation tests run without a Vulkan ICD.

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
| Cold pipeline creation | 500,000 ns/create |
| Cached duplicate / batch | 20,000 ns/create |

These thresholds catch accidental algorithmic or lifecycle regressions. They
are not cross-machine performance rankings.

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
| Allocation | Allocate / free | 4,000 per phase | 497.3 / 174.3 ns | 487.7–546.3 / 169.8–213.5 ns |
| Command recording | Barrier | 20,000 × 5 | 131.5 ns/record | 130.1–131.7 |
| Command recording | Hazard barrier | 20,000 × 5 | 137.6 ns/record | 136.6–140.4 |
| Command recording | Indirect dispatch | 20,000 × 5 | 179.6 ns/record | 169.9–180.4 |
| Command recording | Generated dispatch | 1 warmup + 1,000 × 5 | 745.6 ns/record | 740.3–755.4 |
| Lifecycle | Submission | 256 × 5 | 8,653.9 ns/submit | 8,641.4–9,190.2 |
| Lifecycle | Completed-point poll | 100,000 × 5 | 42.2 ns/poll | 41.9–42.8 |
| Lifecycle | Texture destruction | 300 × 5 | 244.0 ns/destroy | 240.7–247.3 |
| Pipeline | Cold creation | 200 | 52,304.0 ns/create | 48,971.5–52,474.5 |
| Pipeline | Cached duplicate | 200,000 | 1,063.9 ns/create | 1,055.6–1,069.9 |
| Pipeline | Cached batch | 64 × 2,000 | 2,048.1 ns/create | 2,041.7–2,095.6 |

Every run reported:

```text
invariants: registry_locks=0 recording_allocations=0 draw_compilations=0 preprocess_allocations=0
invariants: point_allocations=0 destruction_waits=0 deferred_releases=0
```

The installed local validation layer was Vulkan 1.3.250 and did not recognize
the newer generated-command and maintenance structures used by the current
driver. It emitted compatibility diagnostics, so no validation-enabled timing
is published from that layer. The separate runner mode is the reproducible
path for collecting debug cost with a matching layer.

## Interpretation

- Reuse caller-owned upload and destination allocations only after their
  covering completion point completes.
- Cache pipelines; cold creation and cached lookup are different workloads.
- Generated command recording reuses preprocess storage after the first
  compatible allocation instead of allocating per command.
- Queue overlap depends on topology, driver scheduling, and workload balance;
  treat it as an observation, not a guarantee.
