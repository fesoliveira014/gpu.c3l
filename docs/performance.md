# Performance

The repository provides manual benchmarks for investigating broad performance
questions. Their timings, labels, line ordering, and ratios are advisory. CI
builds the executables but does not run them, parse their output, enforce
thresholds, or treat private implementation details as a stable schema.

Compare results only when the compiler, optimization level, adapter, driver,
validation configuration, and queue topology are comparable. Prefer a profiler
and repeated local measurements before drawing conclusions from a single run.

## Prerequisites

Install the native dependencies described in
[Testing](testing.md), initialize the submodules, and build the shaders:

```sh
git submodule update --init --recursive
python3 scripts/build_shaders.py
```

For a headless Linux run, select a Vulkan 1.3 ICD explicitly:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Use a hardware ICD instead when the question concerns real GPU behavior.

## Manual targets

Build or run one target directly with C3 `-O1`:

```sh
c3c run resource_create_bench --path test -O1
c3c run upload_throughput_bench --path test -O1
c3c run command_path_baseline_bench --path test -O1
c3c run lifecycle_bench --path test -O1
c3c run pipeline_cache_bench --path test -O1
c3c run async_overlap_bench --path test -O1
```

The Vulkan targets cover:

- `resource_create_bench`: texture and allocation creation across worker
  counts, CPU-visible allocation/free, texture-view batches, and sampler hits;
- `upload_throughput_bench`: caller-owned uploads across payload sizes and
  worker counts;
- `command_path_baseline_bench`: representative direct Vulkan and public
  command recording, including nonzero dispatch and copy output comparisons;
- `lifecycle_bench`: submission, completed-point polling, destruction, and
  submission batches;
- `pipeline_cache_bench`: pipeline alias creation, duplicate lookup, and
  complete `GraphicsState` packet recording across raster permutations;
- `async_overlap_bench`: serialized and independent graphics/compute work,
  with an explicit not-applicable result on single-queue devices.

The pipeline-cache command timing is an advisory complete-packet measurement.
Raster-only and complete-packet recording costs are not comparable, and the
measurement has no threshold, parsed schema, or CI gate.

The CPU-only command-wrapper baseline needs no Vulkan loader, ICD, VMA, or
native library:

```sh
c3c run command_wrapper_bench --path test/cpu -O1
```

It compares representative public wrapper calls with an observable direct
floor. The operation mix and printed output are deliberately not a stable
schema.

## Interpretation

A benchmark may fail for normal setup or execution errors, incorrect
readback/output, or a small local sanity assertion. A timing change alone is
never a repository correctness failure.

Use the ordinary C3 and Vulkan validation suites for correctness:

```sh
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
c3c test vk_core --path test --test-show-output
c3c test vk_wsi --path test --test-show-output
c3c test vk_optional_generated_work --path test --test-show-output
```

Those tests own behavior such as allocation and handle lifetimes, command
recording and rollback, submission and completion, reflection and shader ABI,
pipeline caching, synchronization, WSI, output/readback, and capability
handling. The manual benchmarks do not define a second correctness protocol.

Timestamp measurements are meaningful only when both writes execute on the
same native queue. Logical graphics, compute, and transfer roles may alias that
queue; values from distinct native queues are not calibrated and must not be
compared. Use the selected role's valid-bit width and `period_ns` through
`timestamp_delta_ns` rather than subtracting raw values directly.

The caller resets each query before reuse, writes every query before resolving
it, and orders host reads after the covering completion point. Command resolve
uses a device-side availability wait and does not block the recording thread,
but an unwritten query can stall the queue. Direct host reads neither wait nor
allocate hidden staging; `DEVICE_BUSY` leaves the requested output unspecified.
`FULL` retains referenced resources but adds no per-slot reset/write history
tracker, so it does not turn timestamp history into a recording-time metric or
diagnostic.
