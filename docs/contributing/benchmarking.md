# Benchmarking

Benchmarks are manual evidence, not CI pass/fail gates. Run correctness tests
first and record enough environment context for another contributor to explain
the result.

## Prerequisites

Use C3 0.8.3, initialize submodules, build generated shader assets, and satisfy
the Vulkan/VMA setup in [Architecture](../architecture.md#platform-and-dependency-boundary).

```sh
python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
```

For native measurements, record:

- operating system and CPU;
- GPU, driver, and Vulkan loader;
- C3 compiler version and build mode;
- validation policy and whether Vulkan validation layers are enabled;
- queue topology and whether semantic roles alias;
- window/display environment for presentation tests; and
- benchmark arguments, warm-up, sample count, and reported statistic.

## Targets

List current benchmark targets from the repository configuration:

```sh
c3c build --path test
```

The maintained manual targets cover command recording, allocator lifecycle,
submission/retirement, pipeline cache behavior, sparse binding, presentation,
and representative GPU workloads. Run a target by name:

```sh
c3c run <target> --path test
```

Some targets require a display; headless GPU targets can use a deterministic
software Vulkan driver:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  c3c run <target> --path test
```

## Method

1. Verify the target on the exact revision.
2. Warm caches and one-time driver initialization.
3. Run enough repetitions to report a robust statistic, not one sample.
4. Compare one changed variable at a time.
5. Keep correctness validation results alongside performance results.

For multithreaded recording, compare:

- one worker/allocator;
- multiple workers with one allocator each; and
- the same cases with Vulkan validation layers disabled and enabled.

Validation layers commonly serialize native command calls, and software
drivers may dominate CPU-side library costs. Those are environment results,
not evidence that the public concurrency contract changed.

For allocator lifecycle scaling, hold the per-allocator capacity constant while
varying historical create/destroy churn and concurrently live allocators.
Report create, begin/end, discard/retirement, and destroy separately where the
target exposes them. Historical churn must not consume a device-lifetime
allocator context.

## Interpreting results

- A lavapipe pipeline-cache blob may contain only a header; use blob size to
  distinguish driver behavior from cache plumbing.
- Xvfb has no physical vblank, so FIFO presentation timing there is structural
  only.
- Full contract validation performs bounded retained-resource bookkeeping;
  compare it separately from `TRUSTED`.
- Queue-role labels do not imply distinct hardware queues; report actual
  `QueueInfo`.
- Never promote a benchmark observation into a public guarantee. The stable
  contract is qualitative: bounded fixed scratch, independent allocators, and
  no hidden waits or per-call application policy.
- [Root-pointer data vs a fixed dynamic-uniform path](root_pointer_data.md)
  records the per-command data-path comparison, its derived byte and native-work
  costs, and the measurements that still need the primary development GPU.
