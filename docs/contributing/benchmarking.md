# Benchmarking

Benchmarks are manual evidence, not CI gates. Run the correctness tests
first. Record enough environment detail that someone else can explain the
number.

## Setup

C3 0.8.3, submodules initialized, shader assets built, and the Vulkan and VMA
setup from [testing](testing.md#toolchain).

```sh
python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
```

Record with every result:

- OS and CPU;
- GPU, driver, and Vulkan loader;
- C3 version and build mode;
- contract validation policy and whether Vulkan validation layers are on;
- queue topology, including which roles alias (report `QueueInfo`);
- display environment for presentation targets;
- arguments, warm-up, sample count, and the statistic reported.

## Targets

```sh
c3c build --path test        # lists targets
c3c run <target> --path test
```

Manual targets cover command recording, allocator lifecycle, submission and
retirement, pipeline cache, sparse binding, presentation, and representative
GPU workloads. Headless targets can run on lavapipe:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json c3c run <target> --path test
```

## Method

1. Build the target on the exact revision under test.
2. Warm caches and one-time driver initialization.
3. Report a statistic over repetitions, not a single sample.
4. Change one variable at a time.
5. Keep the correctness result next to the performance result.

Multithreaded recording: compare one worker with one allocator, several
workers with one allocator each, and each case with validation layers off and
on. Validation layers serialize command calls and software drivers dominate
library CPU cost; those are environment results, not contract changes.

Allocator lifecycle: hold per-allocator capacity fixed while varying
create/destroy churn and concurrently live allocators. Report create,
begin/end, discard/retirement, and destroy separately. Churn must not consume
device-lifetime allocator state.

## Reading results

- A lavapipe pipeline-cache blob may be only a header. Use the blob size to
  tell driver behavior from cache plumbing.
- Xvfb has no vblank; FIFO timing there is structural only.
- Compare `FULL` and `TRUSTED` validation separately.
- Never promote a measurement into a public guarantee. The contract is
  qualitative: bounded fixed scratch, independent allocators, no hidden waits.

## Decision records

Investigations that ended in "keep the current design", with the evidence and
the condition that would reopen them:

- [Graphics pipeline identity](pipeline_identity.md): `polygon_mode` and
  `sample_count` stay in `GraphicsPipelineDesc`.
- [Texture-index ranges](texture_index_ranges.md): no contiguous descriptor
  range allocator.
- [Root-pointer data](root_pointer_data.md): no library-owned dynamic-uniform
  path.
- [Shader variants](shader_variants.md): no specialization constants.
