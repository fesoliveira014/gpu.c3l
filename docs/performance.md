# Performance

Benchmarks report advisory timings alongside blocking deterministic allocation,
work, state, fault, and native-emission observations. Compare numbers only when
the compiler, optimization level, adapter, driver, contract policy,
lifetime-tracking and Vulkan-layer state, and queue topology match.

## Run

Install the native dependencies in [Testing](testing.md), then run:

```sh
python -B scripts/run_benchmarks.py
```

The runner builds every target once with C3 `-O1`, uses trusted/no-tracking/no-
layer defaults for release evidence, validates output schemas and zero-work
fields, records hand-maintained `expectation_version=2`, and writes
`test/build/benchmark-report.md`. Command recording is built in both public
token representations. The default bounded `command_record_bench` executable
and fixed workload run in this required order:

| Mode | Contract | Tracking | Vulkan layers |
|---|---|---:|---:|
| 1 | `TRUSTED` | off | off |
| 2 | `OBJECT_BOUNDARIES` | off | off |
| 3 | `FULL` | on | off |
| 4 | `FULL` | on | on |

Only mode 1 participates in release timing thresholds. All four elapsed times
remain advisory. `command_record_fast_bench` is compiled with
`DIRECT_COMMAND_TOKENS` and runs only trusted/no-tracking/no-layer because a
direct-token build cannot provide `FULL` token diagnostics. Whenever the runner
is executed locally or on a self-hosted machine, exact work-counter violations
hard-fail the run. Hosted CI builds the benchmark targets and unit-tests their
schemas but does not execute this live runner; its blocking equivalents are
`vk_validation_policy` for the bounded matrix and `vk_command_tokens_fast` for
the direct representation. To collect the former all-enabled debug
configuration for the other benchmark devices in a separate report, run:

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
`command_record_fast_bench` always uses its single trusted direct-token mode.
`command_path_baseline_bench` likewise keeps its fixed trusted, tracking-off,
layers-off policy so its native/public comparison stays on the release contract.

For the bounded command-only matrix, build once and set all three required
variables:

```sh
c3c build command_record_bench --path test -O1
GPU_C3L_BENCH_CONTRACT=trusted GPU_C3L_BENCH_TRACKING=false GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=object_boundaries GPU_C3L_BENCH_TRACKING=false GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=full GPU_C3L_BENCH_TRACKING=true GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_bench
GPU_C3L_BENCH_CONTRACT=full GPU_C3L_BENCH_TRACKING=true GPU_C3L_BENCH_LAYERS=true ./test/build/command_record_bench
```

For the one-pointer FAST representation:

```sh
c3c build command_record_fast_bench --path test -O1
GPU_C3L_BENCH_CONTRACT=trusted GPU_C3L_BENCH_TRACKING=false GPU_C3L_BENCH_LAYERS=false ./test/build/command_record_fast_bench
```

The executables reject missing or malformed policy variables. Each output has
one exact `validation policy=...` line reporting semantic checks, tracking
calls, reference allocations/increments/releases, and layer selection. Trusted
and object-boundary modes require every policy-work counter to be zero. Both
full modes require semantic and tracking work, releases must equal increments,
and allocations cannot exceed increments. Every warm interval must report zero
device registry, command-table, pipeline-table/cache, and policy reselection.
Each target creates its explicit queue-bound allocator before warmup and outside
every measured interval. Its cold counters report allocator host allocation,
one pool creation, and one complete native command-buffer allocation separately
from warm recording. All four policy modes require zero warm host/native/VMA
allocation; tracking modes use the reference slab allocated at allocator create.
Both targets report recording/executable token size, authoritative-record and
cell size, total fixed command storage, and command-list workloads of 1, 16,
256, and 4,096 commands. The FAST token sizes must equal one pointer.

### Command-path baselines

The command-path baseline separates four kinds of evidence so driver cost does
not obscure library work:

1. `command_wrapper_bench` is an ICD-free CPU floor. It compares an observable
   direct no-op with the equivalent public wrapper for dispatch, draw, barrier,
   viewport, and buffer copy.
2. `command_path_baseline_bench` records the same five operations through
   direct Vulkan and public gpu.c3l paths, using separate warmed command lists
   from one device and alternating which path is timed first. The runner repeats
   it for trusted/no-tracking, object-boundaries/no-tracking, full/tracking, and
   full/tracking-with-layers policies.
3. Dispatch and buffer-copy equivalence executes both paths into distinct,
   pre-seeded outputs. Each output must match a non-zero expectation and its
   paired output.
4. Full lifecycle cases measure begin, bind, 0/1/16/256 commands, end, submit,
   and successful completion wait. Non-zero cases also report incremental cost
   per command relative to the zero-command case.

The paired operation timers include only warmed command recording. Handles,
native objects, command lists, and other shared prerequisites are resolved or
created before timing. Deterministic work snapshots around every operation loop
must show zero host or pool allocation, command-buffer allocation/reset,
image-view or VMA allocation, shader-module or native-pipeline creation, and
device-registry lock acquisition. Lifecycle cases allow only the exact work
permitted by their contract and reject unrelated work.

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
direct/public ratios remain advisory; schema, deterministic work/state, and
semantic failures are blocking.

Static command-policy checking verifies complete field coverage for each
literal runtime `CommandOps` table that exists. Table names, table count,
initializer expressions, and table-free specialization are not contracts.
Allocation, policy selection, tracking, reference publication/release ordering,
and performance are established by allocator operations, work/state counters,
native emissions/faults, and ownership transitions; wall time blocks only when
runner, driver, and comparison profile are pinned.

## Evidence and regression gates

The suite covers:

| Target | Evidence |
|---|---|
| `allocation_bench` | Explicit `CPU_WRITE` allocation and free |
| `command_wrapper_bench` | ICD-free direct no-op and public-wrapper floor for five command classes, with exact volatile observation |
| `command_path_baseline_bench` | Paired direct/public Vulkan recording, zero forbidden work, dispatch/copy readback equivalence, and 0/1/16/256 full-lifecycle cases |
| `command_reference_bench` | Exact public lookup/publication/retain/release outcomes plus upper-bounded private probe, equality, and mutex work for unique, repeated, mixed, forced-collision, and capacity scenarios |
| `command_record_bench` | Bounded-token barrier, semantic-hazard barrier, indirect dispatch, and generated dispatch recording across the validation/tracking matrix |
| `command_record_fast_bench` | One-pointer FAST token/storage evidence; 1/16/256/4,096 command lists; exact native output; zero removed proof, lookup, pin, and warm-allocation work |
| `lifecycle_bench` | Submission, cached completed-point polling, and immediate texture destruction |
| `submit_batch_bench` | Real submit batches of 1/8/32/128/1,024 lists with exact one-visit-per-list duplicate-detection work |
| `pipeline_cache_bench` | Dynamic raster matrix aliasing, raster-state recording, cached duplicate lookup/batches, and exact 1 KiB/64 KiB/1 MiB shader-identity work |
| `resource_create_bench` | Texture, shader-code, allocation, and mixed creation across 1/2/4 workers |
| `descriptor_churn_bench` | Texture-view publication and sampler hits across 1/2/4 workers; zero-through-eight sampler probes at occupancy 8/64/1,024/65,536; upper-bounded texture/swapchain ownership work at descriptor high-water 16/4,096/65,536 |
| `upload_throughput_bench` | Explicit uploads at 4 KiB, 256 KiB, and 4 MiB across 1/2/4 workers |
| `async_overlap_bench` | Serialized and independent graphics/compute submissions |

Each timing target performs its own warmup and fixed repetitions. Generated
dispatch first reserves 64 allocator-local preprocess buffers, prewarms one
untimed 64-record command list, then measures five 64-record lists. Every
execute call in a live list receives a distinct reserved preprocess address.
Completed or discarded lists return compatible buffers to the same allocator.

Required performance evidence executes complete operations and compares narrow
subsystem snapshots:

Blocking records use three outcome classes:

| Class | Blocking rule |
|---|---|
| Semantic invariant | Exact output, fault/state preservation, ownership balance, submission, and completion outcomes |
| Forbidden work | Exact zero allocation, object creation, unrelated locking, policy reselection, and post-bind resolution |
| Minimal native lowering | Exact Vulkan emission count for the named scenario |

Private probes, identity comparisons, and mutex decisions use documented upper
bounds that include zero. Removed FAST encoder/lease/frontend-phase proof work
is exact zero. Timings and unpinned generated assembly are advisory.

| Invariant | Required observation |
|---|---|
| Cold allocator creation | Host allocations, exactly one exact-family command-pool create, one complete native command-buffer allocation call, and configured buffer count |
| Warm begin/bind/dispatch/end | `RecordingWorkCounters`, pipeline/shader creation counts, and pre-bind `CommandResolutionStats` |
| Warm minimal begin + state packet + draw | `RecordingWorkCounters`, pipeline/shader creation counts, pre-bind `CommandResolutionStats`, exactly one native begin per pass, exactly ten dynamic-state commands per explicit complete packet, and native draw emission |
| Generated dispatch/draw/indexed draw | Per-family `RecordingWorkCounters` emissions plus `CommandRecordingStats` reservation/allocation state |
| Cached completion | `CompletionWorkCounters` across 100,000 polls, cached waits, and concurrent first observers |
| Immediate destruction | `CompletionWorkCounters`, injected native-destroy counts, and stalled-queue ordering |
| Submission ownership | Submitted-batch references, caller tokens, retained-reference counts, and ordered retirement state |
| Bound pipeline snapshot | Exact bind-time table/cache lookups, per-bind-point `pipeline_bind_commands`, zero post-bind resolution under cache churn, and unchanged layout use after backing-slot mutation |
| Strict descriptor heap | Exact `strict_heap_set_bind_commands`, `strict_heap_buffer_bind_commands`, and `strict_heap_offset_commands`: one set or offset per used bind point and one descriptor-buffer bind per command record |
| Shader identity | Owned clone/free balance, zero post-intern shader work, and bounded intern/compact-key work at 1 KiB/64 KiB/1 MiB |
| Sampler buckets | Bounded collision-chain probes and a zero-probe empty-bucket miss at 65,536 entries |

In the bounded build, warm `CommandResolutionStats` permit one bounded
command-table resolution and authoritative phase check per recorded public
command. The FAST build requires zero encoder-cell computations, packed-lease
comparisons, frontend phase transitions, command-table lookups, registry/pin
operations, and warm allocation. Both representations require exact native
emission and GPU output. Pipeline and descriptor-heap counters measure native
emission, not public bind attempts, so compatible pipeline switches can increase
pipeline binds without increasing heap binds.

Minimal pass begin accounts for exactly one native begin-rendering command and
no dynamic-state commands. The convenience begin accounts for that begin plus
one ten-command packet. A complete `cmd_set_graphics_state` replacement
accounts for exactly ten; additional passes and draws scale only with their
begin/end/draw commands unless the caller records another setter. The
`{2, 16, 256}` pass-count matrix records one packet before the first pass and
requires no hidden state replay, allocation, state diff, or dirty-bit work.

Warm command-buffer reset is expected reuse evidence. Warm host allocation is
prohibited in every policy mode; tracking modes retain into fixed reference
storage allocated with the command allocator. VMA allocation,
command-buffer allocation/free, image-view creation, pipeline/shader creation,
and registry, retained-pin, lifecycle-vtable, command-table, and policy work
remain prohibited in FAST warm recording; bounded command-token lookup is
reported separately and limited to one per public recording call. Binding an opaque pipeline handle performs exactly one
pipeline-table and one pipeline-cache lookup; dispatch and draw perform no
additional resolution and each emits exactly one root push plus its native
execution command. The dispatch invariant mutates the bound handle's backing
slot after binding and observes the pushed layout, proving command recording
uses the bind-time value snapshot. Private pipeline and command-record member
inventories are intentionally not static contracts.

The lifecycle benchmark first establishes full retirement for each measured
point, then resets completion-work counters before 100,000 repeated polls. The
measured interval requires zero native counter queries and zero retirement-lock
acquisitions; these counters are compiled only for tests and this benchmark
target.

The descriptor-churn benchmark separately varies descriptor high-water state at
16, 4,096, and 65,536. Texture destruction permits zero through one ownership
decision per texture, and the swapchain seam permits zero through one per
wrapped image examined. `ns/destroy` and `ns/check` remain advisory because
runner, driver, build, and profile identity are not fully pinned by that single
observation.

The same target builds synthetic sampler tables at occupancy 8, 64, 1,024, and
65,536 with the production canonical hash, bucket, link, equality, and lookup
helpers. Every tier requires a power-of-two bucket count at least twice the
occupancy and between zero and eight candidate probes for the selected hit.
Every tier also requires zero candidate probes for a guaranteed empty-bucket
miss. The 65,536-entry zero-probe observation is also enforced by the live
`vk_descriptor_heap` test target. Elapsed lookup time is advisory.

The submit-batch benchmark submits real executable command lists in batches of
1, 8, 32, 128, and 1,024. Its feature-gated counters require exactly one token
visit per list and zero epoch-reset cells for each ordinary batch. These exact
work counts are blocking; `ns/submit` remains advisory.

The pipeline-cache benchmark separately interns synthetic 1 KiB, 64 KiB, and
1 MiB identities. For every size, the live `vk_pipeline_cache` test and the
benchmark require one complete owned clone and free, zero or one compact-key
probe, and zero shader probes, byte comparisons, and clones after interning.
Collision and distinct-storage tests bound intern probes and compared bytes
while preserving exact identity and clone outcomes. Boundary elapsed time is
advisory.

### Expectation changes and generated assembly

Exact semantic or native-work changes are hand-edited. The same review updates
the scenario, expected value, `expectation_version`, rationale, and before/after
raw records. Observed output is never copied back into expectations
automatically; increases require a reason that the additional native work is
necessary.

The FAST zero-proof gates depend on #438's outcome taxonomy, which permits a
removed private mechanism to report zero rather than requiring one observation.
They cover token resolution and authoritative record lifetime only. Runtime
`CommandOps` indirect dispatch and fallible recording signatures remain
intentional here; #440 owns their later build-time specialization and any final
FAST/CHECKED public artifact split.

Representative dispatch, draw, barrier, viewport, and buffer-copy assembly can
be reported locally:

```sh
python -B scripts/report_command_asm.py --emit
```

The reviewed Linux C3 0.8.0 profile can be enforced with:

```sh
python -B scripts/report_command_asm.py \
  --emit \
  --pinned-compiler 0.8.0 \
  --pinned-target linux-x64 \
  --comparison-profile command-fastpath-o1-v1 \
  --limits scripts/command_asm_profiles/c3-0.8.0-linux-x64-o1-v1.json
```

The reporter invokes C3 0.8.0 with `-O1 --emit-asm` and records broad function,
call, indirect-call, atomic, branch, load/store, and native-dispatch
observations. A blocking run verifies the installed compiler version, passes
the named target to C3, and requires the CLI identity and optimization mode to
match the versioned JSON profile. Missing symbols, unknown instruction forms,
and count variation remain advisory outside that profile. Exact instruction
bytes are never compared.

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

### Lifetime-reference index

Tracking-enabled command scratch uses a fixed open-addressed index beside the
sequential release list. Ordinary lookup/insertion is expected constant-time,
compares the complete owner/index/generation identity after every hash hit, and
allocates no recording memory. N unique references therefore require N lookups,
N retains, N publications, and expected O(N) total probes rather than repeated
scans of the accumulated list. Duplicate hits acquire no resource mutex and add
no retain. Forced-collision timing is advisory; bounded probes, exact identity,
and balanced retain/release counters are blocking evidence.
`command_reference_bench` exercises unique sets of 1/8/64/256/1,024/4,096,
100,000 duplicate hits, a 2,048-unique/2,048-duplicate mix, a 64-entry forced
collision chain, and a two-reference preflight at 4,095 retained entries. Its
schema requires zero warm host allocation and exact publication, mutex,
retain, release, duplicate, probe, and equality work.

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
- Cache pipelines; record topology, cull, front-face, depth-bias, viewport,
  scissor, and depth once with the complete setter, then use minimal begin for
  compatible passes that reuse it. Use the convenience begin when pass and
  packet should be one transactional operation; later permutations should
  change state with complete or partial setters.
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
  choose the earliest real consumer in `CompletionWait.before` and
  `CompletionWait.consumers`; an unnecessarily broad `.all` mask can reduce
  available overlap. Use `consumers.draw_arguments` for GPU-produced indirect,
  count, and implicitly preprocessed generated-command input. The Vulkan backend
  includes command-preprocess scope automatically when generated work is enabled.
  Narrower waits permit overlap but do not guarantee it.

`completion_wait_scope_bench` compares `.all` with
`consumers.draw_arguments` over a real compute-producer to indirect-dispatch
consumer chain on two distinct compute queues. Each consumer records independent
compute work before the indirect dispatch so the narrow wait exposes a real
overlap opportunity; warmups and alternating case order reduce timing bias. It
reports not-applicable when that topology is unavailable. Output equivalence and
exact native stage-mask tests are hard evidence; the elapsed ratio is advisory.
