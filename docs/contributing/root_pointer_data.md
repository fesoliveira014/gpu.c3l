# Root-pointer data vs a fixed dynamic-uniform path

Small per-dispatch and per-draw data reaches shaders through the root push
contract and ordinary GPU-addressable memory. This page records what a fixed,
library-owned dynamic-uniform alternative would cost, what can be settled
statically, and what still needs a device measurement.

## Current path

- Allocate `CPU_WRITE` storage and take a mapped span
  (`mapped_gpu_span`, `gpu/gpu.c3:932`; `MappedGpuSpan`, `gpu/gpu.c3i:738`).
- Write the record, then `flush_mapped_span` (`gpu/gpu.c3:840`) when the range
  is not coherent; `invalidate_mapped_span` (`gpu/gpu.c3:857`) covers the read
  direction.
- Pass the span address as the `root` argument of `cmd_dispatch`
  (`gpu/gpu.c3:2249`) or the two root arguments of `cmd_draw`
  (`gpu/gpu.c3:2385`).
- The backend pushes 8 bytes for compute (`ROOT_PUSH_ABI.block_size == 8`,
  `gpu/internal/shader_abi.c3:20`) and 16 bytes for graphics
  (`docs/shader_abi.md:17-26`).
- The shader dereferences the address as a `buffer_reference` block.

## Comparison path

The alternative under consideration is deliberately minimal: one fixed
library-owned uniform binding, one persistently mapped ring, one dynamic offset
per command, one fixed pipeline layout, and no public descriptor layouts, sets,
pools, or update calls.

## Scope of the evidence here

The comparison path is **not built**, and no timing on this page comes from a
software driver. `benchmarking.md:65-67` records that software drivers may
dominate CPU-side library costs, so a lavapipe number cannot support or refute
a GPU-side binding change in either direction. The decision needs the primary
development GPU. Everything below that carries a number is derived from the
repository or from a stated arithmetic assumption.

## Workloads and record layouts

Two record shapes come from the repository:

| Record | Source | Bytes |
| --- | --- | --- |
| `ComputeRoot` | `test/src/command_path_baseline_bench.c3:25-40` | 32 |
| `DoublerRoot` | `examples/getting_started/src/main.c3:9-13` | 24 |

`DoublerRoot` is 8 + 8 + 4 bytes of payload and 24 bytes with trailing
alignment padding. `ComputeRoot` pins its offsets with `$assert`.

A comparison target belongs beside `command_path_baseline_bench`
(`test/project.json:34-40`), which already exercises the root-pointer path with
`test/shaders/root_pointer.comp.glsl`, a warm-up/repetition/median harness
(`COMMAND_PATH_ITERATIONS = 20_000`, `COMMAND_PATH_REPETITIONS = 5`,
`test/src/command_path_baseline_bench.c3:11-12`, `sample_median` at `:243`), and
sentinel-checked output equivalence (`:285`, `:444`). Reuse those conventions
rather than inventing a harness. Cover 32/64/256-byte records, compute dispatch
and graphics draw, and a low and a high command count.

## Measurements

Report these axes separately. Environment reporting
follows `benchmarking.md:17-25`; do not restate it in the target.

| Axis | Status |
| --- | --- |
| Bytes written and flushed | derived below |
| Native commands and pipeline-layout state | derived below |
| CPU record preparation time | not collected — requires the primary development GPU |
| Command-recording time | not collected — requires the primary development GPU |
| Descriptor/pipeline binding time | not collected — requires the primary development GPU |
| GPU execution time (`cmd_write_timestamp`, `gpu/gpu.c3:633`) | not collected — requires the primary development GPU |
| End-to-end time | not collected — requires the primary development GPU |

### Bytes written and flushed

A dynamic uniform offset must be a multiple of
`minUniformBufferOffsetAlignment`. The root-pointer path has no such
constraint; its slot stride is the record rounded to its own alignment (8 bytes
for both records above).

That limit varies by an order of magnitude across implementations, so the
effect is reported as a sensitivity rather than a single number. 256 bytes is
the largest value Vulkan permits, not a typical one; 64 and 16 are commonly
reported on desktop discrete parts, and the lavapipe adapter used for this
repository's software runs reports 16.

Uniform-ring stride, and amplification against the root-pointer stride:

| Record | Root stride | at 16 B | at 64 B | at 256 B |
| --- | --- | --- | --- | --- |
| 24 B (`DoublerRoot`) | 24 B | 32 B (1.33x) | 64 B (2.67x) | 256 B (10.67x) |
| 32 B (`ComputeRoot`) | 32 B | 32 B (1.00x) | 64 B (2.00x) | 256 B (8.00x) |
| 64 B | 64 B | 64 B (1.00x) | 64 B (1.00x) | 256 B (4.00x) |
| 256 B | 256 B | 256 B (1.00x) | 256 B (1.00x) | 256 B (1.00x) |

At 20 000 commands the 24-byte record moves 480 000 bytes through the root
path, against 640 000 at 16-byte alignment and 5 120 000 at 256.

So this cost is real only where the limit is large, and it nearly vanishes at
16 bytes. Read the actual value from `VkPhysicalDeviceLimits` on the primary
development GPU before using this table as evidence.

This is a bandwidth and footprint statement, not a latency one. It says nothing
about which path executes faster. On non-coherent memory a per-record flush
rounds both paths up to `nonCoherentAtomSize` (64 bytes on the same adapter),
which erodes the root path's advantage for the small records; the strides above
hold for one contiguous end-of-frame flush.

### Native commands and pipeline-layout state

The bindless descriptor set is bound at most once per bind point per command
buffer and carries no dynamic offsets (`bind_descriptor_heap`,
`gpu/internal/vk/descriptor_heap.c3:714-745`, `dynamic_offset_count: 0`).

| Per compute dispatch | Root pointer | Fixed dynamic uniform |
| --- | --- | --- |
| Push constants | 8 B (`push_compute_root`, `gpu/internal/vk/command.c3:2238-2252`) | droppable for compute only — see below |
| Descriptor-set binds | once per command buffer | once per command, with one dynamic offset |
| Dispatch calls | 1 | 1 |

At 20 000 dispatches that is one set bind plus 20 000 8-byte pushes against
20 000 set binds. Which is cheaper on real hardware is exactly the open
question.

Graphics does not fit the one-binding shape. A graphics command pushes two
independent roots in a 16-byte block (`docs/shader_abi.md:17-26`,
`gpu/internal/vk/pipeline_cache.c3:577-582`), so one uniform binding with one
dynamic offset cannot express it: the uniform path would need two bindings, two
dynamic offsets, or retained push constants for the graphics case alone.

## Implementation-complexity comparison

The root-pointer path adds nothing: `GpuAddress` and mapped spans already exist
and are already used for indirect arguments and generated work. The uniform
path adds, at minimum:

- A seventh accepted set-0 binding. `check_heap_convention`
  (`gpu/internal/vk/shader.c3:317-374`) faults `SHADER_INVALID` on any set other
  than 0 and on any set-0 binding outside the six heap bindings
  (`gpu/internal/vk/descriptor_heap.c3:11-16`), so the published binding domain
  in `docs/shader_abi.md` widens and every consuming shader inherits the change.
- Extra state on all three shared pipeline layouts — graphics, compute, and ray
  tracing (`gpu/internal/vk/pipeline_cache.c3:571-650`) — plus the invariant
  check at `:555`.
- Per-command dynamic-offset state in recording, and the loss of the
  bind-once-per-command-buffer property above.
- Ring lifetime, wraparound, and completion-based reuse rules, an alignment rule
  driven by a device limit, and a bounded per-command size limit.
- A second set of flush rules alongside the existing mapped-span rules.

## Recommendation

**Retain root pointers only.** The decision rule is asymmetric: a production
implementation is warranted only when the fixed uniform path shows a
repeatable, material improvement, so absent that evidence the current model
stands. This is not a finding that the uniform path is slower — it has not been
measured. The recommendation rests on the absent evidence plus the complexity
list above. The one static performance-adjacent fact, ring write amplification,
favours the current path on footprint only, and only where
`minUniformBufferOffsetAlignment` is large — at the 16-byte alignment this
repository's software adapter reports, it is close to nothing.

Revisit when a sibling of `command_path_baseline_bench` on the primary
development GPU shows the uniform path reducing median end-to-end time for the
20 000-command case by a repeatable margin large enough to justify the list
above — a 10% threshold is a reasonable starting bar — with validation layers
disabled, at least five repetitions, and the improvement holding for the 32-byte
record.
