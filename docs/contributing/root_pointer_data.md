# Root-pointer data vs a dynamic-uniform path

**Question.** Should small per-command shader data get a library-owned
dynamic-uniform path (one fixed binding, one mapped ring, one dynamic offset
per command) beside the root-pointer path?

**Decision.** No. Root pointers only.

## Current path

Write a record into `CPU_WRITE` memory, flush, pass its `GpuAddress` as the
root of `cmd_dispatch` or the two roots of `cmd_draw`. The backend pushes 8
bytes for compute and 16 for graphics. The shader reads the record through a
`buffer_reference` block. See [shader ABI](../shader_abi.md#root-push).

## What was measured

Nothing on a device. The alternative was not built. Software-driver timing
cannot decide a GPU-side binding question (see
[benchmarking](benchmarking.md#reading-results)). The numbers below are
derived from the repository or stated arithmetic.

A future measurement belongs beside `command_path_baseline_bench`
(`test/src/command_path_baseline_bench.c3`), which already has the
root-pointer path, a warm-up/repetition/median harness, and output
equivalence checks. Cover 32, 64, and 256-byte records, dispatch and draw,
and low and high command counts. Report record preparation, recording,
binding, GPU time, and end-to-end time separately.

## Derived costs

**Write amplification.** A dynamic uniform offset must be a multiple of
`minUniformBufferOffsetAlignment` (16 to 256 bytes by device; lavapipe
reports 16). The root path strides at the record's own alignment.

| Record | Root stride | at 16 B | at 64 B | at 256 B |
|---|---:|---:|---:|---:|
| 24 B | 24 | 32 (1.3x) | 64 (2.7x) | 256 (10.7x) |
| 32 B | 32 | 32 (1.0x) | 64 (2.0x) | 256 (8.0x) |
| 64 B | 64 | 64 | 64 | 256 (4.0x) |
| 256 B | 256 | 256 | 256 | 256 |

This is footprint, not latency. Per-record flushes on non-coherent memory
round both paths to `nonCoherentAtomSize` and erase the difference for small
records.

**Native commands per dispatch.** Root path: one heap set bind per command
buffer plus an 8-byte push per dispatch. Uniform path: one set bind with a
dynamic offset per dispatch. Which is cheaper on hardware is the open
question.

**Graphics does not fit.** A draw pushes two independent roots. One binding
with one dynamic offset cannot express that; the uniform path would need two
bindings or keep push constants for graphics anyway.

## Implementation cost of the alternative

- A seventh set-0 binding, widening the published heap convention that
  `check_heap_convention` enforces and every shader inherits.
- Extra state on all three shared pipeline layouts.
- Per-command dynamic-offset state and loss of bind-once-per-command-buffer.
- Ring lifetime, wraparound, completion-based reuse, a device-limit-driven
  alignment rule, and a per-command size limit.
- A second flush rule beside the mapped-span rules.

The root-pointer path adds nothing; addresses and mapped spans already serve
indirect arguments and generated work.

## Revisit when

A sibling of `command_path_baseline_bench` on the primary development GPU,
validation layers off, at least five repetitions, shows the uniform path
cutting median end-to-end time for 20,000 commands by a repeatable margin
(10% is a reasonable bar) and the gain holds for the 32-byte record.
