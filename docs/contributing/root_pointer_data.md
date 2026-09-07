# Root-pointer data vs a dynamic-uniform path

**Question.** Should small per-command shader data get a library-owned
dynamic-uniform path (one fixed binding, one mapped ring, one dynamic offset
per command) beside the root-pointer path?

**Decision.** No. Root pointers, plus an inline payload of up to 120 bytes
in the 128-byte push block for small per-command data. The cookbook's
"inline for records that fit" default is provisional until the hardware
run below exists.

## Current paths

Record: write a record into `CPU_WRITE_GPU_LOCAL` memory, flush, pass its
`GpuAddress` as the root of `cmd_dispatch` or the two roots of `cmd_draw`.
The shader reads the record through a `buffer_reference` block.

Inline: pass up to 120 bytes as `inline_root`; the backend pushes them after
the 8 or 16-byte header in the same `vkCmdPushConstants`. See
[shader ABI](../shader_abi.md#root-push).

## What was measured

`command_path_baseline_bench` has inline-vs-record phases. Recording: 20,000
dispatches, payload 32, 64, and 112 bytes, inline against a memcpy into a
mapped ring plus a dispatch by address, median of 5. Execution: 1,000
dispatches that each double one input element into a disjoint output
element with the same shader logic and the same 24-byte root
(input address, output address, count), carried inline against a
per-command record, both legs validated, timed as record-plus-end and as
submit-plus-wait.

lavapipe (Mesa 25.0.7, WSL2, validation off), 2026-09-06:

| Phase | Inline | Record |
|---|---:|---:|
| Recording, 32 B payload | 84.7 ns/op | 80.6 ns/op |
| Recording, 64 B payload | 84.2 ns/op | 86.8 ns/op |
| Recording, 112 B payload | 86.1 ns/op | 79.6 ns/op |
| 1,000 dispatches, record | 165.1 ns/op | 144.5 ns/op |
| 1,000 dispatches, execute | 72.5 ms | 74.2 ms |

The two paths are within run-to-run noise on the software driver. The
dependent load a record costs on hardware is not visible here. The issue's
required run (dispatch and draw, 32/64/112-byte payloads, 1k and 20k
commands, preparation, recording, GPU, and end-to-end separated, both
memory classes) is still owed on hardware; the cookbook default rests on the
removed allocation, mapping, flush, and ring and is provisional until then.

The dynamic-uniform alternative was not built. The numbers below are
derived from the repository or stated arithmetic.

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
