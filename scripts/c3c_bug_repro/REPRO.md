# c3c 0.8.0 codegen SIGSEGV — reproduction & investigation notes

## Summary

`c3c 0.8.0` intermittently/​deterministically SIGSEGVs during LLVM codegen when
building the `gpu.c3l` Vulkan backend test targets. No diagnostic is printed, no
object file is emitted; exit code 139. The crash is in a background codegen
thread (gdb backtrace lands in stripped libLLVM frames).

This is **not** disk-related (reproduces with 163 GB free), **not** raw code
volume (c3vq — a much larger C3/Vulkan project on the same c3c — never hits it),
and **not** cleanly attributable to a single language construct (see negative
results below). It appears to be an accumulation/interaction effect in codegen.

## Reliable reproduction (in-project)

Requires the `gpu.c3l` repo + a Vulkan 1.3 loader (lavapipe is fine).

```sh
# from repo root
sh scripts/build_shaders.sh
cd test
c3c test vk_bootstrap          # -> SIGSEGV (139), no object emitted
```

- Deterministic in the current tree: `vk_bootstrap` (6 source files) crashes ~100%.
- **Each of the 6 files compiled ALONE builds fine** (rc 0). Only the combination crashes.
- Smaller targets built from the same backend are green: `vk_command`, `vk_root_pointer`, `unit`, and the `samples/root_pointer_compute` executable.
- For the sample, `c3c --emit-llvm` writes `std.*`, `gpu.ll`, `gpu.slots.ll`, `vk.ll`, then dies emitting `gpu.vk.ll` — i.e. the crash is in the `gpu::vk` module's codegen.

## Confirmed contributing pattern (worked around in-tree)

`vk::ComputePipelineCreateInfo.set_stage(stage)` takes the large embedded
`PipelineShaderStageCreateInfo` **by value**. In the full backend this
participated in a codegen crash; assigning the struct fields directly
(`pipe.stage = ...; pipe.layout = ...;`) instead of the builder setter avoided
that instance. (See `vk/pipeline_compute.c3`.)

## Negative results — what does NOT reproduce standalone (safety on)

Each of the following compiles cleanly in isolation; none triggers the SIGSEGV:

1. A generic slot-table module (`SlotTable{Handle, Value}`) instantiated **16×**
   with distinct value types, methods returning optionals, bitstruct handles,
   called from `main`. Compiles clean.
2. The `set_stage` by-value builder pattern above, standalone with `import vk`.
   Compiles clean.
3. The `vk_submit`-shaped function (`@pool` + `talloc_array` + `foreach` +
   by-value builder chain + slice setter) with dummy types. Compiles clean.

So the trigger is the **combination** in the real `gpu::vk` module (VMA + vk
bindings + slot tables + command/submit/pipeline paths reachable via a
function-pointer vtable), not any one of these alone. It resisted reduction to a
minimal self-contained case within a large time budget.

## In-tree impact & mitigation

- The generic slot-table module (`gpu::slots`) was replaced with concrete
  per-type tables (matching c3vq's pattern). This made the M9 targets
  (`vk_root_pointer`, `vk_command`, sample) reliably green with safety on.
- `vk_bootstrap` still crashes regardless of generic-vs-concrete tables, so the
  generic module is **not** the sole cause.

## c3c edge cases noticed while minimizing (possibly separate issues)

- All-uppercase identifiers (e.g. a type named `H`) are rejected as types in
  local declarations ("Parameter names may not be all uppercase" / "Expected a
  type here"). Using a mixed-case name resolves it.
- A generic-instantiation `alias T = SlotTable{H, V};` used as a **local
  variable** type reports "Expected a type here", while the same instantiation
  used as a **struct field** type or written inline compiles.

## Environment

- c3c 0.8.0 (git d78f10d), linux-x64, WSL2.
- Reproduces at `--threads 1` and default; `--safe=no` reduces frequency for
  small targets but does not fix `vk_bootstrap`.
