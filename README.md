# gpu.c3l

[![ci](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml/badge.svg)](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml)

A GPU programming library for [C3](https://c3-lang.org/), built on Vulkan 1.3
— with the Vulkan kept out of your way. One public module (`gpu`), strongly-typed
handles, C3 optionals for every error, and an execution model built around
two ideas:

- **Root pointers instead of descriptor sets.** Generic GPU data is reached
  through 64-bit addresses; each dispatch/draw pushes one root address and the
  shader follows pointers from that struct. No binding numbers, set layouts,
  or descriptor churn for generic data.
- **One bindless heap for textures and samplers.** `TextureIndex` and
  `SamplerIndex` are shader-visible slots for root data. Sampler state is
  interned as a device-owned `Sampler` identity, then published once into the
  strict heap.

```c3
gpu::GpuSpan input_span = gpu::get_allocation_span(&device, input)!;
gpu::GpuSpan output_span = gpu::get_allocation_span(&device, output)!;
gpu::AllocationDesc root_desc = {
    .size         = DoublerRoot::size,
    .alignment    = DoublerRoot::alignment,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .compute },
    .debug_name   = "doubler_root",
};
gpu::GpuAllocation root_allocation =
    gpu::allocate_memory(&device, &root_desc)!;
gpu::GpuSpan root_span =
    gpu::get_allocation_span(&device, root_allocation)!;
DoublerRoot* root =
    (DoublerRoot*)gpu::get_span_mapping(&device, root_span)!.ptr;
root.input_gpu  = gpu::get_span_address(&device, input_span)!;
root.output_gpu = gpu::get_span_address(&device, output_span)!;
root.count      = COUNT;
gpu::flush_mapped_span(&device, root_span)!;

gpu::cmd_dispatch(
    commands: &cmd,
    pipeline: pipeline,
    root:     gpu::get_span_address(&device, root_span)!,
    groups:   { (COUNT + 63) / 64, 1, 1 },
)!;
```

Keep `root_allocation` live until the submission's `CompletionPoint` finishes,
then free or reuse it.

## Features

- Vulkan 1.3 backend (dynamic rendering, timeline semaphores, sync2, BDA);
  the public API is GPU-shaped, not Vulkan-shaped — no `vk::` types leak
- Shader ABI generator: one `.abi` schema emits the C3 struct (with
  compile-time size/offset asserts) and the GLSL include, plus a CI drift gate
- VMA-backed memory: independent `GpuAllocation` storage with checked spans,
  mapping/address queries, explicit transfers, and leak reporting
- Explicit barrier model with tracked texture layouts; validation-clean is a
  test gate across the whole suite
- GPU-driven path: multi-draw indirect (+ count), draw tables via `gl_DrawID`
- Tiered threading: automatic per-worker command pools, thread-safe allocation,
  completion-driven command-buffer reclamation
- Pipeline dedup cache + driver-cache save/load; swapchain with present-mode
  query; compare samplers, depth bias, MRT
- Runs entirely on lavapipe (CPU Vulkan) — CI needs no GPU, and neither does
  your first program

## Status

Pre-1.0, pinned to **C3 0.8.0** (the language is pre-1.0 too; syntax moves).

| Target | State |
|---|---|
| linux-x64 | Full library test suite, validation-clean on lavapipe |
| windows-x64 | Build and CPU gates; advisory Vulkan sweep with mesa-dist-win |

## Start here

- **[Getting started](docs/getting_started.md)** — empty directory to a
  running GPU compute program; the walkthrough is executed by CI, so it
  cannot rot.
- **[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples)** —
  18 runnable samples plus the helper self-test: triangle → textured cube →
  GPU-driven culling → shadow mapping → deferred shading → PBR → stress/perf
  harnesses.
- **[docs/document_index.md](docs/document_index.md)** — map of the full
  documentation set (architecture, API, memory model, shader ABI, backend,
  testing, style).

## Layout

```text
gpu.c3l/
├── abi/               shader ABI schemas
├── gpu/               shipped library sources (module `gpu`)
│   └── vk/            Vulkan backend (private module `gpu::vk`)
├── lib/               vendored bindings: vk.c3l, vma.c3l, spvreflect.c3l
├── include/shaders/   GLSL includes consumed by shaders
├── scripts/           ABI, shader, and documentation checks
├── tools/gen_shader_abi/
├── test/              whitebox test suite (compiles library sources directly)
└── docs/
```
