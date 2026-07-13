# gpu.c3l

[![ci](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml/badge.svg)](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml)

A GPU programming library for [C3](https://c3-lang.org/). The package contains
a strict GPU-shaped root (`gpu`) and the stabilized Vulkan 1.3 API
(`gpu::compat`). Compatibility is explicit, runtime-inert on import, and built
around two ideas. The implemented runtime currently lives in `gpu::compat`:

- **Root pointers instead of descriptor sets.** Buffers are reached through
  a single 64-bit GPU address pushed per dispatch/draw; your shader casts it
  to a struct and walks to its data. No binding numbers, no set layouts, no
  descriptor churn for buffer data.
- **One bindless heap for textures.** `TextureIndex`/`SamplerIndex` are compact
  generation-checked CPU tokens with shader-visible slots. Put them in your own
  structs; the library-managed global heap is the only set that exists.

```c3
import gpu::compat;

gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame, DoublerRoot::size, 16)!;
DoublerRoot* root = (DoublerRoot*)root_span.cpu;
root.input_gpu  = gpu::compat::get_buffer_address(&device, input)!;
root.output_gpu = gpu::compat::get_buffer_address(&device, output)!;
root.count      = COUNT;

gpu::compat::cmd_dispatch(
    commands: &cmd,
    pipeline: pipeline,
    root:     root_span.gpu,
    groups:   { (COUNT + 63) / 64, 1, 1 },
)!;
```

## Features

- Explicit Vulkan 1.3 compatibility backend (dynamic rendering, timeline semaphores, sync2, BDA);
  the public API is GPU-shaped, not Vulkan-shaped — no `vk::` types leak
- Shader ABI generator: one `.abi` schema emits the C3 struct (with
  compile-time size/offset asserts) and the GLSL include, plus a CI drift gate
- VMA-backed memory: per-frame arenas, persistent arena, staging/readback
  arenas, dedicated allocations, leak reporting
- Explicit barrier model with tracked texture layouts; validation-clean is a
  test gate across the whole suite
- GPU-driven path: multi-draw indirect (+ count), draw tables via `gl_DrawID`
- Tiered threading: per-thread recording contexts, lock-free frame-arena
  allocation, one submit
- Pipeline dedup cache + driver-cache save/load; swapchain with present-mode
  query; compare samplers, depth bias, MRT
- Runs entirely on lavapipe (CPU Vulkan) — CI needs no GPU, and neither does
  your first program

## Status

Pre-1.0, pinned to **C3 0.8.0** (the language is pre-1.0 too; syntax moves).

| Target | State |
|---|---|
| linux-x64 | Full library test suite, validation-clean on lavapipe |
| windows-x64 | Full blocking library test suite on mesa-dist-win lavapipe |

## Start here

- **[Getting started](docs/getting_started.md)** — empty directory to a
  running GPU compute program; the walkthrough is executed by CI, so it
  cannot rot.
- **[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples)** —
  18 runnable samples: triangle → textured cube → GPU-driven culling →
  shadow mapping → deferred shading → PBR → stress/perf harnesses.
- **[docs/document_index.md](docs/document_index.md)** — map of the full
  documentation set (architecture, API, memory model, shader ABI, backend,
  testing, style).
- **[Strict GPU profile](docs/strict_gpu_profile.md)** — target architecture
  and the gated implementation plan for `gpu` and `gpu::compat`.

## Layout

```
gpu.c3l/
├── abi/               shader ABI schemas
├── gpu/
│   ├── gpu.c3(i)      strict root (`gpu`)
│   ├── types.c3       strict nominal boundary types
│   └── compat/        stabilized API (`gpu::compat`)
│       └── vk/        private Vulkan backend (`gpu::compat::vk`)
├── lib/               vendored bindings: vk.c3l, vma.c3l, spvreflect.c3l
├── include/shaders/   GLSL includes consumed by shaders
├── scripts/           ABI, shader, and documentation checks
├── tools/gen_shader_abi/
├── test/              whitebox test suite (compiles library sources directly)
└── docs/
```
