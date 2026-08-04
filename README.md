# gpu.c3l

[![ci](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml/badge.svg)](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml)

`gpu.c3l` is a GPU programming library for [C3](https://c3-lang.org/). It
provides a GPU-shaped public API over a private Vulkan 1.3 backend: strongly
typed handles, explicit memory and synchronization, root-pointer shader data,
and bindless texture and sampler heaps.

The library aims to make modern, explicit GPU programming practical without
exposing Vulkan objects or descriptor-set management. It does not provide a
render graph, resource streaming, hidden state transitions, implicit lifetime
management, or a compatibility descriptor path.

## Highlights

- one `gpu` module plus platform-specific surface modules;
- root pointers (`GpuAddress`) for per-dispatch and per-draw shader data;
- bindless texture and sampler indices;
- VMA-backed allocations, checked spans, mapping, and explicit visibility;
- compute, graphics, and explicitly opted-in direct ray-tracing pipelines;
- dynamic rendering, indirect and generated work, acceleration structures,
  ray queries, sparse textures, timestamp queries, and swapchains;
- caller-owned command allocators with explicit completion-based reuse;
- optional full contract validation and structured diagnostics; and
- a schema generator for matching C3 and GLSL shader ABI declarations.

`TextureIndex`, `SamplerIndex`, `AccelerationStructureIndex`, and `GpuAddress`
are raw shader values, not ownership tokens. Applications must keep their
backing resources alive and explicitly order all reuse, transitions, and
destruction.

## Requirements and status

The current release targets **C3 0.8.0** and a Vulkan 1.3 implementation with
the required modern synchronization, dynamic-rendering, descriptor-indexing,
buffer-device-address, and dynamic-state features. Supported library targets
are `linux-x64` and `windows-x64`. SDL3 is used by the samples, not by the
library itself.

This project is pre-1.0. See
[features and limitations](docs/features_and_limitations.md) before adopting
it for a platform or workload.

## Start here

- [Getting started](docs/getting_started.md) — run a minimal compute program,
  then build an SDL3 triangle.
- [Documentation](docs/index.md) — concepts, recipes, API reference, and
  contributor guides.
- [Public API](docs/api/index.md) — domain-oriented symbol reference.
- [Sample applications](https://github.com/fesoliveira014/gpu.c3l-samples) —
  maintained end-to-end examples.

Clone with submodules:

```sh
git clone --recurse-submodules https://github.com/fesoliveira014/gpu.c3l.git
```
