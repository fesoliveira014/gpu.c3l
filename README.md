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

- one `gpu` module plus one surface module per window system;
- root pointers (`GpuAddress`) and bindless texture and sampler indices
  instead of descriptor sets;
- explicit memory classes, checked spans, mapping, and visibility;
- compute, graphics, mesh, and opt-in ray-tracing pipelines; indirect and
  generated work; sparse textures; timestamp queries; swapchains;
- caller-owned command allocators and completion points; no hidden waits;
- optional `gpu::util` device context, full contract validation, and
  structured diagnostics; and
- `gpu_shaders`: one tool that generates matching C3 and GLSL ABI declarations
  and compiles shaders.

`TextureIndex`, `SamplerIndex`, `AccelerationStructureIndex`, and `GpuAddress`
are raw shader values, not ownership tokens. Applications must keep their
backing resources alive and explicitly order all reuse, transitions, and
destruction.

## Requirements and status

The current release targets **C3 0.8.3** and a Vulkan 1.3 implementation with
the required modern synchronization, dynamic-rendering, descriptor-indexing,
buffer-device-address, and dynamic-state features. Supported library targets
are `linux-x64` and `windows-x64`. SDL3 is used by the samples, not by the
library itself.

This project is pre-1.0. See
[features and limitations](docs/features_and_limitations.md) before adopting
it for a platform or workload.

## Start here

- [Concepts](docs/concepts.md) — explicit GPU programming as this library
  exposes it, for readers who have not used Vulkan or D3D12.
- [Getting started](docs/getting_started.md) — run a minimal compute program,
  then build an SDL3 triangle.
- [Documentation](docs/index.md) — concepts, recipes, and API reference.
- [Public API](docs/api/index.md) — domain-oriented symbol reference.
- [Device context](docs/util/device_context.md) — headless or windowed setup
  with explicit application-controlled work and shutdown.
- [Sample applications](https://github.com/fesoliveira014/gpu.c3l-samples) —
  maintained end-to-end examples.

## Install

Add the repository as a submodule at `lib/gpu.c3l`, pinned to a release tag.
The checkout brings its three binding packages (`vk`, `vma`, `spvreflect`)
with vendored native libraries for `linux-x64` and `windows-x64`:

```sh
git submodule add https://github.com/fesoliveira014/gpu.c3l lib/gpu.c3l
git -C lib/gpu.c3l checkout v0.5.0
git submodule update --init --recursive lib/gpu.c3l
git config -f .gitmodules submodule.lib/gpu.c3l.shallow true
```

Without git, download the archive for your target from the
[latest release](https://github.com/fesoliveira014/gpu.c3l/releases/latest)
and extract it into `lib/`; it unpacks to the same `lib/gpu.c3l` layout with
the library, bindings, native libraries for that target, the `gpu_shaders`
tool, shader include files, licenses, and consumer documentation.

[Getting started](docs/getting_started.md#install) shows the `project.json`
that resolves the four packages and the shader build.
