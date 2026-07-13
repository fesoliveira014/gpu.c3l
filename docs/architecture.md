# gpu.c3l Architecture

## Purpose

`gpu.c3l` is a direct GPU programming library for C3. It is not a renderer,
render graph, material system, asset system, or windowing layer.

The package exposes two profiles:

| Module | Role |
|---|---|
| `gpu` | Strict GPU-shaped type boundary. |
| `gpu::compat` | Stabilized Vulkan 1.3 API. |
| `gpu::compat::vk` | Private compatibility backend. |

The [strict GPU profile](strict_gpu_profile.md) defines the root API target.
The implemented runtime currently lives entirely in `gpu::compat`.

## Profile boundary

Strict and compatibility devices, resources, commands, barriers, descriptors,
and pipelines are nominally distinct C3 types. There are no aliases between
profiles and neither profile can fall back to the other.

C3 imports submodules recursively, so `import gpu;` may expose
`gpu::compat`. Importing either module is runtime-inert. Compatibility state is
created only by `gpu::compat::create_device`.

```c3
import gpu::compat;

gpu::compat::Device device = gpu::compat::create_device(&desc)!;
defer gpu::compat::destroy_device(&device)!!;
```

## Package layout

```text
gpu.c3l/
├── abi/                       shader ABI schemas
├── gpu/
│   ├── gpu.c3                 strict root
│   ├── gpu.c3i
│   ├── types.c3               strict nominal types
│   └── compat/
│       ├── compat.c3          compatibility entry point
│       ├── compat.c3i
│       ├── *.c3               compatibility API
│       └── vk/*.c3            private Vulkan backend
├── include/shaders/           published shader ABI includes
├── lib/                       vk, vma, and spvreflect bindings
├── scripts/                   generation and contract checks
└── test/                      CPU, Vulkan, and profile-boundary tests
```

`manifest.json` provides the single `gpu` package and lists both profiles.
There is one relocated compatibility implementation; no root-level copy is
retained.

Compatibility files declare:

```c3
module gpu::compat;
```

Backend files declare:

```c3
module gpu::compat::vk @private;
```

The backend imports `gpu::compat` with a visibility override. Consumer code
must not import backend declarations.

## Compatibility runtime

### Device

`gpu::compat::Device` is an opaque generation token for one process-wide live
device. Device state owns queues, resource tables, VMA, descriptor storage,
arenas, pipeline caches, diagnostics, and deferred destruction.

All compatibility handles, indices, addresses, spans, commands, and timeline
values belong to that device lifetime. Multi-device operation is unsupported.

### Resources and memory

Buffers and textures are backend-owned Vulkan resources. VMA remains private.
Frame, persistent, staging, and readback arenas expose checked CPU/GPU spans.
Resource destruction invalidates the public token immediately and retires the
backend object after in-flight work completes.

### Commands and queues

Public queue kinds map to backend-selected Vulkan queues. Command lists are
owner-bearing aliases of a generation-checked backend record:

```text
RECORDING -> RECORDING_RENDER_PASS -> RECORDING -> EXECUTABLE -> SUBMITTING -> consumed
```

A successful submit consumes every alias. A pre-submit fault restores the
record. Recording contexts provide per-thread command pools.

### Descriptors

`TextureHandle` owns an image. `TextureIndex` and `SamplerIndex` identify
entries in the compatibility descriptor heap. The backend uses descriptor
indexing by default and supports opt-in descriptor buffers when available.

### Synchronization

Compatibility barriers name resources and explicit texture layouts. Texture
layout changes are staged per command list and committed only by successful
submission. Render-pass boundaries add no implicit synchronization.

### Pipelines and shaders

Shaders are validated through SPIR-V reflection. Compute and graphics pipeline
creation uses explicit compatibility descriptors and a device-owned cache.
Draw and dispatch commands pass root GPU addresses through the generated shader
ABI.

### Presentation

Swapchains are optional. The core library accepts platform-neutral native
surface handles; SDL3 helpers remain in `gpu.c3l-samples`.

## Backend boundary

Public signatures contain no `vk::`, `vma::`, or SDL types. The private Vulkan
backend owns feature chains, queue families, native handles, layouts, allocator
objects, and loader calls.

The compatibility runtime requires Vulkan 1.3 semantics. Vulkan 1.2 plus
extensions is not currently supported.

## Verification

The repository gates:

- import-only execution with zero backend creation calls;
- compile-pass and compile-fail profile-boundary fixtures;
- CPU and shader ABI tests;
- Linux and Windows Vulkan sweeps;
- generated ABI and public-documentation scans;
- benchmark target builds and a fixed-method baseline.

See [Testing](testing.md) for commands and [Compatibility](compatibility.md)
for the frozen public contract and migration rules.