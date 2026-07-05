# gpu.c3l Platforms and Dependencies

## 1. Library identity

```text
Library package: gpu.c3l
Public module:   gpu
Manifest name:   provides "gpu"
Language target: C3 0.8.0
Primary backend: Vulkan 1.3
```

## 2. Required library dependencies

The shipped library depends on:

```text
vk.c3l
vma.c3l
```

Expected vendored layout:

```text
gpu.c3l/
└── lib/
    ├── vk.c3l/
    └── vma.c3l/
```

Manifest concept:

```json
{
  "provides": "gpu",
  "dependency-search-paths": [ "lib" ],
  "dependencies": [ "vk", "vma" ]
}
```

Exact C3 manifest syntax should be verified against C3 0.8.0 during implementation.

## 3. Vulkan binding: vk.c3l

`vk.c3l` is the Vulkan binding used by the backend.

Backend import:

```c3
import vk;
```

Public API rule:

```text
No public gpu function or struct exposes vk:: types.
```

The Vulkan backend should call Vulkan through this binding only. Do not mix multiple Vulkan loaders or bindings in the same backend.

## 4. Memory allocator binding: vma.c3l

`vma.c3l` provides Vulkan Memory Allocator bindings.

Backend import:

```c3
import vma;
```

The binding depends on `vk`. The `gpu` library depends on both `vk` and `vma`.

Backend use:

```text
create vma::Allocator once per gpu::Device
create buffers through allocator.try_create_buffer
create images through allocator.try_create_image
map/flush/invalidate through VMA wrappers
query heap budgets/statistics through VMA wrappers
use vma::VirtualBlock for CPU-side suballocation of persistent arenas
```

Public API rule:

```text
No public gpu function or struct exposes vma:: types.
```

## 5. SDL3 binding: sdl3.c3l

SDL3 belongs to the `gpu.c3l-samples` repository, which vendors it alongside
this library:

```text
gpu.c3l-samples/
└── lib/
    ├── gpu.c3l/        (this repository, pinned submodule; nested lib/ holds vk/vma/spvreflect)
    └── sdl3.c3l/
```

Samples project dependency:

```json
{
  "dependency-search-paths": [ "lib", "lib/gpu.c3l/lib" ],
  "dependencies": [ "gpu", "vk", "vma", "spvreflect", "sdl3" ]
}
```

Sample source import:

```c3
import gpu;
import sdl;
```

The package/dependency name is `sdl3`; the C3 module name is `sdl`. This
library repository carries no SDL3 dependency or submodule.

Public API rule:

```text
No core gpu public function requires sdl::Window.
```

If convenience helpers are needed, put them in samples or an optional helper module rather than the core API.

## 6. Static/native library requirements

### Vulkan

The system must provide a Vulkan loader:

```text
Linux:   libvulkan.so.1
Windows: vulkan-1.dll / vulkan-1 import library
```

`vk.c3l` handles link declarations according to its manifest.

### VMA

`vma.c3l` requires a compiled VMA static library for the target under its `linked-libs/<target>/` directory.

Initial support target:

```text
linux-x64
```

Second target:

```text
windows-x64
```

Other targets require building/providing the VMA static library.

### SDL3

`sdl3.c3l` ships or documents the SDL3 native library for the samples
repository's targets. Nothing in this repository links SDL3.

## 7. Platform support plan

| Platform | Library build | Headless Vulkan tests | SDL3 samples | Notes |
|---|---:|---:|---:|---|
| linux-x64 | Required first | Required first | Required first (samples repo) | Primary development target. |
| windows-x64 | Required second | Desired | Desired | Needs VMA static lib and SDL3 native setup. |
| linux-aarch64 | Deferred | Deferred | Deferred | Requires VMA static lib and Vulkan ICD. |
| macOS | Deferred | Deferred | Deferred | Vulkan requires portability stack; not first scope. |
| wasm | Out of scope | No | No | Vulkan backend not applicable. |

## 8. Developer setup

Library + tests (this repository):

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l
./scripts/gen_abi.sh --check && ./scripts/build_shaders.sh
c3c test unit --path test
```

The test harness compiles the library sources directly and resolves the
vendored bindings from `lib/` by real directory name — no symlinks, no
requirement on the checkout directory's name.

Samples (consumer path):

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l-samples
```

## 9. Build organization

Recommended separation:

```text
manifest.json        -> shipped library metadata
project.json         -> optional developer workspace if useful
test/project.json    -> test harness
```

The shipped library manifest should not pull sample/test sources or SDL3 into consumers.

## 10. Shader toolchain dependencies

First implementation should consume SPIR-V. The shader build tool can be one of:

```text
glslangValidator
shaderc
custom C3 wrapper around shaderc.c3l if adopted later
```

Keep shader compilation outside the core runtime unless runtime compilation becomes an explicit feature.

Shader ownership: the shipped library contains no application shaders. Shader programs belong to the consuming project (and, in this repository, to each sample/test). The library publishes only shader-side ABI includes under `include/shaders/` for consumers to `#include`. Compiling application shaders to SPIR-V is therefore the consumer's build step, not part of the library manifest.

## 11. Environment variables

Do not require environment variables for normal library use. Tests may document optional environment variables for selecting a Vulkan ICD.

Examples:

```text
VK_ICD_FILENAMES
VK_INSTANCE_LAYERS
```

These belong in testing documentation, not in core code.

## 12. Dependency acceptance criteria

Dependency setup is acceptable when:

```text
gpu.c3l consumers depend on gpu, vk, and vma only as required by manifest
SDL3 lives only in the gpu.c3l-samples repository
linux-x64 builds with vendored vk.c3l and vma.c3l
the samples repository can import gpu and sdl through vendored submodules
public API signatures contain no vk::, vma::, or sdl:: types
platform setup steps are documented
```
