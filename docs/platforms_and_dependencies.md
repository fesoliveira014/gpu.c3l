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

### windows-x64 setup

The documented shell for `scripts/*.sh` on Windows is git-bash (ships with Git
for Windows); CI runs the same scripts under `shell: bash`.

```sh
# 1. c3c: unpack the pinned release and put it on PATH, then fetch the MSVC
#    link libraries once (c3c discovers msvc_sdk beside its own binary):
#    https://github.com/c3lang/c3c/releases/download/v0.8.0/c3-windows.zip
c3c fetch-sdk windows && cp -r ~/AppData/Local/c3/msvc_sdk <c3c-install-dir>/
# 2. Vulkan SDK (headers, glslc, vulkan-1 import lib). A GPU driver provides
#    the vulkan-1.dll loader; headless machines get it from LunarG's
#    VulkanRT-<ver>-Components.zip.
# 3. MSVC build tools (cl/lib) — any Visual Studio or Build Tools install
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l
sh lib/vma.c3l/scripts/build-vma-windows.sh   # from a shell with cl/lib on PATH; uses VULKAN_SDK
cp "$VULKAN_SDK/Lib/vulkan-1.lib" lib/vma.c3l/linked-libs/windows-x64/
./scripts/gen_abi.sh --check && ./scripts/build_shaders.sh
c3c build smoke --path test && ./test/build/smoke.exe
c3c test unit --path test && c3c test shader_abi --path test
```

Notes proven by CI: harness `project.json`s declare no `target` (the host is
correct on both platforms — a pinned `linux-x64` makes windows c3c
cross-compile); `.gitattributes` enforces LF so golden tests and the drift
gate compare byte-exact; the windows Vulkan test sweep is advisory (see the
tracking issue for mesa-dist-win lavapipe failures).

Headless Vulkan tests on Windows use lavapipe from
[mesa-dist-win](https://github.com/pal1000/mesa-dist-win) via
`VK_DRIVER_FILES=<extracted>/x64/lvp_icd.x86_64.json`. They are advisory: CI
runs them non-blocking, and behavior may differ from linux lavapipe.

### VMA static library artifact policy

Per supported target, `vma.c3l` either ships a prebuilt static library in
`linked-libs/<target>/` (linux-x64 today) or provides a build script that
produces it from Vulkan headers alone (`build-vma.sh`, `build-vma-windows.sh`),
both guarded by the ABI size probe. The end-state for windows-x64 is a
committed prebuilt `.lib` mirroring linux; until a maintainer blesses one, CI
and developers build it in place with the script.

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

## 12. Continuous integration

`.github/workflows/ci.yml` — one workflow, two jobs, `bash` on both:

```text
linux (ubuntu-24.04, blocking):
    pinned c3c release, glslc, mesa-vulkan-drivers (lavapipe)
    generator unit tests, gen_abi.sh --check, build_shaders.sh
    full test-target sweep under VK_DRIVER_FILES (any failure fails the job)

windows (windows-2022, blocking except the last step):
    pinned c3c release, Vulkan SDK, MSVC env
    VMA static lib built in-job (build-vma-windows.sh)
    generator unit tests, gen_abi.sh --check, build_shaders.sh
    link proof (smoke) + pure-CPU test targets
    lavapipe (mesa-dist-win) Vulkan sweep — advisory, continue-on-error
```

The c3c version is pinned once, in the workflow's `C3C_VERSION` env var
(currently 0.8.0). Tool downloads are cached by version key. Compiler upgrades
are a one-line workflow change plus this document.

## 13. Dependency acceptance criteria

Dependency setup is acceptable when:

```text
gpu.c3l consumers depend on gpu, vk, and vma only as required by manifest
SDL3 lives only in the gpu.c3l-samples repository
linux-x64 builds with vendored vk.c3l and vma.c3l
the samples repository can import gpu and sdl through vendored submodules
public API signatures contain no vk::, vma::, or sdl:: types
platform setup steps are documented
```
