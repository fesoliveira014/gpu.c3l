# gpu.c3l Platforms and Dependencies

## 1. Library identity

```text
Library package: gpu.c3l
Public modules:  gpu
                 gpu::surface::{win32,wayland,x11}
Manifest name:   provides "gpu"
Language target: C3 0.8.0
Primary backend: Vulkan 1.3
```

## 2. Required library dependencies

The shipped library depends on:

```text
vk.c3l
vma.c3l
spvreflect.c3l
```

Vendored layout:

```text
gpu.c3l/
└── lib/
    ├── vk.c3l/
    ├── vma.c3l/
    └── spvreflect.c3l/
```

Manifest shape (shipped):

```json
{
  "provides": "gpu",
  "linklib-dir": "linked-libs",
  "sources": [ "gpu/gpu.c3", "gpu/gpu.c3i", "gpu/types.c3", /* ...all public files... */ "gpu/vk/**" ],
  "targets": {
    "linux-x64":   { "dependencies": [ "vk", "vma", "spvreflect" ] },
    "windows-x64": { "dependencies": [ "vk", "vma", "spvreflect" ] }
  }
}
```

`manifest.json` accepts no top-level `dependency-search-paths` (a
`project.json` key): dependencies are declared per-target and resolved by the
consumer's search path.

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
import gpu::surface::win32; // select the host platform module
```

The package/dependency name is `sdl3`; the C3 module name is `sdl`. This
library repository carries no SDL3 dependency or submodule. A consumer converts
SDL's native window properties to the selected module's distinct handle types
when creating a runtime-owned `Surface`.

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

`vk.c3l` handles link declarations according to its manifest. Headless use
requires Vulkan 1.3, `VK_EXT_extended_dynamic_state3`, and
`dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. Presentation additionally
requires
`VK_KHR_get_surface_capabilities2` and `VK_EXT_surface_maintenance1` on the
instance plus `VK_KHR_swapchain` and
`VK_EXT_swapchain_maintenance1` on the device; support queries reject adapters
that cannot provide the private present-fence lifecycle.

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
| windows-x64 | Required | Required | Desired | CI builds VMA and runs the headless matrix on mesa-dist-win. |
| linux-aarch64 | Deferred | Deferred | Deferred | Requires VMA static lib and Vulkan ICD. |
| macOS | Deferred | Deferred | Deferred | Vulkan requires portability stack; not first scope. |
| wasm | Out of scope | No | No | Vulkan backend not applicable. |

## 8. Developer setup

Library + tests (this repository):

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l
python3 scripts/gen_abi.py --check && python3 scripts/build_shaders.py
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
```

The test harness compiles the library sources directly and resolves the
vendored bindings from `lib/` by real directory name — no symlinks, no
requirement on the checkout directory's name.

Samples (consumer path):

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l-samples
```

### windows-x64 setup

Use Git Bash from a Visual Studio x64 developer shell. Install C3 0.8.0, the
Vulkan SDK, and MSVC Build Tools, then run:

```sh
c3c fetch-sdk windows
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l

git clone --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers.git /tmp/vh
git clone --depth 1 --branch v3.3.0 https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git /tmp/vma
mkdir -p /tmp/vma-inc/vma
cp /tmp/vma/include/vk_mem_alloc.h /tmp/vma-inc/vma/

VULKAN_HEADERS=$(cygpath -w /tmp/vh) VMA_INCLUDE=$(cygpath -w /tmp/vma-inc) sh lib/vma.c3l/scripts/build-vma-windows.sh
cp "$VULKAN_SDK/Lib/vulkan-1.lib" lib/vma.c3l/linked-libs/windows-x64/

python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
c3c build smoke --path test
./test/build/smoke.exe
```

Use VMA 3.3.0. Older SDK-bundled VMA headers do not match the binding.

For headless tests, set `VK_DRIVER_FILES` to a lavapipe ICD. Elevated shells
may require registering the ICD under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`
because the Vulkan loader can ignore that environment variable.

### VMA static library artifact policy

`vma.c3l` ships the Linux static library. Windows users build the library with
`build-vma-windows.sh`. Both build scripts verify the C ABI with a size probe.

## 9. Build organization

Recommended separation:

```text
manifest.json        -> shipped library metadata
project.json         -> optional developer workspace if useful
test/project.json    -> test harness
```

The shipped library manifest should not pull sample/test sources or SDL3 into consumers.

## 10. Shader toolchain dependencies

The library consumes SPIR-V only. The shipped shader build uses `glslc`
(`scripts/build_shaders.py`); linux CI installs it via apt, windows CI uses
the Vulkan SDK's copy. `glslangValidator` appears only in the getting-started
walkthrough.

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

`.github/workflows/ci.yml` — one workflow, three jobs, `bash` on all:

```text
linux (ubuntu-24.04, blocking):
    pinned c3c release, glslc, mesa-vulkan-drivers (lavapipe)
    generator unit tests, gen_abi.py --check, build_shaders.py
    full test-target sweep under VK_DRIVER_FILES (any failure fails the job)
    c3c docgen API reference, uploaded as the api-reference artifact

docs-walkthrough (ubuntu-24.04, blocking):
    executes docs/getting_started.md verbatim via scripts/run_doc.py on a
    bare runner — the walkthrough is its own regression test

windows (windows-2022, blocking except the last step):
    pinned c3c release, Vulkan SDK, MSVC env
    VMA static lib built in-job (build-vma-windows.sh)
    generator unit tests, gen_abi.py --check, build_shaders.py
    link proof (smoke) + pure-CPU test targets
    lavapipe (mesa-dist-win) registered under the HKLM Vulkan driver key,
    then the Vulkan sweep — advisory, continue-on-error
```

The c3c version is pinned once, in the workflow's `C3C_VERSION` env var
(currently 0.8.0). Tool downloads are cached by version key. Compiler upgrades
are a one-line workflow change plus this document.

## 13. Dependency acceptance criteria

Dependency setup is acceptable when:

```text
gpu.c3l consumers depend on gpu, vk, vma, and spvreflect only as required by manifest
SDL3 lives only in the gpu.c3l-samples repository
linux-x64 builds with vendored vk.c3l, vma.c3l, and spvreflect.c3l
the samples repository can import gpu and sdl through vendored submodules
public API signatures contain no vk::, vma::, or sdl:: types
platform setup steps are documented
```
