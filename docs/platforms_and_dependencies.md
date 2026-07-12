# gpu.c3l Platforms and Dependencies

## 1. Library identity

```text
Library package: gpu.c3l
Public module:   gpu
Manifest name:   provides "gpu"
Language target: C3 0.8.0
Primary backend: Vulkan 1.3
```

## 2. Development and consumer boundaries

The development checkout vendors the backend bindings separately:

```text
gpu.c3l/
└── lib/
    ├── vk.c3l/
    ├── vma.c3l/
    └── spvreflect.c3l/
```

Its root `manifest.json` and white-box test projects name `vk`, `vma`, and
`spvreflect` directly. That shape is for contributors; C3 0.8.0 does not
activate those packages transitively for an external project.

The supported consumer artifact is generated for one target. Its manifest
provides `gpu`, compiles package-owned copies of the complete backend source
closure, declares no C3 package dependencies, and owns native link and CRT
metadata. A consumer needs one search root and one dependency:

```json
{
  "dependency-search-paths": [ "lib/gpu-package" ],
  "dependencies": [ "gpu" ]
}
```

Supported package targets are `linux-x64` and `windows-x64`. Package and
runtime verification reject a bundle whose recorded target does not match the
requested target.

## 3. Backend bindings

The generated package compiles the existing modules without exposing them as
supported consumer APIs:

```text
vk.c3l         -> module vk, Vulkan calls
vma.c3l        -> module vma, Vulkan Memory Allocator
spvreflect.c3l -> module spvreflect, SPIR-V reflection
```

No public `gpu` function or struct exposes `vk::`, `vma::`, or
`spvreflect::` types. The repository keeps the bindings as submodules so
contributors can inspect, test, and update them independently. The package
copies only the explicitly allowed production sources.

## 4. SDL3 ownership

SDL3 belongs to applications and the `gpu.c3l-samples` repository, not this
library. The samples repository retains a pinned gpu.c3l source checkout as
package input, materializes its generated GPU package, and owns SDL separately:

```text
gpu.c3l-samples/
└── lib/
    ├── gpu.c3l/        pinned source input
    ├── gpu-package/
    │   └── gpu.c3l/    generated consumer package
    └── sdl3.c3l/
```

```json
{
  "dependency-search-paths": [ "lib/gpu-package", "lib" ],
  "dependencies": [ "gpu", "sdl3" ]
}
```

The dependency name is `sdl3`; source imports its module as `sdl`. The GPU
package does not check or stage `SDL3.dll`.

## 5. Native link and runtime requirements

### Vulkan

Consumers must install a Vulkan 1.3 loader and a suitable driver:

```text
Linux:   libvulkan.so.1
Windows: vulkan-1.dll
```

The package owns the corresponding link declarations and Windows import
library, but it never redistributes or stages a loader or driver from an SDK or
driver installation.

### VMA and SPIRV-Reflect

The `linux-x64` package contains byte-locked
`libVulkanMemoryAllocator.a` and `libspvreflect.a` artifacts. Its system link
requirements include the Vulkan loader and required C++ runtime.

The `windows-x64` package contains byte-locked `spvreflect.lib` and
`vulkan-1.lib`. It builds `VulkanMemoryAllocator.lib` from locked source and
build inputs, then records the normalized toolchain identity and generated
archive SHA-256 in `artifact-manifest.json`.

Windows VMA is compiled for the release dynamic CRT (`/MD`). Package
verification inspects the archive directives, the generated manifest selects
`wincrt: dynamic`, and CI inspects the final consumer executable's PE imports
for release CRT DLLs while rejecting debug CRT DLLs. Consumers do not repeat a
CRT setting in their project; the package owns it.

### Runtime metadata

Each bundle includes `runtime.json` and `tools/runtime.py`. The runtime tool:

- checks package target identity and normal Vulkan-loader discovery;
- reports the Windows dynamic release CRT as a declared, non-authoritatively
  discovered system prerequisite; and
- hash-checks and stages only package-owned runtime files.

Current GPU packages own no runtime files, so staging is a successful no-copy
operation that prints the remaining Vulkan and CRT prerequisites. It never
harvests a Vulkan loader or CRT from SDK, driver, or toolchain directories.

## 6. Package locking and provenance

`packaging/package.json` is the explicit source, public shader-ABI asset,
license, native-artifact, and Windows-build-input allowlist. Binding-source
and asset globs are forbidden. Generated packages include the locked
`include/shaders/descriptor_heap.glsl` and
`include/shaders/generated/shader_abi.glsl` assets.
`packaging/package-lock.json` binds:

```text
package format and C3 version
binding submodule commits
every allowed source, public asset, and build-input hash
committed native-artifact hashes
Windows VMA upstream header, wrapper, build script, and SDK identity
```

For Windows assembly, the packager derives the actual MSVC version by running
`cl` and validates `VULKAN_HEADERS` or `VULKAN_SDK` by parsing
`include/vulkan/vulkan_core.h`. The headers must identify Vulkan 1.3 and
header revision 290; caller-provided provenance strings are not accepted.

An intentional dependency update refreshes the lock in the same review.
Normal assembly and CI use the read-only lock check.

Assembly writes a temporary target directory and publishes it only after
verification. `artifact-manifest.json` records the target, locked-input digest,
normalized toolchain identity, every payload hash, and an aggregate digest over
sorted payload path/hash pairs. Verification rejects stale, missing, extra,
duplicate, non-normalized, or mismatched payloads. Committed native inputs stay
byte-locked. The generated Windows VMA archive records its actual output hash
instead of pretending separate conforming toolchains produce identical bytes.

## 7. Platform support

| Platform | Consumer package | Package fixture | SDL3 samples | Notes |
|---|---:|---:|---:|---|
| linux-x64 | Supported | Blocking | Blocking in samples repo | Primary target; system Vulkan loader/driver. |
| windows-x64 | Supported | Blocking | Blocking in samples repo | Dynamic release CRT; system Vulkan loader/driver. |
| linux-aarch64 | Deferred | No | Deferred | Requires locked native artifacts and CI. |
| macOS | Deferred | No | Deferred | Requires a portability stack; not first scope. |
| wasm | Out of scope | No | No | Vulkan backend not applicable. |

## 8. Developer setup

The contributor checkout remains deliberately white-box:

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l
python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
c3c test unit --path test/cpu
c3c build smoke --path test
./test/build/smoke
```

The test harness resolves separate bindings from `lib/` by their real
directory names. This arrangement is useful for backend work but is not the
supported external-consumer contract.

On Windows, install c3c 0.8.0, fetch its MSVC SDK, install MSVC Build Tools and
the Vulkan SDK, and run commands from Git Bash after loading the MSVC
environment. The development harness builds VMA through
`lib/vma.c3l/scripts/build-vma-windows.sh`. Headless CI uses mesa-dist-win;
elevated runners register its ICD under
`HKLM\SOFTWARE\Khronos\Vulkan\Drivers` because elevated Vulkan loaders ignore
`VK_DRIVER_FILES`.

## 9. Consumer package workflow

From a recursive source checkout, the package workflow is:

```text
1. Check the source/build-input lock.
2. Assemble exactly one target package.
3. Verify its artifact manifest and aggregate payload digest.
4. Run the included runtime target/prerequisite check.
5. Stage package-owned runtime files to the application directory.
6. Build the application with one package search root and dependency gpu.
```

The exact commands are shown by `python scripts/package_gpu.py --help` and in
`docs/getting_started.md`. The runtime checker and stager are invoked from the
generated package so they operate on the package's own metadata.

Application shaders remain consumer-owned. The library publishes shader-side
ABI includes under `include/shaders/`; compiling application shaders to SPIR-V
is the consumer's build step.

## 10. Environment variables

Development and CI may use:

```text
VULKAN_SDK       Vulkan headers and developer tools
VMA_INCLUDE      pinned VMA header root for the Windows VMA build
VK_DRIVER_FILES  explicit Vulkan ICD selection in non-elevated environments
VK_LAYER_PATH    explicit validation-layer search path when needed
```

These are build/test inputs, not public API configuration.

## 11. Continuous integration

The existing `linux`, `windows`, and `docs-walkthrough` jobs continue to test
the development tree. Two additional jobs prove the consumer boundary from a
clean recursive checkout:

```text
package-consumer-linux (blocking):
    lock and fixture-policy checks
    assemble and verify linux-x64
    runtime check and no-op stage
    build and run the gpu-only device + embedded-SPIR-V shader fixture
    verify the aggregate payload digest

package-consumer-windows (blocking):
    lock and fixture-policy checks
    build VMA from locked inputs, then assemble and verify windows-x64
    verify VMA /MD directives
    runtime check and no-op stage
    build the unchanged gpu-only fixture without a fixture CRT override
    require release CRT PE imports and reject debug CRT imports
    run the device/shader fixture under lavapipe
    verify the aggregate payload digest
```

The existing Windows backend Vulkan sweep remains advisory because of
mesa-dist-win variability. The Windows package-consumer proof is blocking.

## 12. Acceptance criteria

Dependency and platform setup is acceptable when:

```text
external consumers name only gpu and one generated-package search root
generated packages are target-scoped, lock-checked, and manifest-verified
linux-x64 and windows-x64 fixtures create/destroy a device and reflected shader
windows-x64 proves VMA /MD, package dynamic CRT selection, and final PE imports
runtime staging copies only hash-checked package-owned files
Vulkan loader/driver, CRT, and SDL remain system/application-owned
the samples repository consumes generated gpu plus its app-owned sdl3 package
public API signatures contain no vk::, vma::, spvreflect::, or sdl:: types
```
