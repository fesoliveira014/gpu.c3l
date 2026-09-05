# Testing

Tests are ordinary C3 programs and `@test` targets. Nothing parses source
text, generated docs, CI configuration, or benchmark output as a policy.

Layers:

1. compile real consumers of the public bundle and surface modules;
2. CPU tests for public contracts and shader ABI layout;
3. regenerate and check shipped ABI artifacts;
4. compile every GLSL and SPIR-V assembly fixture;
5. Vulkan tests with validation layers on.

## Toolchain

```sh
c3c --version                         # 0.8.3
git submodule update --init --recursive
```

Vulkan targets need a Vulkan 1.3 loader, the VMA static library, SPIRV-Reflect,
and a driver meeting the [required profile](../features_and_limitations.md#required-device-profile).
Shader fixtures need `glslc` and `spirv-as`.

Headless Linux: select lavapipe and keep the Khronos validation layer
installed.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

## Consumer and CPU checks

The getting-started program under `examples/getting_started` is the canonical
consumer. It imports only `gpu` and resolves the shipped bundle plus its
declared dependencies. Build its shader, then build and run it from a checkout
named `gpu.c3l`:

```sh
python3 scripts/build_shaders.py
c3c build hello_gpu --path examples/getting_started
./examples/getting_started/build/hello_gpu
```

Release archive shape, its runtime-only dependency boundary, and relative
links in the bundled consumer docs (a link to a file outside the bundle, such
as `examples/`, fails; use a GitHub URL instead):

```sh
python3 -B -m unittest scripts.test_package_release -v
python3 scripts/package_release.py --version 0.0.0-ci --target linux-x64 --output-dir dist
```

CI builds both target archives, assembles them into a throwaway `lib/gpu.c3l`
consumer layout, and builds and runs the getting-started program from
`project.release.json` without touching the repository checkout.

CPU targets under `test/cpu` compile the public module against a stub backend
with no native libraries:

```sh
c3c run import_gpu --path test/cpu
c3c build import_surface_win32 --path test/cpu
c3c build import_surface_wayland --path test/cpu
c3c build import_surface_x11 --path test/cpu
c3c build span_data_operations --path test/cpu
c3c build sampler_operations --path test/cpu
c3c build texture_view_operations --path test/cpu
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
```

Compiler visibility and type checking define the public surface; there is no
symbol inventory.

## Shader ABI

```sh
c3c test unit --path tools/gen_shader_abi
python3 -B -m unittest scripts.test_gen_abi
python3 scripts/gen_abi.py --check
```

After changing a schema:

```sh
python3 scripts/gen_abi.py
python3 scripts/build_shaders.py
```

`build_shaders.py` compiles test shaders, assembles SPIR-V fixtures, and
builds the getting-started shader. `.spv` outputs are git-ignored.

## Link proof

The whitebox project under `test` compiles the library and backend directly
so tests can reach private seams:

```sh
c3c build smoke --path test
./test/build/smoke
```

## Vulkan tests

```sh
c3c test vk_sparse_texture --path test --test-show-output
c3c test vk_sparse_bind --path test --test-show-output
c3c test vk_core --path test --test-show-output
c3c test vk_wsi --path test --test-show-output
c3c test vk_optional_generated_work --path test --test-show-output
c3c test vk_ray_tracing --path test --test-show-output
```

| Target | Covers |
|---|---|
| `vk_sparse_texture` | sparse descriptor validation, flags, requirement translation, transaction rollback, cached queries, capability-gated image lifecycle |
| `vk_sparse_bind` | tile and tail geometry, unbind, allocation compatibility and overlap, allocation-failure rollback, timeline chaining, result mapping, retention and retirement, cross-thread lock boundaries, capability-gated bind/use/unbind with readback |
| `vk_core` | device selection, allocations, spans, textures, views, samplers, reflection and root ABI, pipelines and cache, command lifecycle, graphics/compute/transfer output, depth, threading, queues, submission and completion, timestamps, diagnostics, rollback, validation policy |
| `vk_wsi` | swapchain configuration, acquire, present, resize, ownership, result mapping, WSI diagnostics; no native window needed |
| `vk_optional_generated_work` | indirect and generated dispatch and draw, reservation and exhaustion, caller-owned spans, output when the driver supports it |
| `vk_ray_tracing` | on a capable device: BLAS and TLAS builds, clone, SBT packing, direct and indirect trace, dynamic stack size, indirect build ranges. On an adapter without ray-tracing pipelines it fails with `UNSUPPORTED_FEATURE`; that is not a skip. |

Capability-gated tests query support in C3 and report an unavailable feature
explicitly. Some tests use private seams to observe transactional behavior;
those seams are not public interfaces.

Timestamp tests keep two contracts distinct: `cmd_resolve_timestamps` records
a device-side availability wait; `read_timestamps` requests no wait and
returns `DEVICE_BUSY` with unspecified output when values are not ready.
Neither validation policy tracks per-slot reset or write history, so tests do
not assert diagnostics for it.

## Benchmarks

CI may build benchmark executables to catch compiler drift; it does not run
them. See [benchmarking](benchmarking.md).

## CI

`.github/workflows/ci.yml` runs Linux and Windows jobs with platform-specific
setup and then the same steps:

```text
consumer and surface compilation
CPU unit and shader ABI tests
generator tests and ABI drift check
shader build
benchmark builds
smoke link and run
vk_core, vk_wsi, vk_optional_generated_work, vk_ray_tracing
```

Linux also uploads `c3c docgen` output from the whitebox project as an
informational artifact.

## Full local sweep

```sh
c3c run import_gpu --path test/cpu
c3c build import_surface_win32 --path test/cpu
c3c build import_surface_wayland --path test/cpu
c3c build import_surface_x11 --path test/cpu
c3c build span_data_operations --path test/cpu
c3c build sampler_operations --path test/cpu
c3c build texture_view_operations --path test/cpu
c3c build command_wrapper_bench --path test/cpu -O1
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
c3c test unit --path tools/gen_shader_abi
python3 -B -m unittest scripts.test_gen_abi
python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
c3c build hello_gpu --path examples/getting_started
./examples/getting_started/build/hello_gpu
for target in resource_create_bench upload_throughput_bench command_path_baseline_bench lifecycle_bench pipeline_cache_bench async_overlap_bench; do
    c3c build "$target" --path test -O1
done
c3c build smoke --path test
./test/build/smoke
c3c test vk_sparse_texture --path test --test-show-output
c3c test vk_sparse_bind --path test --test-show-output
c3c test vk_core --path test --test-show-output
c3c test vk_wsi --path test --test-show-output
c3c test vk_optional_generated_work --path test --test-show-output
c3c test vk_ray_tracing --path test --test-show-output
git diff --check
```
