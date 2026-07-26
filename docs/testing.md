# Testing

The test strategy is compiler- and behavior-first:

- compile real consumers of the public `gpu` bundle and surface modules;
- run CPU C3 tests for public contracts and shader ABI layout;
- regenerate and check shipped shader ABI artifacts;
- compile every GLSL and SPIR-V assembly fixture;
- run consolidated C3 tests against Vulkan with validation enabled;
- exercise output/readback, ownership, rollback, synchronization, WSI, and
  optional capabilities through the ordinary C3 test framework.

The repository does not parse source text, generated documentation, CI
configuration, assembly, or benchmark output as a correctness policy.

## Toolchain and dependencies

Use C3 0.8.0 and initialize the binding submodules:

```sh
c3c --version
git submodule update --init --recursive
```

Vulkan tests require a Vulkan 1.3 loader, the VMA static library described in
[Platforms and dependencies](platforms_and_dependencies.md), SPIRV-Reflect,
and a driver satisfying the baseline in
[Vulkan backend](vulkan_backend.md). Install `glslc` and `spirv-as` before
building shader fixtures.

On headless Linux, select lavapipe explicitly:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Keep the Khronos validation layer installed. A real hardware ICD may be used
for additional local coverage.

## Public consumer and CPU checks

The canonical getting-started consumer lives under
[`examples/getting_started`](../examples/getting_started/). Build its shader,
then compile and run the project from a normal checkout named `gpu.c3l`:

```sh
python3 scripts/build_shaders.py
c3c build hello_gpu --path examples/getting_started
./examples/getting_started/build/hello_gpu
```

The consumer imports only `gpu` and resolves the shipped bundle plus its
declared `vk`, `vma`, and `spvreflect` dependencies. The separate
`gpu.c3l-samples` repository provides broader downstream consumer coverage.

Run the CPU compile and test targets:

```sh
c3c run import_gpu --path test/cpu
c3c build canonical_gpu_surface --path test/cpu
c3c build import_surface_win32 --path test/cpu
c3c build import_surface_wayland --path test/cpu
c3c build import_surface_x11 --path test/cpu
c3c build span_data_operations --path test/cpu
c3c build sampler_operations --path test/cpu
c3c build texture_view_operations --path test/cpu
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
```

These are ordinary C3 consumers. Compiler visibility and type checking—not a
symbol inventory—establish what the public modules expose.

## Generated shader ABI

The ABI generator produces shipped C3 and GLSL artifacts from schemas. Check
the generator and committed outputs with:

```sh
c3c test unit --path tools/gen_shader_abi
python3 -B -m unittest scripts.test_gen_abi
python3 scripts/gen_abi.py --check
```

After changing an ABI schema, regenerate the outputs and shader fixtures:

```sh
python3 scripts/gen_abi.py
python3 scripts/build_shaders.py
```

`scripts/build_shaders.py` compiles the test shaders, assembles deterministic
SPIR-V fixtures, and builds the getting-started shader. Generated `.spv` files
are ignored.

## Link proof

The whitebox test project directly compiles the library implementation and
private backend so it can exercise narrow test seams. Confirm it links and
runs:

```sh
c3c build smoke --path test
./test/build/smoke
```

This is separate from the real bundle-consuming getting-started project.

## Consolidated Vulkan tests

The Vulkan suite is grouped by runtime needs rather than by historical feature
increments:

```sh
c3c test vk_core --path test --test-show-output
c3c test vk_wsi --path test --test-show-output
c3c test vk_optional_generated_work --path test --test-show-output
```

- `vk_core` covers device selection and creation, allocations, spans,
  textures/views/samplers, reflection and root ABI, pipelines/cache, command
  recording and lifecycle, graphics/compute/transfer output, depth,
  threading, queues, submission/completion, diagnostics, rollback, and
  validation policy.
- `vk_wsi` covers swapchain configuration, acquisition, present, resize,
  ownership, result mapping, and WSI diagnostics without requiring a native
  window.
- `vk_optional_generated_work` covers indirect and generated dispatch/draw
  behavior, scratch reservation/exhaustion, caller-owned spans, and observable
  output when the driver supports the capability.

Capability-specific tests query support in C3 and handle an unavailable
feature explicitly. CI does not parse `EXERCISED` text or require a particular
message format.

Some tests use narrow private helpers or operation-local seams to observe
otherwise hidden transactional behavior. Those are test implementation
details, not public interfaces or output schemas.

## Manual benchmarks

CI may build the manual benchmark executables to catch compiler drift, but
does not execute or parse them. See [Performance](performance.md) for direct
commands and interpretation. In particular, `pipeline_cache_bench` records
complete cached `GraphicsState` packets across raster permutations; its timing
is advisory and is not compared with raster-only measurements.

## CI

`.github/workflows/ci.yml` runs Linux and Windows jobs. Both retain their
platform-specific C3, Vulkan, VMA, shader-tool, and lavapipe setup, then invoke
the same direct categories:

```text
consumer and surface compilation
CPU unit and shader ABI tests
generator tests and ABI drift check
shader build
manual benchmark builds
smoke link/run
vk_core
vk_wsi
vk_optional_generated_work
```

Linux also uploads `c3c docgen` output as an unparsed whole-compile reference.
The artifact is informational; no script treats its symbols or wording as a
second API specification.

## Full local sweep

From the repository root:

```sh
c3c run import_gpu --path test/cpu
c3c build canonical_gpu_surface --path test/cpu
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
c3c test vk_core --path test --test-show-output
c3c test vk_wsi --path test --test-show-output
c3c test vk_optional_generated_work --path test --test-show-output
git diff --check
```
