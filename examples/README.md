# Examples

Small headless programs that consume the library the way an application does.
Windowed programs live in the
[samples repository](https://github.com/fesoliveira014/gpu.c3l-samples).

| Directory | Target | Shows |
|---|---|---|
| `getting_started/` | `hello_gpu` | Manual runtime, device, memory, pipeline, root data, dispatch, readback. The program [getting started](../docs/getting_started.md) walks through. |
| `device_context/` | `device_context` | The same work through `gpu::util::DeviceContext`: one call for runtime, device, queue, and command allocator. |

## Build from this checkout

Each example's `project.json` resolves `gpu` from the repository root and the
bindings from `lib/`. Install the VMA static libraries and compile the shaders
once, then build and run:

```sh
python3 scripts/fetch_vma_libs.py
python3 scripts/build_shaders.py
c3c run hello_gpu --path examples/getting_started
c3c run device_context --path examples/device_context
```

On a headless Linux machine set `VK_DRIVER_FILES` to the lavapipe ICD first.

## Build as a consumer

`getting_started/project.release.json` is the `project.json` an application
uses when the library sits at `lib/gpu.c3l` (submodule or extracted archive).
Copy `src/` and `shaders/` next to it, build the shader tool from
`lib/gpu.c3l/tools/gpu_shaders`, run it with `--shader-dir shaders`, then
`c3c run hello_gpu`. [Getting started](../docs/getting_started.md#install)
lists the exact commands.
