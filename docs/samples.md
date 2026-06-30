# gpu.c3l Samples and Windowed Tests

## 1. Purpose

Samples prove real usage and provide regression coverage. They are also the preferred way to document the API through working code.

Samples are standalone consumers of `gpu.c3l`. Windowed samples use SDL3 through `sdl3.c3l`.

Because samples are consumers, each sample owns its shaders in its own `shaders/` subdirectory (`samples/<name>/shaders/`). The library ships no application shaders; sample shaders `#include` the published shader-side ABI includes from `include/shaders/` (see `docs/shader_abi.md`).

## 2. Sample project structure

```text
samples/
├── project.json
├── README.md
├── shared/
│   ├── sample_app.c3
│   ├── sample_window_sdl.c3
│   ├── shader_loader.c3
│   ├── readback.c3
│   └── camera.c3
├── hello_compute/
├── root_pointer_compute/
├── bindless_texture_compute/
├── offscreen_triangle/
├── hello_triangle_sdl/
├── texture_viewer_sdl/
├── swapchain_resize_sdl/
└── gpu_driven_draw_sdl/
```

Project dependencies:

```json
{
  "dependency-search-paths": [ "../lib", ".." ],
  "dependencies": [ "gpu", "sdl3" ]
}
```

Imports:

```c3
import gpu;
import sdl;
```

Headless samples should not import SDL.

## 3. Shared sample helpers

Shared sample code may provide:

```text
load_spirv
create_default_device
create_validation_device
create_sdl_window
create_surface_from_sdl_window
run_sdl_event_loop
write_buffer_upload
readback_buffer
readback_texture
wait_for_device_idle
```

Shared helpers are sample infrastructure, not public library API.

## 4. Sample: hello_compute

Type:

```text
headless
```

Purpose:

```text
prove device creation, command submission, and readback with minimal shader state
```

Expected behavior:

```text
create device
create small output buffer
dispatch shader that writes known values
copy/readback output
print pass/fail
```

## 5. Sample: root_pointer_compute

Type:

```text
headless
```

Purpose:

```text
first core architecture proof
```

Expected behavior:

```text
create addressable input/output buffers
allocate RootArgs from frame arena
write input/output GpuAddress values into RootArgs
dispatch shader with root pointer
read back output
verify deterministic result
```

This sample is the first release gate.

## 6. Sample: bindless_texture_compute

Type:

```text
headless
```

Purpose:

```text
prove TextureIndex and SamplerIndex shader access
```

Expected behavior:

```text
create texture
upload known texels
create texture descriptor
create sampler
a material/root struct stores TextureIndex and SamplerIndex
compute shader samples texture by index
read back output buffer
verify sampled values
```

## 7. Sample: offscreen_triangle

Type:

```text
headless
```

Purpose:

```text
prove graphics pipeline, dynamic rendering, render target texture, and readback without swapchain
```

Expected behavior:

```text
create color render target
create graphics pipeline
begin render pass
clear and draw triangle
end render pass
copy render target to readback
verify selected pixels
```

## 8. Sample: hello_triangle_sdl

Type:

```text
windowed SDL3
```

Purpose:

```text
prove surface creation, swapchain, acquire/present, and basic resize handling
```

Expected behavior:

```text
initialize SDL
create window
create gpu device
create surface/swapchain
render triangle each frame
handle resize
present
shutdown cleanly
```

SDL module import:

```c3
import sdl;
```

The sample depends on package `sdl3`.

## 9. Sample: texture_viewer_sdl

Type:

```text
windowed SDL3
```

Purpose:

```text
prove texture upload, sampling, and swapchain presentation together
```

Expected behavior:

```text
load or generate texture data
upload texture
create TextureIndex/SamplerIndex
render fullscreen triangle sampling the texture
present to swapchain
```

## 10. Sample: swapchain_resize_sdl

Type:

```text
windowed SDL3
```

Purpose:

```text
stress swapchain resize and out-of-date handling
```

Expected behavior:

```text
repeated resize
recreate swapchain and dependent render targets
no leaked swapchain images/views
no validation errors
```

## 11. Sample: gpu_driven_draw_sdl

Type:

```text
windowed SDL3
```

Purpose:

```text
prove compute-generated indirect draw commands
```

Expected behavior:

```text
compute shader writes indirect draw arguments
barrier shader write -> indirect read
graphics pass consumes indirect draw buffer
present result
```

## 12. Sample naming

Use behavior names and platform suffixes:

```text
root_pointer_compute
bindless_texture_compute
offscreen_triangle
hello_triangle_sdl
gpu_driven_draw_sdl
```

Do not use milestone names in sample directory names.

## 13. Sample verification

Every sample should support:

```text
--validation
--gpu <index or name substring>
--frames <n>
--headless where applicable
--dump-stats
```

Windowed samples should support:

```text
--width
--height
--present-mode
```

## 14. Sample output policy

Headless samples should return non-zero process exit on failure and print concise diagnostics.

Windowed samples should print:

```text
selected GPU
backend feature path
swapchain format
descriptor heap implementation
memory usage summary on shutdown
```

## 15. Acceptance criteria

Samples are acceptable when:

```text
headless samples run without SDL3
windowed samples import sdl and depend on sdl3
all samples clean up resources before device destruction
validation-enabled runs produce no errors for release-gate samples
root_pointer_compute remains small and readable as the canonical minimal example
```
