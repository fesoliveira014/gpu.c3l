# gpu.c3l Samples

Samples live in their own repository:
[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples). It
vendors this library as a pinned submodule (`lib/gpu.c3l`) and consumes it
exactly as an external project would — the genuine consumer path.

## 1. What the repository contains

- 18 samples plus `shared_selftest`, which tests the shared helpers themselves.
- `shared/` helpers compiled into each sample's target as needed:
  `sample_args`, `sample_camera`, `sample_window_sdl`, `screenshot`, `png`,
  `selftest`.
- Per-sample READMEs, with a committed screenshot where the output is visual.
- CI runs every headless sample directly and the 10 windowed ones under
  xvfb/lavapipe (`--frames N --screenshot`), uploading screenshots as an
  artifact.

## 2. Sample index

| Sample | Type | Summary |
|---|---|---|
| **Fundamentals** | | |
| `hello_triangle_sdl` | windowed | SDL3 window, surface, swapchain with fault-driven resize; presents the triangle. |
| `root_pointer_compute` | headless | Root-pointer compute ABI end to end; the canonical minimal example. |
| `offscreen_triangle` | headless | Textured triangle to an offscreen target, no window; image written out. |
| `bindless_texture_compute` | headless | Writes a storage image by `TextureIndex`, samples it back by the same index. |
| `memory_report` | headless | `MemoryStats`, per-heap VMA budgets, arena stats, clean-teardown check. |
| **Textures & draw paths** | | |
| `textured_cube` | windowed | Depth-tested rotating cube textured through the descriptor heap — hello-3D. |
| `texture_filtering` | windowed | Per-level-tinted mip chain on a receding plane; LOD and filter modes visible. |
| `gpu_driven_draw_sdl` | windowed | Compute culls a quad grid and writes indirect draw args; one multi-draw via `gl_DrawID`. |
| **Compute techniques** | | |
| `image_processing` | headless | Compute chain: procedural generation, gaussian blur, histogram, readback. |
| `particle_sim` | windowed | 65,536 particles integrated in compute, rendered as additive billboards. |
| `frustum_culling` | windowed | GPU frustum culling feeding indirect draws — 3D graduation of `gpu_driven_draw_sdl`. |
| **Rendering techniques** | | |
| `shadow_mapping` | windowed | Depth-only pass from an orbiting light into a sampled shadow map. |
| `deferred_shading` | windowed | Multi-target G-buffer plus fullscreen resolve with 16 animated point lights. |
| `pbr_materials` | windowed | 7×7 sphere grid in one instanced draw, materials picked from a GPU table by index. |
| **Performance / stress** | | |
| `bindless_stress` | headless | Descriptor heap at scale and under churn; timing table, hard pass/fail. |
| `multithreaded_recording` | headless | 32 command lists recorded on 1 vs 8 threads via recording contexts; identical submits. |
| `pipeline_cache_timing` | headless | Cold build vs warm start from an exported pipeline-cache blob; timed tiers. |
| `present_mode_explorer` | windowed | Enumerates and cycles supported present modes at runtime, measuring frame pacing. |

## 3. Conventions

- Windowed samples own SDL3: they depend on package `sdl3` and `import sdl`;
  headless samples touch neither.
- Samples are consumers of the public `gpu` API only — no `vk::` or `vma::`
  in sample code.
- Each sample owns its shaders (`<name>/shaders/`), `#include`-ing the
  library's published ABI includes.
- Screenshots go through the shared capture path (`shared/screenshot.c3` +
  `shared/png.c3`).

## 4. Maintenance

This document is a pointer. Sample details — flags, behavior, expected
output — live in the samples repository's per-sample READMEs, not here.
