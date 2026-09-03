# Graphics pipeline identity

**Question.** Should `polygon_mode` and `sample_count` move out of
`GraphicsPipelineDesc` into command-time state?

**Decision.** No. Both stay in pipeline identity.

## Current split

Pipeline identity: shaders, color formats, depth format, `sample_count`,
`polygon_mode`. Everything else is command-time state under `GraphicsState`
(`DynamicRasterState`, `DepthState`, `ColorState`). The private cache
(`PipelineKey` in `gpu/internal/vk/pipeline_cache.c3`) keys on identity minus
`debug_name`; equal descriptors share one native pipeline.

## Evidence

Source inventory of every `create_graphics_pipeline` call in this repository
and `gpu.c3l-samples`. No third-party consumer was inventoried.

| Source | Workload | Negative path | Identity fixture |
|---|---:|---:|---:|
| `test/src`, `test/cpu` | 30 | 11 | 2 |
| `test_vk_render_pass_validation.c3`, `build_graphics_key` only | 0 | 0 | 13 |
| `gpu.c3l-samples`, distinct rendering identities | 16 | 0 | 0 |
| `gpu.c3l-samples/pipeline_cache_timing`, 24 constructions | 0 | 0 | 6 identities |

Workload: created to render. Negative path: expected to fault. Identity
fixture: built to assert cache identity or a feature gate, never recorded.

- **Polygon-mode-only variants: 0.** Every workload descriptor uses `FILL`.
  `LINE` appears only in a feature-gate test and a key comparison.
- **Sample-count-only variants: 0.** One workload sets `sample_count != ONE`
  (`test_vk_multisample_render.c3`), derived from format support, with no
  single-sample twin.
- **Re-creation is rare and off the frame path.** Samples re-create on a
  swapchain format change, not on resize, and destroy before creating.
  `pipeline_cache_bench` makes 200 handles from one descriptor while varying
  raster state; `pipeline_cache_timing` builds 24 variants that collapse to 6
  native pipelines because only formats are identity.

## Feature cost

The backend already requires `VK_EXT_extended_dynamic_state3` for color blend
and write mask. Dynamic polygon mode and rasterization samples are separate
feature bits, not yet required. `LINE` is additionally gated on
`fillModeNonSolid` and reported through `DeviceCaps.line_polygon_mode`.
Lavapipe advertises all three bits; that is a CPU driver, not a target
adapter.

## Reasoning

- `polygon_mode`: no workload variants. Making it dynamic keeps the capability
  check and fault path and adds a command-time path plus a device-baseline
  feature bit.
- `sample_count`: no workload variants. Dynamic samples must agree with the
  active attachments, a per-pass validation obligation the static field
  discharges once at creation.

No timing was collected; a software driver cannot settle a GPU-side question.

## Revisit when

A real workload creates otherwise identical fill and line pipelines, or keeps
parallel single-sample and multisample pipeline sets, and the duplication
appears during latency-sensitive work rather than initialization.
