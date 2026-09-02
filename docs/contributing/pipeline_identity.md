# Graphics pipeline identity

Why `polygon_mode` and `sample_count` stay in `GraphicsPipelineDesc` instead of
becoming command-time state. The evidence is a source inventory of the pipelines
real workloads create, covering this repository and `gpu.c3l-samples`; no
third-party consuming project was inventoried.

## What identity already excludes

`GraphicsPipelineDesc` (`gpu/gpu.c3i:1176-1184`) holds shaders, color formats,
depth format, `sample_count`, `polygon_mode`, and a debug name. Everything else
is command-time, owned transitively by `GraphicsState` (`gpu/gpu.c3i:1288-1294`)
through `DynamicRasterState` (`:1158-1166`: topology, cull mode, front face,
depth bias), `DepthState` (`:1128-1132`: test, write, compare), and `ColorState`
(`:1152-1154`) over `ColorTargetState` (`:1146-1149`: blend equation, write
mask).

The private cache keys on identity only: `PipelineKey`
(`gpu/internal/vk/pipeline_cache.c3:52-62`, `$assert PipelineKey::size == 60`)
carries both candidate fields, `build_graphics_key` (`:358-376`) omits
`debug_name`, and `hash_key` (`:386`), `find_entry` (`:424`), and the refcount
released by `release_entry` (`:484`) alias equal descriptors onto one native
pipeline. These are `@private`, not consumer-observable.

## Inventory

Sites are partitioned before counting. **Workload**: a create call expected to
succeed and feed rendering or command recording. **Negative path**: a create
call wrapped in `@catch` and asserted to fault, so no native pipeline exists.
**Identity fixture**: a descriptor built only to assert cache identity or an
optional-feature gate, either keyed through `build_graphics_key` with no
create, or created and destroyed without recording.

| Source | Workload | Negative path | Identity fixture |
| --- | --- | --- | --- |
| `test/src`, `test/cpu` — `gpu::create_graphics_pipeline` call sites | 30 | 11 | 2 |
| `test/src/test_vk_render_pass_validation.c3:674-811` — `build_graphics_key` only | 0 | 0 | 13 |
| `gpu.c3l-samples` — distinct rendering identities over 14 programs | 16 | 0 | 0 |
| `gpu.c3l-samples/pipeline_cache_timing` — 24 constructions | 0 | 0 | 6 identities |
| `examples/` | 0 | 0 | 0 |

`examples/getting_started/src/main.c3:65` creates a compute pipeline only, and
`docs/getting_started.md:309` shows one illustrative graphics pipeline with
every raster field defaulted.

**Polygon-mode-only variants: 0.** No workload descriptor sets `polygon_mode`;
every one defaults to `FILL`. The only `PolygonMode.LINE` uses are
`test/src/test_vk_pipeline_cache.c3:1649-1662` — a feature gate where one
descriptor faults `UNSUPPORTED_FEATURE` and the other creates and destroys
without recording — and `test/src/test_vk_render_pass_validation.c3:762`, a key
comparison with no create. `polygon_mode` does not appear in `gpu.c3l-samples`.

**Sample-count-only variants: 0.** Exactly one workload descriptor sets
`sample_count != ONE`: `test/src/test_vk_multisample_render.c3:160-169`, whose
value is derived at runtime from device format support
(`common_render_sample_count`, `:8-25`) and which returns early when only 1x is
available (`:114`). It has no otherwise identical single-sample twin. The
remaining hits are a deliberate preflight failure
(`test/src/test_vk_render_pass_validation.c3:653-655`) and key assertions
(`:664-690`, `:753`). No `gpu.c3l-samples` graphics pipeline sets
`sample_count`; the one occurrence, `volume_texture/main.c3:100`, is a
`TextureDesc`.

**Timing and reuse.** Almost every sample pipeline is created during
initialization; `present_mode_explorer/main.c3:105` is the exception, creating
inside the frame loop. Samples that re-create on the swapchain path guard it on
a format change, not on resize — `bool pipeline_changed = !pipeline_valid ||
color_formats[0] != swapchain_info.format` (`textured_cube/main.c3:279`, same
shape at `deferred_shading/main.c3:370`, `hello_triangle_sdl/main.c3:341`,
`present_mode_explorer/main.c3:100`). A resize that keeps the format re-creates
nothing, and when the branch does fire the sample destroys first and then
creates against the new format, so it is a distinct key rather than a cache
hit. Re-creation is therefore rare, bounded, and off the steady-state frame
path, and neither field under review participates in it. Two measurements show
raster permutations already collapsing:
`test/src/pipeline_cache_bench.c3:129-145` creates
`BENCH_RASTER_PERMUTATION_COUNT` (200, `:14`) handles from a single descriptor
while varying `DynamicRasterState`, and
`gpu.c3l-samples/pipeline_cache_timing/main.c3:46-89` builds 24 variants over
blend, topology, color format, and depth format that collapse to 6 native
identities, because only the two format dimensions are identity.

## Feature support

The backend already requires `VK_EXT_extended_dynamic_state3`
(`gpu/internal/vk/device.c3:36`) for three bits:
`extendedDynamicState3ColorBlendEnable`, `...ColorBlendEquation`, and
`...ColorWriteMask` (`device.c3:1525-1527`, requested at `:1880-1882`).
`extendedDynamicState3PolygonMode` and `...RasterizationSamples` are separate
bits, not covered by that requirement. `PolygonMode.LINE` is additionally an
optional capability gated on `fillModeNonSolid` (`device.c3:1531-1532`, enabled
at `:1955`, enforced at `gpu/internal/vk/pipeline_graphics.c3:207` and `:253`).

Lavapipe (`llvmpipe`, `PHYSICAL_DEVICE_TYPE_CPU`, apiVersion 1.4.318)
advertises `extendedDynamicState3PolygonMode`,
`extendedDynamicState3RasterizationSamples`, and `fillModeNonSolid` all true.
That is one CPU software driver, explicitly not an intended target adapter.
Per intended adapter, still to check: the feature bit, the matching `vkCmdSet*`
command, the pipeline dynamic-state declaration, dynamic-rendering interaction,
and a validation-clean representative run.

## Decisions

Both fields stay in pipeline identity, and no prototype is warranted, because
the inventory shows no candidate-only variants to prototype against. No timing
is offered as evidence; see [benchmarking.md](benchmarking.md) for why
software-driver numbers cannot settle a GPU-side question.

**`polygon_mode` — retain.** Zero workload variants, and it is already an
optional capability. Making it dynamic keeps the `fillModeNonSolid` check and
the creation fault path, and adds a command-time path plus a second feature bit
on the device baseline: strictly more surface. Reverse this if a workload
creates otherwise identical fill and line pipelines and the duplication appears
during latency-sensitive work rather than initialization.

**`sample_count` — retain.** Zero workload variants; the single multisample
pipeline has no single-sample twin. Dynamic rasterization samples must agree
with the active attachments, a new per-render-pass validation obligation that
the static field discharges once at creation. Reverse this if a workload keeps
parallel single-sample and multisample pipeline sets whose only difference is
`sample_count`.
