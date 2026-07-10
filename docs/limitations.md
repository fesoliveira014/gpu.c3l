# Limitations

What the library deliberately does not do, the limits it enforces, and the
driver/environment quirks we know about. If something doesn't work and this
page doesn't explain it, that's a bug in this page — file an issue.

## 1. By design

- **No descriptor-set escape hatch.** The root-pointer model and the single
  bindless heap are the only binding paths. There is no API to create your
  own descriptor sets, set layouts, or push descriptors — that machinery is
  what the library exists to remove. If a workload genuinely needs it, it
  needs a different library.
- **Dynamic rendering only.** No `VkRenderPass`/framebuffer objects, no
  subpasses, no tile-based subpass dependencies. Render targets are
  described per pass begin (`RenderPassDesc`) and that is the whole model.
- **Async compute is capability-gated.** A distinct compute queue is used
  when the hardware offers one (second queue in the main family, or a
  compute-only family); `DeviceCaps.async_compute` reports it, and resources
  touched by both GRAPHICS and COMPUTE must then carry the `shared_queues`
  usage flag (concurrent sharing). Single-queue devices (lavapipe) keep the
  graphics alias — the flag is a no-op there. EXCLUSIVE cross-family
  ownership transfers are deliberately absent
  ([#36](https://github.com/fesoliveira014/gpu.c3l/issues/36)).
- **Vendored distribution.** There is no package registry; consumers vendor
  the repo (with its binding submodules) under `lib/`. See
  `docs/getting_started.md`.
- **C3 0.8.0 pinned.** The language is pre-1.0 and syntax moves between
  releases; the pin is deliberate and bumped explicitly.
- **2D, single-sample textures only.** `TEX_1D`/`TEX_3D`/`CUBE` and
  multisample counts remain outside the backend profile. Unsupported profile
  values preflight false and fault `INVALID_ARGUMENT` at creation.
- **D24S8 texture support is adapter-dependent.** Query before creation and
  choose a fallback when unsupported. D24S8 transfer usages stay masked because
  copies cannot select depth versus stencil. Graphics pipelines remain D32-only, so
  D24S8 is currently useful only outside the graphics depth-attachment path.
- **Matrices are not a schema type.** The ABI DSL has no `mat4`; matrices
  travel as four `vec4` columns and are reassembled in the shader
  (`mat4(c0, c1, c2, c3)`). This keeps layout rules trivial (`docs/shader_abi.md`).

## 2. Limits

Fixed limits fault loudly rather than degrade. Values cite their defining
constant; "knob" is the `DeviceDesc` field that raises the limit where one
exists (else the limit is compile-time).

| Limit | Value | Knob | Fault when exceeded |
|---|---|---|---|
| Texture descriptors in the heap | 4096 default, 65 536 max (`descriptor_heap.c3:3`) | `texture_descriptor_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Sampler descriptors | 256 default (`descriptor_heap.c3:4`) | `sampler_descriptor_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Live textures | 1024 default, 65 536 max (`texture.c3:3`) | `texture_capacity` | `SLOT_TABLE_FULL` |
| Live buffers | 4096 (`buffer.c3:3`) | — | `SLOT_TABLE_FULL` |
| Live pipelines / shaders | 256 each (`pipeline.c3:3-4`) | — | `SLOT_TABLE_FULL` |
| Live semaphores | 256 (`sync.c3:6`) | — | `SLOT_TABLE_FULL` |
| Swapchains | 8 (`swapchain.c3:3`) | — | `SLOT_TABLE_FULL` |
| Color attachments per pass | 8 (`pipeline.c3:5`) | — | `INVALID_ARGUMENT` |
| Frame arena (per frame in flight) | 1 MiB (`memory.c3:36`) | — ([#28](https://github.com/fesoliveira014/gpu.c3l/issues/28)) | `ARENA_FULL` |
| Persistent arena | 64 MiB (`memory.c3:37`) | — (#28) | `ARENA_FULL` |
| Staging arena | 32 MiB default (`memory.c3:38`) | `staging_arena_size` | `ARENA_FULL` |
| Readback arena | 8 MiB default (`memory.c3:39`) | `readback_arena_size` | `ARENA_FULL` |

Two sizing rules that bite:

- **Descriptor retires recycle a frame late.** A destroy inside frame N
  retires against N's timeline value and only drains on a later frame — a
  frame that destroys and recreates K descriptors needs K slots of headroom,
  not zero. `bindless_stress` demonstrates both the failure and the sizing.
- **Frame-arena data is per frame in flight.** Every `alloc_frame_span`
  byte exists once per in-flight frame; large per-frame tables belong in
  persistent buffers you rewrite (see `deferred_shading`'s lights).

## 3. Driver and environment quirks

| Symptom | Cause | Workaround | Tracked |
|---|---|---|---|
| Segfault on any image/sampler access, lavapipe + descriptor-buffer heap | Mesa 25.0.7 lavapipe descriptor-buffer bug | `DescriptorHeapMode.AUTO` already prefers descriptor-indexing on lavapipe; don't force `DESCRIPTOR_BUFFER` there | retest on Mesa upgrade |
| `UNSUPPORTED_FEATURE` at device create with validation on | `vulkan-validationlayers` not installed | install it, or `enable_validation = false` | — |
| Windows: driver not found in elevated shells despite `VK_DRIVER_FILES` | elevated processes ignore loader env vars | register the ICD under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (CI does this for mesa-dist-win) | [#18](https://github.com/fesoliveira014/gpu.c3l/issues/18) (closed — docs) |
| Pipeline-cache blob is 32 bytes, warm start ≈ cold | lavapipe returns a header-only blob (no compiled-shader payload) | expected; real drivers populate it — `pipeline_cache_timing` prints blob size as the signal | — |
| Multithreaded recording shows ~1× scaling with validation on | the validation layer locks every `vkCmd*` | benchmark with validation off; gate correctness with it on (`multithreaded_recording` does both) | — |
| FIFO present does not throttle under xvfb | virtual displays have no vblank | expected; pacing numbers under xvfb are structural only (`present_mode_explorer`) | — |
| Schema field named `sampler` (or other GLSL keyword) breaks shader compile | generator emits the name verbatim into GLSL | rename the field (e.g. `heap_sampler`) | [#26](https://github.com/fesoliveira014/gpu.c3l/issues/26) |
| `TYPE_OPTIONAL` c3c crash building the library with debug info | c3c 0.8.0/0.8.1 debug-codegen bug on optional-of-struct vtable signatures | worked around in-tree (out-param `BeginCommandsFn`); fixed upstream in 0.8.2 — revert lands with the version bump | `scripts/c3c_bug_repro/` |

## 4. Capability queries

Anything the device can answer at runtime lives in `DeviceCaps` (filled at
`create_device`): heap capacities, alignments, `max_sampler_anisotropy`,
feature booleans (`draw_indirect_count`, `descriptor_buffer`, …), and
`get_present_mode_support` answers per-surface present modes. Prefer
querying over hardcoding — the samples show the pattern.

Texture format support is queried separately because it depends on both the
backend profile and the physical adapter:

- `get_texture_format_support(device, format)` reports individually
  creatable optimal-tiling usages, linear filterability, backend dimensions,
  and backend sample counts. Individual usage bits do not guarantee that a
  combination is supported.
- `supports_texture_desc(device, desc)` checks an exact descriptor,
  including combined usages and adapter extent/mip/layer limits, without
  allocating. Use it to preflight optional formats and adapt asset choices.

The required backend profile is currently 2D and single-sample. Per-format
usages and filterability are optional adapter capabilities. The support summary
therefore masks 1D, 3D, cube, and multisample counts. D24S8 is reported only
when its exact texture usage is creatable on the selected adapter.
