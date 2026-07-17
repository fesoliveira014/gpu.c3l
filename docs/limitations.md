# Limitations

What the library deliberately does not do, the limits it enforces, and the
driver/environment quirks we know about. If something doesn't work and this
page doesn't explain it, that's a bug in this page — file an issue.

## 1. By design

- **Descriptor indices still double as release tokens.** `TextureIndex` and
  `SamplerIndex` have no device owner metadata, so passing one to another device
  can release a coincident slot. `GpuAddress` is also device-local and must not
  be persisted or transferred between devices.
- **Debug callbacks are borrowed, synchronous, and non-reentrant.** Public and
  backend messages run before the originating call returns; Vulkan may invoke
  the callback concurrently on arbitrary threads. Payload pointers are valid
  only during the callback. The callback must be nonblocking, synchronize its
  own userdata, and must not call gpu.c3l because internal locks may be held.
  Userdata lives through `destroy_device`; no callback occurs after it returns.
  A configured callback also enables structured teardown diagnostics when
  validation is disabled. Normal live children are rejected before teardown;
  diagnostics cover internal, partial-initialization, and device-loss state.
  A null callback disables structured delivery without changing returned
  faults; validation-enabled teardown retains stderr output. Descriptor/cache
  diagnostics are emitted by device-owned operation
  boundaries; pure lookup, range, and context-free result helpers remain
  fault-only to prevent duplicate or context-free messages.
- **Frame-token aliases share one generation.** Copies may allocate until one
  alias ends successfully. That end consumes the device generation, clears the
  passed copy, and makes every other copy stale. A failed end preserves the
  token for retry.

- **No descriptor-set escape hatch.** The root-pointer model and the single
  bindless heap are the only binding paths. There is no API to create your
  own descriptor sets, set layouts, or push descriptors — that machinery is
  what the library exists to remove. If a workload genuinely needs it, it
  needs a different library.
- **Dynamic rendering only.** No `VkRenderPass`/framebuffer objects, no
  subpasses, no tile-based subpass dependencies. Render targets are
  described per pass begin (`RenderPassDesc`) and that is the whole model.
- **Async compute is capability-gated.** A distinct compute queue is used
  when available and reported by `DeviceCaps.async_compute`. Resources declare
  their semantic access roles; distinct admitted families use private concurrent
  sharing, while aliased roles stay exclusive. Barriers and queue ordering remain
  explicit. Exclusive cross-family ownership transfers are unsupported.
- **Wireframe polygon rasterization is optional.** `DeviceCaps.line_polygon_mode`
  reports whether `PolygonMode.LINE` is enabled. Unsupported adapters reject
  it with `UNSUPPORTED_FEATURE`; filled rasterization and
  `PrimitiveTopology.LINES` remain available.
- **Vendored distribution.** There is no package registry; consumers vendor
  the repo (with its binding submodules) under `lib/`. See
  `docs/getting_started.md`.
- **C3 0.8.0 pinned.** The language is pre-1.0 and syntax moves between
  releases; the pin is deliberate and bumped explicitly.
- **2D, single-sample textures only.** `TEX_1D`/`TEX_3D`/`CUBE` and
  multisample counts remain outside the backend profile. Unsupported profile
  values preflight false and fault `INVALID_ARGUMENT` at creation.
- **D24S8 remains outside the backend profile.** Graphics pipelines and render
  passes are D32-only, so capability queries report empty D24S8 support and
  creation faults `INVALID_ARGUMENT`.
- **Matrices are not a schema type.** The ABI DSL has no `mat4`; matrices
  travel as four `vec4` columns and are reassembled in the shader
  (`mat4(c0, c1, c2, c3)`). This keeps layout rules trivial (`docs/shader_abi.md`).

## 2. Limits

Fixed limits fault loudly rather than degrade. Values cite their defining
constant; "knob" is the `DeviceDesc` field that raises the limit where one
exists (else the limit is compile-time).

| Limit | Value | Knob | Fault when exceeded |
|---|---|---|---|
| Texture descriptors in the heap | 4096 default, 65 536 max (`DEFAULT_TEXTURE_DESCRIPTORS`, `MAX_DESCRIPTOR_SLOTS` in `gpu/descriptor_heap.c3`) | `texture_descriptor_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Sampler descriptors | 256 default, 65 536 max (`DEFAULT_SAMPLER_DESCRIPTORS`, `MAX_DESCRIPTOR_SLOTS` in `gpu/descriptor_heap.c3`) | `sampler_descriptor_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Live textures | 1024 default, 65 536 max (`DEFAULT_TEXTURE_CAPACITY` in `gpu/texture.c3`; `MAX_DESCRIPTOR_SLOTS` in `gpu/descriptor_heap.c3`) | `texture_capacity` | `SLOT_TABLE_FULL` |
| Live independent allocations | 4096 (`ALLOCATION_CAPACITY` in `gpu/vk/allocation.c3`) | — | `SLOT_TABLE_FULL` |
| Live pipelines / shaders | 256 each by default (`MAX_PIPELINES`, `MAX_SHADERS` in `gpu/pipeline.c3`) | `pipeline_capacity` for pipelines | `SLOT_TABLE_FULL` |
| Compute push-constant range | Selected-device `maxPushConstantsSize`, reported by `DeviceCaps.max_push_constant_size` | — | `INVALID_ARGUMENT` |
| Direct dispatch groups per axis | Selected-device `maxComputeWorkGroupCount`, reported by `DeviceCaps.max_compute_work_group_count` | — | `INVALID_ARGUMENT` |
| Direct or count-buffer indirect draws per command | Selected-device `maxDrawIndirectCount`, reported by `DeviceCaps.max_draw_indirect_count` | — | `INVALID_ARGUMENT` |
| Live command records | 4096 (`MAX_DEVICE_COMMANDS` in `gpu/device.c3`) | — | `SLOT_TABLE_FULL` |
| Worker recording-pool caches per device | 256 worker caches; 257 contexts including the helper (`MAX_RECORDING_CONTEXTS` in `gpu/vk/command.c3`) | reuse a bounded worker pool | `SLOT_TABLE_FULL` |
| Swapchains | 8 (`MAX_SWAPCHAINS` in `gpu/swapchain.c3`) | — | `SLOT_TABLE_FULL` |
| Color attachments per pass | Lesser of 8 (`MAX_COLOR_ATTACHMENTS` in `gpu/pipeline.c3`) and `DeviceCaps.max_color_attachments` | — | `INVALID_ARGUMENT` |
| Frame arena (per frame in flight) | 16 MiB (`DEFAULT_FRAME_ARENA_SIZE` in `gpu/memory.c3`) | `frame_arena_size` | `ARENA_FULL` |
| Persistent arena | 64 MiB (`DEFAULT_PERSISTENT_ARENA_SIZE` in `gpu/memory.c3`) | `persistent_arena_size` | `ARENA_FULL` |
| Live persistent allocations | 4096 (`MAX_PERSISTENT_ALLOCATIONS` in `gpu/memory.c3`) | — | `SLOT_TABLE_FULL` |
| Staging arena | 32 MiB default (`DEFAULT_STAGING_ARENA_SIZE` in `gpu/memory.c3`) | `staging_arena_size` | `ARENA_FULL` |
| Readback arena | 8 MiB default (`DEFAULT_READBACK_ARENA_SIZE` in `gpu/memory.c3`) | `readback_arena_size` | `ARENA_FULL` |

Two sizing rules that bite:
- **Packed descriptor ceilings are not guaranteed hardware capacities.** On the
  descriptor-indexing path, texture slots count once as sampled images and once
  as storage images. Per-stage resource usage is `2 *
  texture_descriptor_capacity`; plain samplers are excluded. All-pools usage is
  `2 * texture_descriptor_capacity + sampler_descriptor_capacity`.
  `create_device_from_desc` returns `INVALID_ARGUMENT` rather than clamping when any
  per-type, per-stage aggregate, or all-pools update-after-bind limit is
  exceeded.

- **Descriptor retires recycle a frame late.** A destroy inside frame N
  retires against N's timeline value and only drains on a later frame — a
  frame that destroys and recreates K descriptors needs K slots of headroom,
  not zero. `bindless_stress` demonstrates both the failure and the sizing.
- **Frame-arena data is per frame in flight.** Every `alloc_frame_span`
  byte exists once per in-flight frame; large per-frame tables belong in
  persistent spans or allocations you rewrite (see `deferred_shading`'s lights).
- **Pending texture transitions grow with the command record.** There is no
  16-texture recording cap; the backend starts at 16 entries and doubles the
  host allocation as distinct textures are added. Submit or discard the token
  to release that state; a live token prevents its frame-slot pool reset.

## 3. Driver and environment quirks

| Symptom | Cause | Workaround | Notes |
|---|---|---|---|
| Segfault on any image/sampler access, lavapipe + descriptor-buffer heap | Mesa 25.0.7 lavapipe descriptor-buffer bug | `DescriptorHeapMode.AUTO` already prefers descriptor-indexing on lavapipe; don't force `DESCRIPTOR_BUFFER` there | retest on Mesa upgrade |
| `UNSUPPORTED_FEATURE` at device create with validation on | `vulkan-validationlayers` not installed | install it, or `enable_validation = false` | — |
| Windows: driver not found in elevated shells despite `VK_DRIVER_FILES` | elevated processes ignore loader env vars | register the ICD under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (CI does this for mesa-dist-win) | elevated shells only |
| Pipeline-cache blob is 32 bytes, warm start ≈ cold | lavapipe returns a header-only blob (no compiled-shader payload) | expected; real drivers populate it — `pipeline_cache_timing` prints blob size as the signal | — |
| Multithreaded recording shows ~1× scaling with validation on | the validation layer locks every `vkCmd*` | benchmark with validation off; gate correctness with it on (`multithreaded_recording` does both) | — |
| FIFO present does not throttle under xvfb | virtual displays have no vblank | expected; pacing numbers under xvfb are structural only (`present_mode_explorer`) | — |
| Schema field named `sampler` (or other GLSL keyword) breaks shader compile | generator emits the name verbatim into GLSL | rename the field (for example, `heap_sampler`) | reserved names are not rewritten |
| `TYPE_OPTIONAL` c3c crash building the library with debug info | c3c 0.8.0/0.8.1 debug-codegen bug on optional-of-struct vtable signatures | no consumer action; the in-tree vtable uses an out parameter | `scripts/c3c_bug_repro/` |

## 4. Capability queries

Anything the device can answer at runtime lives in `DeviceCaps` (filled at
`create_device`): heap capacities, alignments, `max_sampler_anisotropy`,
workload limits (`max_compute_work_group_count`,
`max_draw_indirect_count`), and feature booleans such as
`draw_indirect_count` and `descriptor_buffer`.

Surface support is queried separately. `supports_presentation(adapter,
surface)` preflights device creation; `get_present_mode_support(device,
swapchain)` reports modes after swapchain creation. Prefer queries over
hardcoded assumptions.

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
therefore masks 1D, 3D, cube, multisample counts, and D24S8 until the
rendering path supports it end to end.
