# Limitations

What the library deliberately does not do, the limits it enforces, and the
driver/environment quirks we know about. If something doesn't work and this
page doesn't explain it, that's a bug in this page — file an issue.

## 1. By design

- **Shader-visible indices are not ownership tokens.** `TextureIndex`,
  `SamplerIndex`, and `GpuAddress` contain no device owner or generation
  metadata. Retain the owner-bearing `TextureView` for each recyclable texture
  index, and never persist or transfer raw shader values between devices.
- **Debug callbacks are borrowed, synchronous, and non-reentrant.** Public and
  backend messages run before the originating call returns; Vulkan may invoke
  the callback concurrently on arbitrary threads. Payload pointers are valid
  only during the callback. The callback must be nonblocking, synchronize its
  own userdata, and must not call gpu.c3l because internal locks may be held.
  Userdata lives through `destroy_device`; no callback occurs after it returns.
  A configured callback also enables structured teardown diagnostics when
  trusted policy is selected. Normal live children are rejected before
  teardown; diagnostics cover internal, partial-initialization, and device-loss
  state.
  A null callback disables structured delivery without changing returned
  faults; `OBJECT_BOUNDARIES`/`FULL` teardown retains stderr output. Callback
  presence enables no checks, tracking, Vulkan layers, or names.
  Descriptor/cache diagnostics are emitted by device-owned operation
  boundaries; pure lookup, range, and context-free result helpers remain
  fault-only to prevent duplicate or context-free messages.

- **No descriptor-set escape hatch.** The root-pointer model and the single
  bindless heap are the only binding paths. There is no API to create your
  own descriptor sets, set layouts, or push descriptors — that machinery is
  what the library exists to remove. If a workload genuinely needs it, it
  needs a different library.
- **Dynamic rendering only.** No `VkRenderPass`/framebuffer objects, no
  subpasses, no tile-based subpass dependencies. Render targets are
  described per pass begin (`RenderPassDesc`) and that is the whole model.
- **Texture history is caller-owned.** `TextureBarrier.before` asserts the
  layout, stages, and access established by earlier ordering. The backend
  validates those semantics under `ContractValidation.FULL` and lowers the
  state once in every policy, but stores no global or per-subresource layout
  history and inserts no repair transition. Applications must retain separate
  history for independently transitioned mip/layer ranges.
- **Async compute is capability-gated.** A distinct compute queue is used
  when available and reported by `DeviceCaps.async_compute`. Resources declare
  their semantic access roles; distinct admitted families use private concurrent
  sharing, while aliased roles stay exclusive. Barriers and queue ordering remain
  explicit. Completion and swapchain readiness waits require an explicit,
  destination-queue-supported device stage; host and presentation stages are
  not valid wait destinations. Exclusive cross-family ownership transfers are
  unsupported.
- **Wireframe polygon rasterization is optional.** `DeviceCaps.line_polygon_mode`
  reports whether `PolygonMode.LINE` is supported. Unsupported adapters reject
  it with `UNSUPPORTED_FEATURE`; filled rasterization and
  `PrimitiveTopology.LINES` remain available.
- **Dynamic raster state raises the intentional minimum device profile.** The
  minimum is Vulkan 1.3 plus `VK_EXT_extended_dynamic_state3` and requires
  `dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. Device creation
  also requires independent blending and depth-bias clamp. An adapter missing
  any requirement is rejected with `UNSUPPORTED_FEATURE`; the backend does not
  synthesize raster-specific pipeline variants.
- **GPU-generated roots are optional.** `DeviceCaps.generated_work` requires
  one backend facility that supports the draw, indexed-draw, and dispatch
  record layouts together. Unsupported devices retain the shared-root indirect
  path and report a zero `max_generated_work_count`.
- **Command recording has no implicit allocator.** Every command list begins
  from a caller-owned `CommandAllocator` bound to one exact selected queue.
  There is no default, frame-owned, or ambient per-thread recording owner. Create
  one allocator per concurrently recording worker, size it explicitly, and
  destroy it after all of its command units retire.
- **Vendored distribution.** There is no package registry; consumers vendor
  the repo (with its binding submodules) under `lib/`. See
  `docs/getting_started.md`.
- **C3 0.8.0 pinned.** The language is pre-1.0 and syntax moves between
  releases; the pin is deliberate and bumped explicitly.
- **Textures are implicitly 2D and views preserve format.** Descriptors select
  width, height, mip count, and array-layer count; they expose no shape or view
  reinterpretation switch. Multisample textures are supported for color/depth
  attachments when the adapter reports the requested count; they have one mip
  and cannot be sampled, stored, or transferred directly.
- **Depth attachments are D32-only.** No public stencil attachment format or
  stencil clear state is exposed.
- **Matrices are not a schema type.** The ABI DSL has no `mat4`; matrices
  travel as four `vec4` columns and are reassembled in the shader
  (`mat4(c0, c1, c2, c3)`). This keeps layout rules trivial (`docs/shader_abi.md`).

## 2. Limits

Fixed limits fault loudly rather than degrade. Values cite their defining
constant; "knob" is the `RuntimeDesc` field that raises the limit where one
exists (else the limit is compile-time).

| Limit | Value | Knob | Fault when exceeded |
|---|---|---|---|
| Texture views in the heap | 4096 default, 65 536 max (`DEFAULT_TEXTURE_HEAP_CAPACITY`, `MAX_SHADER_HEAP_CAPACITY` in `gpu/gpu.c3i`) | `texture_heap_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Sampler descriptors | 256 default, 65 536 max (`DEFAULT_SAMPLER_HEAP_CAPACITY`, `MAX_SHADER_HEAP_CAPACITY` in `gpu/gpu.c3i`) | `sampler_heap_capacity` | `DESCRIPTOR_HEAP_FULL` |
| Interned samplers | selected-device `maxSamplerAllocationCount`, capped at 65 536 | — | `SLOT_TABLE_FULL` |
| Sampler mip LOD bias | Absolute value up to selected-device `maxSamplerLodBias`, reported by `DeviceCaps.max_sampler_lod_bias` | — | `INVALID_ARGUMENT` |
| Live textures | 1024 default, 65 536 max (`DEFAULT_TEXTURE_CAPACITY`, `MAX_SHADER_HEAP_CAPACITY` in `gpu/gpu.c3i`) | `texture_capacity` | `SLOT_TABLE_FULL` |
| Live independent allocations | 4096 (`ALLOCATION_CAPACITY` in `gpu/internal/vk/allocation.c3`) | — | `SLOT_TABLE_FULL` |
| Live pipelines | 256 by default (`MAX_PIPELINES` in `gpu/gpu.c3i`) | `pipeline_capacity` | `SLOT_TABLE_FULL` |
| Direct dispatch groups per axis | Selected-device `maxComputeWorkGroupCount`, reported by `DeviceCaps.max_compute_work_group_count` | — | `INVALID_ARGUMENT` |
| Direct or count-buffer indirect draws per command | Selected-device `maxDrawIndirectCount`, reported by `DeviceCaps.max_draw_indirect_count` | — | `INVALID_ARGUMENT` |
| Generated work items | Selected-device semantic limit reported by `DeviceCaps.max_generated_work_count`; zero when unsupported | — | — |
| Live command records | 4096 (`MAX_DEVICE_COMMANDS` in `gpu/internal/device.c3`) | — | `SLOT_TABLE_FULL` |
| Live command allocators per device | 256 (`MAX_COMMAND_ALLOCATORS` in `gpu/internal/vk/command.c3`) | destroy quiescent allocators to recycle generational slots | `SLOT_TABLE_FULL` |
| Command buffers per allocator | 8 default, 4096 max (`DEFAULT_COMMAND_ALLOCATOR_CAPACITY`, `MAX_COMMAND_ALLOCATOR_CAPACITY` in `gpu/gpu.c3i`) | `CommandAllocatorDesc.command_buffer_capacity` | `INVALID_ARGUMENT` above the maximum; `DEVICE_BUSY` while all configured units are live |
| Tracked resource references per command list | 64 default, 4096 max (`DEFAULT_COMMAND_REFERENCES_PER_LIST`, `MAX_COMMAND_REFERENCES_PER_LIST` in `gpu/gpu.c3i`); tracking-on scratch also owns a `next_pow2(2 * capacity)` exact-identity index | `CommandAllocatorDesc.max_resource_references_per_list` | `INVALID_ARGUMENT` above the maximum; `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` while recording |
| Generated preprocess reservations retained by one list | 4 default, 64 max (`DEFAULT_COMMAND_PREPROCESS_PER_LIST`, `MAX_COMMAND_PREPROCESS_PER_LIST` in `gpu/gpu.c3i`) | `CommandAllocatorDesc.max_generated_preprocess_buffers_per_list` | `INVALID_ARGUMENT` above the maximum; `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` while recording |
| Generated preprocess reservation bytes per allocator | zero by default (disabled) | `CommandAllocatorDesc.generated_preprocess_bytes` | `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` during reservation |
| Swapchains | 8 (`MAX_SWAPCHAINS` in `gpu/gpu.c3i`) | — | `SLOT_TABLE_FULL` |
| Color attachments per pass | Lesser of 8 (`MAX_COLOR_ATTACHMENTS` in `gpu/gpu.c3i`) and `DeviceCaps.max_color_attachments` | — | `INVALID_ARGUMENT` |

Two sizing rules that bite:
- **Heap capacities are exact device defaults.** `create_device` checks the
  selected adapter against the runtime's texture and sampler capacities,
  including the driver's exact descriptor-buffer layout size when that path
  is needed. It returns `UNSUPPORTED_FEATURE` rather than clamping; the caller
  may then try another adapter from the runtime.

- **Shader-visible indices have caller-managed lifetime.** Destroying a
  `TextureView` recycles its raw index immediately. Wait or discard every use
  before releasing the view, and do not leave stale indices in GPU-visible
  data. Sampler indices remain stable until device destruction.
- **Tracking-off resource lifetime is caller-owned.** With
  `track_resource_lifetimes = true`, command records retain explicitly named
  allocations, spans, textures, attachment views, and pipelines; early
  destruction returns `RESOURCE_IN_USE`. With tracking off, recording allocates
  no reference storage and teardown adds no wait or deferred destruction. Keep
  every referenced owner live until commands are discarded or covering
  completion points are observed. GPU addresses and shader-visible indices
  remain caller-owned even when tracking is on.
- **Transient data is caller-owned.** Applications choose allocation reuse and
  concurrency policy. Flush CPU writes before submission, retain the covering
  completion point, and wait or poll before rewriting or freeing storage.
- **Tracked references are fixed at allocator creation.** When lifetime
  tracking is enabled, every command buffer receives exactly
  `max_resource_references_per_list` stable sequential entries plus a fixed
  open-addressed index at no more than 0.5 target load. Exact-key duplicate
  lookup is expected constant-time; forced collisions remain bounded by index
  capacity. Exceeding the unique-resource ceiling returns
  `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` without a partial retain or native
  command. Tracking-off allocators ignore that storage setting and keep both
  reference structures empty.
- **Generated reservation capacity has two bounds.** The reservation table has
  `command_buffer_capacity * max_generated_preprocess_buffers_per_list` entries,
  and `generated_preprocess_bytes` limits their total exact driver-reported
  bytes. Reservation occurs only while the allocator is quiescent. Recreate a
  quiescent allocator with larger values after a capacity fault.
- **Omitted swapchain acquisition is now nonblocking.** Earlier releases used a
  hidden one-second backend wait when `acquire_next_image` omitted its timeout;
  the current default is zero. Migrating callers must handle retryable
  `WAIT_TIMEOUT` or pass an explicit finite budget. Pass `TIMEOUT_INFINITE` only
  when the surface platform guarantees presentation forward progress; otherwise
  an unbounded acquire can stall event handling and shutdown.

## 3. Driver and environment quirks

| Symptom | Cause | Workaround | Notes |
|---|---|---|---|
| `intern_sampler` faults `INVALID_ARGUMENT` for enabled anisotropy | requested `max_anisotropy` is outside the inclusive range `[1, DeviceCaps.max_sampler_anisotropy]` | query the cap and clamp explicitly before interning | over-limit values are never implicitly clamped |
| Segfault on any image/sampler access, lavapipe + descriptor-buffer heap | Mesa 25.0.7 lavapipe descriptor-buffer bug | no caller action; automatic selection uses descriptor indexing when it satisfies the request | retest on Mesa upgrade |
| `UNSUPPORTED_FEATURE` at runtime create with Vulkan validation on | `vulkan-validationlayers` not installed | install it, or leave `enable_vulkan_validation = false`; `ContractValidation.FULL` still works without layers | — |
| Windows: driver not found in elevated shells despite `VK_DRIVER_FILES` | elevated processes ignore loader env vars | register the ICD under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (CI does this for mesa-dist-win) | elevated shells only |
| Pipeline-cache blob is 32 bytes, warm start ≈ cold | lavapipe returns a header-only blob (no compiled-shader payload) | expected; real drivers populate it — `pipeline_cache_timing` prints blob size as the signal | — |
| Multithreaded recording shows ~1× scaling with Vulkan validation on | the Khronos layer locks every `vkCmd*` | benchmark with layers off; gate native correctness with layers on (`multithreaded_recording` does both) | — |
| FIFO present does not throttle under xvfb | virtual displays have no vblank | expected; pacing numbers under xvfb are structural only (`present_mode_explorer`) | — |
| Schema field named `sampler` (or other GLSL keyword) breaks shader compile | generator emits the name verbatim into GLSL | rename the field (for example, `heap_sampler`) | reserved names are not rewritten |
| Pipeline creation returns `SHADER_INVALID` for a size-correct root push block | exact reflection rejects a nested `RootPush`/`GraphicsRootPush` struct because its member shape differs | declare the generated fields directly in the push block, in schema order | reflected member names may differ; numeric shape may not |
| `TYPE_OPTIONAL` c3c crash building the library with debug info | c3c 0.8.0/0.8.1 debug-codegen bug on optional-of-struct vtable signatures | no consumer action; the in-tree vtable uses an out parameter | `scripts/c3c_bug_repro/` |

## 4. Capability queries

Anything the device can answer at runtime lives in `DeviceCaps` (filled at
`create_device`): heap capacities, alignments, sampler limits
(`max_sampler_lod_bias`, `max_sampler_anisotropy`), workload limits
(`max_compute_work_group_count`, `max_draw_indirect_count`,
`max_generated_work_count`), and semantic feature booleans such as
`draw_indirect_count` and `generated_work`. Native implementation choices are
not reported.

Surface support is queried separately. `supports_presentation(adapter,
surface)` preflights device creation; `get_present_mode_support(device,
swapchain)` reports modes after swapchain creation. Prefer queries over
hardcoded assumptions.

Texture format support is queried separately because it depends on both the
backend profile and the physical adapter:

- `get_texture_format_support(device, format)` reports individually
  creatable optimal-tiling usages, linear filterability,
  and backend sample counts. Individual usage bits do not guarantee that a
  combination is supported.
- `supports_texture_desc(device, desc)` checks an exact descriptor,
  including combined usages and adapter extent/mip/layer limits, without
  allocating. Use it to preflight optional formats and adapt asset choices.

The required backend profile is implicit 2D with same-format views and D32
depth. Per-format usages, sample counts, and filterability are optional adapter
capabilities. Higher sample-count bits reflect exact color-attachment or
depth-attachment descriptors supported end to end.
