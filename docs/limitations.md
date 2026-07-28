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
  Normal live children are rejected before teardown; FULL diagnostics cover
  internal, partial-initialization, and device-loss state.
  A null callback disables structured delivery without changing returned
  faults; `FULL` teardown retains stderr output. Callback presence enables no
  checks, tracking, leak scans, Vulkan layers, or names.
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
- **Graphics state has no implicit default.** It belongs to the command buffer
  and survives render-pass boundaries. Minimal begin does not change or replay
  it. For a fresh recording, begin the pass, bind a compatible graphics
  pipeline, and apply a complete `GraphicsState`. Under `FULL`, regular and
  generated draws reject a recording that has only viewport/scissor overrides.
  Raster, depth, and color changes mutate and resend the caller-owned complete
  packet after a compatible pipeline bind. Neither pass begin nor pipeline
  binding replays that state. When an incompatible pipeline remains selected
  from an earlier pass, bind the next compatible pipeline before beginning.
  `render_geometry_state`
  supplies conventional viewport/raster/depth state and an empty color packet;
  color passes must replace `GraphicsState.color` with a packet matching the
  pipeline's ordered color-format domain. The old
  `full_render_graphics_state` name is retired because it implied that the
  empty color packet was complete.
- **Texture history is caller-owned.** `TextureBarrier.before` asserts the
  layout, stages, and access established by earlier ordering.
  `TextureState.layout` is an operational requirement for native use, not
  descriptive metadata. The backend validates those semantics under
  `ContractValidation.FULL` and lowers the state once in every policy, but
  stores no global or per-subresource layout history and inserts no repair
  transition. Applications must retain separate history for independently
  transitioned mip/layer ranges.
- **Timestamp history and correlation are caller-owned.** The library inserts
  no implicit query reset and does not track whether individual slots were reset
  or written, including under `FULL`. Reset before reuse, write every query
  before resolve or host read, and order reads after execution. Compare raw
  values only when both writes used the same native queue; distinct native
  queues are not calibrated. Resolve waits for availability on the device, so
  resolving an unwritten query can stall that queue indefinitely.
- **Texture layouts use one explicit profile.** Transfer, sampled, storage,
  attachment, initialization, and presentation states lower to their
  corresponding classic Vulkan layouts on every device. A global barrier has
  no texture identity or subresource range and cannot establish a required
  layout. Layout changes, including `UNDEFINED` initialization and `PRESENT`
  transitions, require explicit texture barriers. When migrating from the
  removed unified-layout profile, replace every global `Barrier` that stood in
  for a texture layout change with `cmd_texture_barrier`. A texture cannot be
  sampled and storage-accessed in the same pass; split those uses and record
  the explicit transition between their classic layouts.
- **Async compute is capability-gated.** A distinct compute queue is used
  when available and reported by `DeviceCaps.async_compute`. Resources declare
  their semantic access roles; distinct admitted families use private concurrent
  sharing, while aliased roles stay exclusive. Barriers and queue ordering remain
  explicit. Completion waits require at least one destination-queue-supported
  device stage; `indirect` is supported only on graphics or compute queues and
  includes implicit command preprocessing when generated work is enabled.
  Swapchain readiness rejects `indirect` because it names image consumption.
  Host and presentation stages are not valid wait destinations. Exclusive
  cross-family ownership transfers are unsupported.
- **Wireframe polygon rasterization is optional.** `DeviceCaps.line_polygon_mode`
  reports whether `PolygonMode.LINE` is supported. Unsupported adapters reject
  it with `UNSUPPORTED_FEATURE`; filled rasterization and
  `PrimitiveTopology.LINES` remain available.
- **Dynamic raster state raises the intentional minimum device profile.** The
  minimum is Vulkan 1.3 plus `VK_EXT_extended_dynamic_state3` and requires
  `dynamicPrimitiveTopologyUnrestricted == VK_TRUE`. Device creation also
  requires the EDS3 color blend-enable, blend-equation, and write-mask features
  and commands, independent blending, and depth-bias clamp. An adapter missing
  any requirement is rejected with `UNSUPPORTED_FEATURE`; the backend does not
  synthesize raster- or color-state pipeline variants or replay hidden
  defaults.
- **GPU-generated roots are optional.** `DeviceCaps.generated_work` requires
  one Vulkan facility that supports the draw, indexed-draw, and dispatch
  record layouts together. Unsupported devices retain the shared-root indirect
  path and report a zero `max_generated_work_count`.
- **Command recording has no implicit allocator.** Every command list begins
  from a caller-owned `CommandAllocator` bound to one exact selected queue.
  There is no default, frame-owned, or ambient per-thread recording owner. Create
  one allocator per concurrently recording worker, size it explicitly, and
  destroy it after all of its command units retire.
- **Command tokens are direct and one-shot.** Both command token types carry a
  library-owned typed pointer to one address-stable record, its reuse
  generation, and a packed static device-slot identity. Recording checks slot
  liveness and generation before dereferencing the record, then compares its
  generation and authoritative phase. Callers must not fabricate or mutate
  token storage; aliases remain thread-confined and one-shot.
- **Vendored distribution.** There is no package registry; consumers vendor
  the repo (with its binding submodules) under `lib/`. See
  `docs/getting_started.md`.
- **C3 0.8.0 pinned.** The language is pre-1.0 and syntax moves between
  releases; the pin is deliberate and bumped explicitly.
- **Texture dimensions are deliberately narrow and views preserve format.**
  Zero `TextureDesc.depth` selects 2D; positive depth selects ordinary 3D.
  There is no dimension enum, 1D/cube texture, 3D array, 3D attachment,
  z-slice view, or format reinterpretation switch. Multisample textures are
  2D color/depth attachments with one mip and cannot be sampled, stored, or
  transferred directly.
- **Sparse texture creation and binding are deliberately narrow.**
  It supports single-layer, single-sample color 2D/3D images with sampled,
  storage, and transfer usage. Attachment usage, depth/stencil aspects, and
  native requirement shapes beyond COLOR plus optional METADATA are rejected.
  Creation commits no memory. Binding covers color tiles plus color/metadata
  mip tails; it does not support sparse buffers, arrays, cubes, multisampling,
  depth/stencil, multiplanar images, aliasing residency, eviction, or automatic
  streaming. The backend adds no hidden allocation, residency map, or backing
  lifetime tracking. Callers keep bound bytes live and exclusive until an
  ordered unbind/replacement and all prior users complete.
  Descriptor publication is independent of residency. Because FULL validation
  cannot prove regional residency without a map, it rejects sparse texture
  transfer copies while barriers remain legal; TRUSTED lowering is unchanged.
  `SparseTextureCaps.nonresident_strict` describes device behavior; it is not
  a substitute for establishing residency.
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
| Sparse requirement aspects per texture | 2 (`MAX_SPARSE_TEXTURE_ASPECTS` in `gpu/gpu.c3i`): COLOR plus optional METADATA | — | `UNSUPPORTED_FEATURE` |
| Live independent allocations | 4096 (`ALLOCATION_CAPACITY` in `gpu/internal/vk/allocation.c3`) | — | `SLOT_TABLE_FULL` |
| Live pipelines | 256 by default (`MAX_PIPELINES` in `gpu/gpu.c3i`) | `pipeline_capacity` | `SLOT_TABLE_FULL` |
| Live timestamp pools | 256 (`MAX_TIMESTAMP_POOLS` in `gpu/internal/vk/timestamp.c3`) | destroy quiescent pools to recycle generational slots | `SLOT_TABLE_FULL` |
| Direct dispatch groups per axis | Selected-device `maxComputeWorkGroupCount`, reported by `DeviceCaps.max_compute_work_group_count` | — | `INVALID_ARGUMENT` |
| Direct or count-buffer indirect draws per command | Selected-device `maxDrawIndirectCount`, reported by `DeviceCaps.max_draw_indirect_count` | — | `INVALID_ARGUMENT` |
| Generated work items | Selected-device semantic limit reported by `DeviceCaps.max_generated_work_count`; zero when unsupported | — | — |
| Dynamic viewport dimensions and coordinates | Selected-device `maxViewportDimensions` and `viewportBoundsRange`, consumed privately when validating a complete graphics-state packet or viewport update | — | `INVALID_ARGUMENT` |
| Live command records | 4096 (`MAX_DEVICE_COMMANDS` in `gpu/internal/device.c3`) | — | `SLOT_TABLE_FULL` |
| Live command allocators per device | 256 (`MAX_COMMAND_ALLOCATORS` in `gpu/internal/vk/command.c3`) | destroy quiescent allocators to recycle generational slots | `SLOT_TABLE_FULL` |
| Command buffers per allocator | 8 default, 4096 max (`DEFAULT_COMMAND_ALLOCATOR_CAPACITY`, `MAX_COMMAND_ALLOCATOR_CAPACITY` in `gpu/gpu.c3i`) | `CommandAllocatorDesc.command_buffer_capacity` | `INVALID_ARGUMENT` above the maximum; `DEVICE_BUSY` while all configured units are live |
| Tracked resource references per command list | 64 default, 4096 max (`DEFAULT_COMMAND_REFERENCES_PER_LIST`, `MAX_COMMAND_REFERENCES_PER_LIST` in `gpu/gpu.c3i`); FULL scratch owns one fixed sequential list | `CommandAllocatorDesc.max_resource_references_per_list` | `INVALID_ARGUMENT` above the maximum; `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` while recording |
| Generated preprocess reservations retained by one list | 4 default, 64 max (`DEFAULT_COMMAND_PREPROCESS_PER_LIST`, `MAX_COMMAND_PREPROCESS_PER_LIST` in `gpu/gpu.c3i`) | `CommandAllocatorDesc.max_generated_preprocess_buffers_per_list` | `INVALID_ARGUMENT` above the maximum; `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` while recording |
| Generated preprocess reservation bytes per allocator | zero by default (disabled) | `CommandAllocatorDesc.generated_preprocess_bytes` | `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` during reservation |
| Swapchains | 8 (`MAX_SWAPCHAINS` in `gpu/gpu.c3i`) | — | `SLOT_TABLE_FULL` |
| Color attachments per pass | Lesser of 8 (`MAX_COLOR_ATTACHMENTS` in `gpu/gpu.c3i`) and `DeviceCaps.max_color_attachments` | — | `INVALID_ARGUMENT` |

Two sizing rules that bite:
- **Heap capacities are exact device defaults.** `create_device` checks the
  selected adapter against cached sampled-image, storage-image, sampler,
  per-stage aggregate, and all-pools update-after-bind limits. It returns
  `UNSUPPORTED_FEATURE` with an exact FULL/backend diagnostic rather than
  clamping; the caller may then try another adapter from the runtime. Values
  above `MAX_SHADER_HEAP_CAPACITY` remain `INVALID_ARGUMENT`. The 4096-texture
  and 256-sampler defaults are retained deliberately: every relevant
  update-after-bind limit is at least 500,000 when the required descriptor
  indexing feature is supported, so the heap's largest checked aggregate is
  8,448 descriptors. These checks are necessary compatibility gates, not
  reservations of device-wide capacity shared with other update-after-bind
  pools or pipeline layouts.

- **Shader-visible indices have caller-managed lifetime.** Destroying a
  `TextureView` recycles its raw index immediately. Wait or discard every use
  before releasing the view, and do not leave stale indices in GPU-visible
  data. Sampler indices remain stable until device destruction.
- **TRUSTED resource lifetime is caller-owned.** Under `FULL`, command records
  retain explicitly named
  allocations, spans, textures, attachment views, and pipelines; early
  destruction returns `RESOURCE_IN_USE`. Under `TRUSTED`, recording allocates
  no reference storage and teardown adds no wait or deferred destruction. Keep
  every referenced owner live until commands are discarded or covering
  completion points are observed. GPU addresses and shader-visible indices
  remain caller-owned even under FULL.
- **Submission retains command allocator and device lifetime.** Successful
  submit consumes the public executable value, but its stable record, native
  buffer, fixed scratch, allocator unit, and retained device ownership
  remain live until ordered completion retirement. The allocator cannot be
  destroyed and the unit cannot be reused before that retirement.
- **Transient data is caller-owned.** Applications choose allocation reuse and
  concurrency policy. Flush CPU writes before submission, retain the covering
  completion point, and wait or poll before rewriting or freeing storage.
- **Tracked references are fixed at allocator creation.** Under `FULL`, every
  command buffer receives exactly
  `max_resource_references_per_list` stable sequential entries. Duplicate
  lookup scans the retained prefix and compares the complete
  owner/index/generation identity, so recording work grows linearly with the
  number of already-retained unique resources. Each entry stores its canonical
  retained counter for direct release. Compound operations preflight capacity,
  and exceeding the unique-resource ceiling returns
  `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` without a partial retain or native
  command. TRUSTED allocators ignore that storage setting and keep the
  reference list empty.
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
| `UNSUPPORTED_FEATURE` at runtime create with Vulkan validation on | `vulkan-validationlayers` not installed | install it, or leave `enable_vulkan_validation = false`; `ContractValidation.FULL` still works without layers | — |
| Windows: driver not found in elevated shells despite `VK_DRIVER_FILES` | elevated processes ignore loader env vars | register the ICD under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (CI does this for mesa-dist-win) | elevated shells only |
| Pipeline-cache blob is 32 bytes, warm start ≈ cold | lavapipe returns a header-only blob (no compiled-shader payload) | expected; real drivers populate it — `pipeline_cache_timing` prints blob size as the signal | — |
| Multithreaded recording shows ~1× scaling with Vulkan validation on | the Khronos layer locks every `vkCmd*` | benchmark with layers off; gate native correctness with layers on (`multithreaded_recording` does both) | — |
| FIFO present does not throttle under xvfb | virtual displays have no vblank | expected; pacing numbers under xvfb are structural only (`present_mode_explorer`) | — |
| Schema field named `sampler` (or other GLSL keyword) breaks shader compile | generator emits the name verbatim into GLSL | rename the field (for example, `heap_sampler`) | reserved names are not rewritten |
| Pipeline creation returns `SHADER_INVALID` for a size-correct root push block | exact reflection rejects a nested `RootPush`/`GraphicsRootPush` struct because its member shape differs | declare the generated fields directly in the push block, in schema order | reflected member names may differ; numeric shape may not |

## 4. Capability queries

Anything the device can answer at runtime lives in `DeviceCaps` (filled at
`create_device`): heap capacities, alignments, sampler limits
(`max_sampler_lod_bias`, `max_sampler_anisotropy`), workload limits
(`max_compute_work_group_count`, `max_draw_indirect_count`,
`max_generated_work_count`), and semantic feature booleans such as
`draw_indirect_count` and `generated_work`. Native implementation choices are
not reported.

`DeviceCaps.timestamps` is queue-role-specific. Its graphics, compute, and
transfer widths come from the selected native families rather than one
device-wide counter width. A transfer role is reported only when its selected
queue also supports graphics or compute; an aliased transfer role can therefore
be available while a dedicated transfer-only role is not. The conversion period
is device-wide, but that does not make values from distinct native queues
comparable.

Viewport dimensions and coordinate bounds are an intentional exception: the
backend consumes them privately to validate `cmd_set_graphics_state` and
`cmd_set_viewport`, while callers can submit a value and receive
`INVALID_ARGUMENT` without preflighting a public cap.
Negative viewport height, in-range negative coordinates, reversed depth,
off-pass overscan, and empty scissors are supported; zero viewport height is
rejected. These limits can be promoted into `DeviceCaps` later if a caller-side
layout query becomes necessary.

Surface support is queried separately. `supports_presentation(adapter,
surface)` checks one adapter/surface pair, while `supports_device_desc`
preflights the complete presentation and queue descriptor.
`get_present_mode_support(device, swapchain)` reports modes after swapchain
creation. These queries are optional; callers may create directly and handle
`UNSUPPORTED_FEATURE`.

Texture format support is queried separately because it depends on both the
backend profile and the physical adapter:

- `get_texture_format_support(device, format)` reports individually
  creatable optimal-tiling usages, linear filterability,
  and backend sample counts. Individual usage bits do not guarantee that a
  combination is supported.
- `supports_texture_desc(device, desc)` checks an exact descriptor,
  including dimension, three-axis extent, combined usages, and adapter
  mip/layer/sample limits, without allocating. Use it to preflight optional
  formats and adapt asset choices.

The required backend profile supports ordinary 2D and 3D images with
same-format views and D32 depth attachments for 2D. Per-format usages, sample
counts, dimensions, and filterability remain optional adapter capabilities.
`AdapterLimits.max_texture_dimension_3d` is only a quick ceiling;
`supports_texture_desc` is the exact decision. Higher sample-count bits reflect
2D color-attachment or depth-attachment descriptors supported end to end.
