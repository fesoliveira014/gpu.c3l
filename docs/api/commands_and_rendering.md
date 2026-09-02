# Commands and rendering

## Command allocators

`create_command_allocator` creates a `CommandAllocator` bound to one exact
selected `Queue`. `CommandAllocatorDesc` sizes:

- reusable command-buffer units;
- retained resource references per list under full validation; and
- `max_acceleration_structure_geometries_per_build`, the fixed geometry/range
  scratch reserved for every command unit.

Generated-work storage is not described here. It is sized by
`reserve_generated_work` and owned by the backend.

Defaults and maxima are exposed as
`DEFAULT_COMMAND_ALLOCATOR_CAPACITY`,
`DEFAULT_COMMAND_REFERENCES_PER_LIST`,
`MAX_COMMAND_ALLOCATOR_CAPACITY`, and
`MAX_COMMAND_REFERENCES_PER_LIST`.

Creation preallocates native buffers and fixed per-list scratch before
returning. `destroy_command_allocator` never waits and succeeds only when all
recordings are discarded/ended and all executable or submitted units have
retired.

Acceleration-structure geometry capacity has no nonzero default: set it on an
opted-in device before recording builds. A build with more geometries returns
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` before native emission or retained-state
mutation. TLAS builds consume one native instances geometry and therefore need
capacity at least one.

One allocator has one recording owner while any recording is live. Different
allocators may record concurrently. After the last recording ends, application
synchronization may move the allocator to another worker.

## Command lifecycle

```text
begin_commands
  -> CommandList (recording)
end_commands
  -> ExecutableCommandList
submit or discard_executable_commands
  -> completion retirement returns the unit
```

`discard_commands` consumes a recording without making it executable.
`end_commands`, discard, and successful submission consume their one-shot
input. A failed end/submit leaves the documented token state retryable unless
an authoritative phase fault says otherwise.

Token copies are aliases of one internal record. All aliases are
thread-confined and become unusable when another alias consumes or retires the
record. `is_valid` checks token shape/phase metadata but does not extend
lifetime.

## Generated-work reservations

`reserve_generated_work` and `release_generated_work` mutate one quiescent
allocator's reservation for a pipeline and `GeneratedWorkKind`. They are
allocator-confined cold operations; reservation is never implicit and never
happens during recording.

`GeneratedWorkReservationDesc` describes the reservation in application terms
only:

- `pipeline` and `kind` form the reservation key;
- `max_commands_per_list` bounds the record count one generated call may
  request; and
- `concurrent_lists` bounds how many generated calls may hold the reservation
  at the same time.

Every generated call in flight holds one unit until its command unit retires or
is discarded, including two calls recorded into the same command list. Size
`concurrent_lists` as command units times generated calls per unit; a value of
one admits a single generated call, not a single list.

The backend derives and owns the exact private storage those two limits imply
for the selected device and pipeline. There are no public byte sizes,
alignments, or storage handles.

The allocator's reservation table is fixed at 64 units per command buffer and
is allocated on the first reservation call, so an allocator that never reserves
generated work carries no generated-work storage. The table scales with
`command_buffer_capacity`, and it also sets the ceiling on `concurrent_lists`:
one allocator admits `command_buffer_capacity * 64` reserved units in total,
each backed by private device storage. Size the allocator to the recording work
it actually performs.

Reserving again for the same key replaces the reservation. Replacement is
transactional: on failure the previous reservation stays published and usable.
Release requires the same quiescent allocator and never waits for GPU work.
Allocator destruction releases the allocator's generated-work storage.

Recording only consumes an existing matching reservation. A generated call
performs no requirements query, host allocation, or native object creation.
Requesting more records than `max_commands_per_list`, or more generated calls
in flight than `concurrent_lists`, returns `GENERATED_SCRATCH_EXHAUSTED`
without recording a partial command. Exceeding the allocator's fixed reservation table
returns `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` from the reservation call.

## Transfer and buffer commands

- `cmd_copy_buffer` uses `BufferCopyDesc`.
- `cmd_fill_buffer` fills an aligned span with a 32-bit pattern.
- `cmd_copy_buffer_to_texture` uses `BufferTextureCopyDesc`.
- `cmd_copy_texture_to_buffer` uses `TextureBufferCopyDesc`.

Descriptors identify exact ranges and texture regions. Recording validates
queue support, bounds, usage, alignment, and state according to the contract
policy. Transfers do not insert texture transitions or host visibility
operations.

## Acceleration-structure builds, updates, and clones

`cmd_build_acceleration_structure` records one full BLAS or TLAS build outside
a render pass. A BLAS supplies an ordered
`AccelerationStructureGeometryBuildDesc` array matching its immutable creation
schema. A TLAS instead supplies an instance span and count. Both supply
caller-owned build scratch satisfying the queried size and alignment.

`cmd_update_acceleration_structure` updates the destination in place. It is
valid only after a prior full build or clone has completed, when the structure
was created with `allow_update`, and with the same per-geometry primitive
counts, triangle vertex counts and transform presence, or TLAS instance count,
as that completed structure. It uses the queried update scratch; there is no
separate source handle.

When `DeviceCaps.acceleration_structures.indirect_build` is true,
`cmd_build_acceleration_structure_indirect` and
`cmd_update_acceleration_structure_indirect` use the same descriptor and
scratch contracts but read build ranges from a caller-owned `GpuSpan`.
Descriptor primitive and instance counts are CPU maxima for these commands;
triangle `vertex_count` is the exact `maxVertex + 1` bound. The packet contains
one `AccelerationStructureIndirectBuildRange` per ordered BLAS geometry, or one
for a TLAS, at fixed 16-byte stride. Its address is four-byte aligned and its
allocation supports indirect use. Trailing bytes are ignored.

`primitive_count` may be zero, is no greater than its descriptor maximum, and
together with the remaining fields stays within the explicit input spans.
`primitive_offset` is a byte offset aligned to the index element for indexed
triangles, the smallest vertex-format component size for non-indexed triangles,
8 bytes for AABBs, or 16 bytes for TLAS instances. `transform_offset` is 16-byte
aligned. For triangles, `first_vertex` offsets vertex addressing; every
resulting vertex index, including an indexed value plus `first_vertex`, is less
than the descriptor's exact `vertex_count` bound. Indirect updates additionally
require every actual count to equal the preceding build or update's actual
count. The library does not inspect or read back the packet. A direct update is
rejected after an indirect construction whose actual counts are unknown;
continue with indirect updates or rebuild directly.

`cmd_clone_acceleration_structure` records one device clone from a completed
source into a distinct, matching, caller-created unbuilt destination. It uses
no input or scratch span. A successfully recorded destination is pending until
submission retirement; discarding the command restores it to unbuilt. Clone
does not create storage, submit, wait, or publish a TLAS view.

All construction commands are valid on selected compute or graphics queues and
invalid during a render pass or on transfer-only queues. They insert no
barrier. The caller explicitly orders host writes, construction commands,
later ray queries, and cross-submit consumers. Under full validation builds
and updates retain the destination and every explicit backing span, while a
clone retains exactly its source and destination until retirement. Raw BLAS
addresses already packed into TLAS instances remain caller-owned and are not
rewritten by cloning.

FULL validation also retains the indirect range span. TRUSTED performs no
reference tracking, but still checks the command token, destination identity,
fixed packet range, address alignment, indirect usage, capability, and safe
native lowering. Neither policy inserts a dependency; order a compute-written
packet with a destination containing both `.indirect` and
`.acceleration_structure_build`.

## Compute work

`cmd_bind_pipeline` selects a compatible compute, graphics, or ray-tracing
pipeline in the command state.

- `cmd_dispatch` supplies one root and `Vec3u` group counts.
- `cmd_dispatch_indirect` reads one `DispatchIndirectCommand`.
- `cmd_dispatch_generated` reads generated root/argument records and a count.

Direct counts must fit `DeviceCaps.max_compute_work_group_count`.
Indirect argument storage and count storage are ordinary `GpuSpan` values
whose contents and lifetime are caller-owned.

## Ray tracing

Bind a ray-tracing pipeline, then call `cmd_trace_rays` outside a render pass
with one root `GpuAddress`, a caller-owned `RayTracingShaderBindingTable`, and
nonzero `Vec3u` dimensions. The command requires a selected graphics or compute
queue whose native family supports compute operations. A selected compute queue
always satisfies this contract; a selected graphics queue does only when its
native family also supports compute. Full validation rejects an incompatible
family, while trusted validation treats native queue compatibility as a caller
precondition. The invocation product must fit
`DeviceCaps.ray_tracing_pipelines.max_ray_dispatch_invocation_count`. Each axis
must also fit the corresponding component of
`DeviceCaps.ray_tracing_pipelines.max_ray_dispatch_dimensions`; full validation
rejects an oversized width, height, or depth before native recording.

The ray-generation SBT region contains exactly one record. Optional miss, hit,
and callable regions use canonical empty values when absent; nonempty regions
must satisfy device base/handle alignment, stride, range, ownership, and usage
requirements. Record a ray-tracing pipeline bind before tracing.

For a pipeline created with `dynamic_stack_size = true`, bind the pipeline and
then call `cmd_set_ray_tracing_pipeline_stack_size` with a caller-derived
byte count before each direct or indirect trace that needs the state. The
value must fit `uint` and be sufficient for every possible shader execution
in the dispatch; it may be zero when no group reports a stack requirement.
The library exposes per-group role requirements but
does not derive a whole-pipeline value. Binding another dynamic-stack ray
pipeline preserves the recorded value. A successful logical bind of any
static-stack ray pipeline invalidates it, including when the native bind is
deduplicated; later dynamic tracing requires another setter call.

`cmd_trace_rays` pushes the root to all six ray stages and emits one direct
trace. It allocates nothing and inserts no pipeline bind, barrier, submission,
or wait. Keep the bound pipeline, every nonempty SBT allocation, root data,
TLAS view, and all raw-address/index targets live through completion.

When `DeviceCaps.ray_tracing_pipelines.indirect_dispatch` is true,
`cmd_trace_rays_indirect` reads only width, height, and depth from the first
12 bytes of a caller-owned `GpuSpan`. The root and SBT remain direct. The span
must begin with one `TraceRaysIndirectCommand`; trailing bytes are ignored.
Its resolved device address must be four-byte aligned and its allocation must
support indirect use. The command otherwise has the same pipeline, queue,
render-scope, SBT, and root contract as `cmd_trace_rays`.

The library does not inspect GPU-authored dimensions. At execution, every axis
must fit `max_ray_dispatch_dimensions` and their product must fit
`max_ray_dispatch_invocation_count`. Under full validation the command retains
the bound pipeline, nonempty SBT allocations, and argument allocation through
completion. Under trusted validation those lifetimes are caller preconditions.
No path inserts a barrier, readback, bind, submission, wait, or allocation.
If the capability is false, the command returns `UNSUPPORTED_FEATURE`; the
application may choose direct tracing as its fallback.

When `DeviceCaps.ray_tracing_pipelines.indirect2_dispatch` is true,
`cmd_trace_rays_indirect2` reads one `TraceRaysIndirectCommand2` from the
first 104 bytes of an indirect-capable, four-byte-aligned `GpuSpan`; trailing
bytes are ignored. It still pushes the caller's direct `root`, but reads the
ray-generation record, optional miss/hit/callable SBT regions, and dimensions
from the GPU-authored packet. The active pipeline, recording scope, queue, and
dynamic-stack requirements are otherwise the same as direct tracing.

`TraceRaysIndirectCommand2` has this exact generated layout:

| Offset | Field |
|---:|---|
| 0 | `ray_generation_record_address` (`GpuAddress`) |
| 8 | `ray_generation_record_size` (`ulong`) |
| 16 | `miss_table_address` (`GpuAddress`) |
| 24 | `miss_table_size` (`ulong`) |
| 32 | `miss_table_stride` (`ulong`) |
| 40 | `hit_table_address` (`GpuAddress`) |
| 48 | `hit_table_size` (`ulong`) |
| 56 | `hit_table_stride` (`ulong`) |
| 64 | `callable_table_address` (`GpuAddress`) |
| 72 | `callable_table_size` (`ulong`) |
| 80 | `callable_table_stride` (`ulong`) |
| 88 | `width` (`uint`) |
| 92 | `height` (`uint`) |
| 96 | `depth` (`uint`) |
| 100 | `_pad0` (`uint`) |

The ray-generation region contains exactly one valid record. Empty optional
regions use the canonical zero address, size, and stride; nonempty regions must
meet the selected device's SBT alignment, stride, range, addressability, and
usage requirements. GPU-authored dimensions must be nonzero, fit the published
per-axis limits, and have a product within the invocation limit. The library
does not inspect, repair, upload, or synchronize this packet or its raw SBT
addresses. Record a compute-to-indirect barrier after a GPU producer.

Under `ContractValidation.FULL`, the command retains the named bound pipeline
and packet span through completion. The SBT addresses carried inside the packet,
root-reachable data, TLAS view, and every other raw-address target remain
caller-owned under every validation policy; retain their owners through the
completion point. Under `TRUSTED`, the packet span is caller-owned too. If
`indirect2_dispatch` is false, choose basic indirect or direct tracing; the
Indirect2 command returns `UNSUPPORTED_FEATURE`. It inserts no hidden barrier,
readback, pipeline bind, upload, allocation, submission, or wait.

The stack-size command is valid only while recording outside a render pass on a
selected graphics or compute queue whose native family supports compute. FULL
validation reports phase, queue, scope, active-pipeline mode, and numeric misuse
before native emission or state mutation. TRUSTED validation treats those
semantic checks, representability, and sufficiency as caller preconditions.

## Graphics state and drawing

`GraphicsState` is a complete packet containing viewport, scissor, dynamic
raster state, depth state, and ordered color state. It has no implicit
default. `render_geometry_state(width, height)` supplies conventional geometry
state and an empty color packet; color rendering must replace that packet with
one matching the pipeline.

`cmd_set_graphics_state` applies a complete packet.
`cmd_set_viewport` and `cmd_set_scissor` update only those fields after complete
state is established.

Draw commands are:

- `cmd_draw` and `cmd_draw_indexed`;
- `cmd_draw_indirect` and `cmd_draw_indexed_indirect`;
- `cmd_draw_indexed_indirect_count`; and
- `cmd_draw_generated` and `cmd_draw_indexed_generated`.

`IndexType` selects `U16` or `U32` interpretation for the supplied index span.
Direct and shared-root indirect draws supply one vertex and fragment
`GpuAddress`. Generated commands read per-work roots from generated records.
The ABI records `DrawIndirectCommand` and
`DrawIndexedIndirectCommand` match their shader-side twins.

Draw-count and generated-work limits come from `DeviceCaps`. Generated calls
also require a matching reservation on the originating allocator.

## Attachments and render passes

`create_attachment_view` publishes an `AttachmentViewHandle` for one texture
mip/layer selected by `AttachmentViewDesc`. Destroy it only after every
recorded and submitted use retires.

`RenderPassDesc` contains ordered `ColorTargetDesc` values, optional
`DepthTargetDesc`, and the render area. `LoadOp` and `StoreOp` control content
lifetime. `ClearColor` is a floating-point or unsigned-integer union;
`ClearDepth` contains a normalized depth.

The render sequence is:

1. transition every attachment to the required layout;
2. `cmd_begin_render_pass`;
3. bind a compatible graphics pipeline;
4. `cmd_set_graphics_state` with a complete packet;
5. issue draws;
6. `cmd_end_render_pass`; and
7. transition outputs for their next use.

Pass begin is attachment-only. It does not bind a pipeline, create graphics
defaults, replay state, or transition textures. Pipeline compatibility includes
ordered color formats, depth format, and sample count.

## Debug labels

`cmd_begin_label` and `cmd_end_label` annotate recorded work when debug-utils
support is active and otherwise lower to no-ops. Labels still follow command
token confinement and balanced nesting.

## Recording faults

Malformed state, unsupported queue operations, incompatible pipelines/passes,
and one-shot phase misuse return `COMMAND_RECORDING_ERROR`,
`INVALID_RESOURCE_STATE`, `INVALID_ARGUMENT`, `INVALID_HANDLE`, or
`UNSUPPORTED_FEATURE` as documented by the operation.
Fixed reference limits and exhausted generated-work reservations return
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` or
`GENERATED_SCRATCH_EXHAUSTED` without emitting a partial compound command.
