# Commands and rendering

## Command allocators

`create_command_allocator` creates a `CommandAllocator` bound to one exact
selected `Queue`. `CommandAllocatorDesc` sizes:

- reusable command-buffer units;
- retained resource references per list under full validation;
- generated-work reservations per list;
- allocator-owned preprocess bytes; and
- `max_acceleration_structure_geometries_per_build`, the fixed geometry/range
  scratch reserved for every command unit.

Defaults and maxima are exposed as
`DEFAULT_COMMAND_ALLOCATOR_CAPACITY`,
`DEFAULT_COMMAND_REFERENCES_PER_LIST`,
`DEFAULT_COMMAND_PREPROCESS_PER_LIST`,
`MAX_COMMAND_ALLOCATOR_CAPACITY`,
`MAX_COMMAND_REFERENCES_PER_LIST`, and
`MAX_COMMAND_PREPROCESS_PER_LIST`.

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

`reserve_generated_scratch` and `release_generated_scratch` mutate one
quiescent allocator's reservation for a pipeline and `GeneratedWorkKind`.
They are allocator-confined cold operations. Capacity faults are explicit.

## Transfer and buffer commands

- `cmd_copy_buffer` uses `BufferCopyDesc`.
- `cmd_fill_buffer` fills an aligned span with a 32-bit pattern.
- `cmd_copy_buffer_to_texture` uses `BufferTextureCopyDesc`.
- `cmd_copy_texture_to_buffer` uses `TextureBufferCopyDesc`.

Descriptors identify exact ranges and texture regions. Recording validates
queue support, bounds, usage, alignment, and state according to the contract
policy. Transfers do not insert texture transitions or host visibility
operations.

## Acceleration-structure builds and updates

`cmd_build_acceleration_structure` records one full BLAS or TLAS build outside
a render pass. A BLAS supplies an ordered
`AccelerationStructureGeometryBuildDesc` array matching its immutable creation
schema. A TLAS instead supplies an instance span and count. Both supply
caller-owned build scratch satisfying the queried size and alignment.

`cmd_update_acceleration_structure` updates the destination in place. It is
valid only after a prior full build has completed, when the structure was
created with `allow_update`, and with the same per-geometry primitive counts,
triangle vertex counts and transform presence, or TLAS instance count, as that
completed build. It uses the queried update scratch; there is no separate
source handle.

Both commands are valid on selected compute or graphics queues and invalid
during a render pass or on transfer-only queues. They insert no barrier and do
not allocate scratch. The caller explicitly orders host writes, consecutive
builds/updates, later ray queries, and cross-submit consumers. Under full
validation the command retains the destination and every explicit backing
span until retirement; raw BLAS addresses already packed into TLAS instances
remain caller-owned.

## Compute work

`cmd_bind_pipeline` selects a compatible compute, graphics, or ray-tracing
pipeline in the command state.

- `cmd_dispatch` supplies one root and `Vec3u` group counts.
- `cmd_dispatch_indirect` reads one `DispatchIndirectCommand`.
- `cmd_dispatch_generated` reads generated root/argument records and a count.

Direct counts must fit `DeviceCaps.max_compute_work_group_count`.
Indirect argument storage and count storage are ordinary `GpuSpan` values
whose contents and lifetime are caller-owned.

## Direct ray tracing

Bind a ray-tracing pipeline, then call `cmd_trace_rays` outside a render pass
with one root `GpuAddress`, a caller-owned `RayTracingShaderBindingTable`, and
nonzero `Vec3u` dimensions. The command is valid on a compatible selected
graphics or compute queue. The invocation product must fit
`DeviceCaps.ray_tracing_pipelines.max_ray_dispatch_invocation_count`. Each axis
must also fit the selected device's effective core ray-launch limit; full
validation rejects an oversized width, height, or depth before native
recording.

The ray-generation SBT region contains exactly one record. Optional miss, hit,
and callable regions use canonical empty values when absent; nonempty regions
must satisfy device base/handle alignment, stride, range, ownership, and usage
requirements. Record a ray-tracing pipeline bind before tracing.

`cmd_trace_rays` pushes the root to all six ray stages and emits one direct
trace. It allocates nothing and inserts no pipeline bind, barrier, submission,
or wait. Keep the bound pipeline, every nonempty SBT allocation, root data,
TLAS view, and all raw-address/index targets live through completion.

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
Fixed reference or generated-preprocess limits return
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` or
`GENERATED_SCRATCH_EXHAUSTED` without emitting a partial compound command.
