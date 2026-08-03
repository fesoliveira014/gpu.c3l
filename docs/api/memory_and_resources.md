# Memory and resources

## Allocations, spans, and addresses

`GpuAllocation` owns one device-scoped storage allocation.
`allocate_memory` consumes an `AllocationDesc` containing size, alignment,
`MemoryClass`, semantic access, and an optional debug name.
`free_allocation` invalidates the owner only on success and never waits for GPU
use.

`get_allocation_info` reports the actual class, size, alignment, mapping state,
and related properties. `get_allocation_span` returns a non-owning `GpuSpan`
covering the allocation.

`GpuSpan.checked_subspan` and `MappedGpuSpan.checked_subspan` validate range
arithmetic; `unchecked_subspan` is for already-proved bounds.
`mapped_gpu_span` combines a checked span, mapping, and address for mapped
addressable storage.

`get_span_mapping` returns a borrowed byte slice for a host-visible span.
`get_span_address` returns a `GpuAddress` for addressable storage. Both values
inherit the allocation lifetime. Flush CPU-written ranges before submission
and invalidate completed GPU-written ranges before reading. Coherent memory
may make these calls no-ops.

Allocation and mapping operations are thread-safe, but applications synchronize
writes to overlapping mapped bytes. Freeing storage happens after its last GPU
and host use.

## Memory classes

| Class | Intended use |
|---|---|
| `CPU_WRITE` | persistently mapped upload and root data |
| `GPU_PRIVATE` | device-local addressable data |
| `CPU_READ` | persistently mapped readback |
| `TEXTURE` | placed texture backing; not a generic GPU-address source |

`MemoryStats` contains up to `MAX_MEMORY_HEAPS` advisory
`MemoryHeapBudget` records. Exact reporting calls are described in
[Presentation and diagnostics](presentation_and_diagnostics.md#memory-and-debug-reporting).

## Texture capability and creation

`get_texture_format_support` reports independently supported usages,
filterability, and sample counts for one `Format`.
`supports_texture_desc` preflights a complete `TextureDesc`.
`get_texture_requirements` returns size, alignment, and a
`TextureCompatibility` value for placed storage.

Texture creation forms are:

- `create_texture`: transactional dedicated storage hidden behind a handle;
- `create_dedicated_texture`: returns `DedicatedTexture`, exposing both the
  texture and its owner allocation;
- `create_placed_texture`: borrows a compatible texture allocation range; and
- `create_sparse_texture`: creates an image with no committed residency.

`destroy_texture` releases the texture object, not a separately returned or
placed allocation. Keep backing allocations alive until every texture use and
the texture itself have ended.

`TextureDesc` defines dimensions, mips, format, sample count, usage, and
debug name. Zero depth selects 2D; positive depth selects ordinary 3D.
Capabilities and exclusions are summarized in
[Features and limitations](../features_and_limitations.md).

## Sparse textures

`get_sparse_texture_requirements` returns the cached color and optional
metadata aspect shapes described by `SparseTextureRequirements`.
`SparseTextureBindDesc` supplies tile and opaque-tail binds plus completion
waits. `bind_sparse_texture_memory` targets one selected queue and returns a
`CompletionPoint`.

Binding is externally synchronized on that queue. It does not allocate memory,
track a residency map, or retain arbitrary backing bytes. The caller prevents
overlap, retains bound storage, and orders unbind/replacement after all prior
users. Invalid geometry or incompatible allocations fail before a native bind
is accepted.

## Acceleration structures

Ray-query-enabled devices expose bottom-level acceleration structures (BLAS)
and top-level acceleration structures (TLAS) through one strongly typed
`AccelerationStructureHandle`. A BLAS descriptor contains one or more ordered
triangle geometries or one or more ordered AABB geometries; one BLAS cannot mix
the two kinds. A TLAS descriptor instead supplies `max_instance_count`.
Descriptors are immutable capacity schemas: every later build/update uses the
same order, kind, index type, and counts no larger than the declared maxima.

Triangle inputs are float32 XYZ vertices with a byte stride and either no
indices or U16/U32 indices. Non-indexed inputs need three vertices per
primitive. The optional transform is one row-major 3-by-4 float matrix. AABB
inputs are records of six floats—`min_xyz` then `max_xyz`—with a byte stride of
at least 24. Procedural intersection and candidate confirmation happen in the
shader.

Call `get_acceleration_structure_requirements` before allocating storage. It
reports persistent storage size/alignment, full-build scratch, update scratch,
and scratch alignment. Update scratch is zero unless `allow_update` was set.
Creation forms are:

- `create_acceleration_structure`, with hidden GPU-private storage;
- `create_placed_acceleration_structure`, in a nonoverlapping compatible range
  of a caller-owned generic allocation; and
- `create_dedicated_acceleration_structure`, returning the structure and its
  explicit `GpuAllocation` owner.

Placed and dedicated allocations remain explicit owners. Destroy the
structure before freeing its allocation. A live placement makes
`free_allocation` return `RESOURCE_IN_USE`; destruction releases the placement
but never frees caller-owned storage or waits.

Build inputs and scratch are caller-owned `GpuSpan` values. Build scratch and
update scratch are different queried contracts. Keep every input, transform,
instance record, scratch range, and structure live until the covering
submission completes. The library allocates no hidden scratch and inserts no
barrier.

`make_acceleration_structure_instance` validates a live BLAS and returns the
exact 64-byte `AccelerationStructureInstance` ABI. Its address field is an
ordinary `GpuAddress`; its transform is three row-major `Vec4f` rows. Custom
index is 24 bits, mask is 8 bits, and the shader-binding-table offset must be
zero in this ray-query-only API. Store packed records in addressable instance
input memory before building a TLAS.

`create_acceleration_structure_view` publishes a TLAS in the independently
sized shader heap. The owner-bearing `AccelerationStructureView` contains a raw
`AccelerationStructureIndex`. As with texture indices, the raw index has no
owner/generation and may be recycled immediately after view destruction. Wait
for all shader use, destroy the view, then destroy the TLAS. Destroy BLAS values
only after all TLAS build/update work that reads their packed addresses has
completed.

Updates are in-place only. The structure must have been created with
`allow_update`, completed one full build, retain the identical schema, and use
the queried update scratch. There is no distinct source structure or implicit
rebuild.

## Views and bindless indices

`create_texture_view` validates a `TextureViewDesc`, publishes the descriptor,
and returns an owner-bearing `TextureView`. `create_texture_views` performs a
batch described by `TextureViewCreateDesc` values and publishes no partial
batch on failure. `destroy_texture_view` releases its slot.

`TextureView.index` is the raw `TextureIndex` stored in shader data. It carries
no owner or generation. Destroying the view makes the index immediately
recyclable, so wait for all shader uses before destruction.

`AccelerationStructureView.index` follows the same raw-value rule for TLAS
descriptors, while the view itself blocks TLAS destruction until retired.

Attachment views are a rendering-domain resource documented in
[Commands and rendering](commands_and_rendering.md#attachments-and-render-passes).

## Samplers

`intern_sampler` validates `SamplerDesc` and returns the stable device-wide
`SamplerIndex` for equivalent state. Equal concurrent requests converge.
There is no individual sampler destroy call; indices remain live until device
destruction.

Sampler state covers filters, address modes, LOD range and bias, compare
operation, and optional anisotropy. Requested values are checked against
`DeviceCaps`; they are never silently clamped.

## Formats and support values

`Format`, `SampleCount`, `TextureUsage`, `TextureFormatFeatures`,
`TextureSampleCountSupport`, and `TextureFormatSupport` describe public
texture behavior. `Filter` and `AddressMode` describe sampler behavior.
`TextureRequirements` and `TextureCompatibility` are only for matching placed
textures to allocation ranges; compatibility is not a general format identity.

## Fault and lifetime summary

| Cause | Typical fault |
|---|---|
| invalid range, alignment, descriptor, or requested limit | `INVALID_ARGUMENT` |
| stale, foreign, or zero owner | `INVALID_HANDLE` |
| unsupported usage/format/capability | `UNSUPPORTED_FEATURE` |
| no host/device memory | `OUT_OF_HOST_MEMORY` / `OUT_OF_DEVICE_MEMORY` |
| fixed resource table exhausted | `SLOT_TABLE_FULL` |
| bindless heap exhausted | `DESCRIPTOR_HEAP_FULL` |
| live retained use blocks destruction | `RESOURCE_IN_USE` |

Resource creation is internally synchronized and transactional. Query values
are snapshots; memory budget reports may be advisory under concurrent
allocation mutation.
