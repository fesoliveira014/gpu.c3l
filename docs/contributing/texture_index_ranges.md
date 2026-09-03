# Contiguous texture-index ranges

Whether base-plus-offset bindless addressing is worth a descriptor range
allocator, compared with storing explicit `TextureIndex` values in application
material data.

## Addressing model

`TextureIndex` is a `bitstruct ... : uint` (`gpu/gpu.c3i`), so it is a
4-byte shader value. Zero is the invalid sentinel; a live value is the backend
slot plus one. `texture_index_from_slot` returns `slot + 1`
(`gpu/internal/vk/descriptor_heap.c3:412-414`) and the shader side undoes it
with `GPU_HEAP_SLOT(index) ((index) - 1u)`
(`include/shaders/descriptor_heap.glsl:33`). Index space is therefore affine to
slot space, and `base + offset` needs no shader change if contiguity holds.

## Slot allocation behaviour

`commit_texture_descriptor_item`
(`gpu/internal/vk/descriptor_heap.c3:932-955`) takes a slot from a private LIFO
free stack when one exists, otherwise from a bump pointer:

- fresh heap, no prior frees: every slot comes from `texture_next_new++`, so one
  `create_texture_views` batch already returns contiguous **ascending** indices.
  Base-plus-offset works today for load-once asset sets, with no API change;
- after churn: `recycle_texture_descriptor_slot` pushes freed slots
  (`descriptor_heap.c3:649`) and commit pops them, so a recovered set arrives in
  **reverse free order**, which the application controls and the heap does not
  constrain. Base-plus-offset requires ascending order, so contiguity alone is
  not sufficient; ordering is a second requirement the heap does not provide.

The capacity check is a total-count check, not a contiguity check: headroom is
`texture_free_count + capacity - texture_next_new` and only `DESCRIPTOR_HEAP_FULL`
(`gpu.c3i:177`) is raised (`descriptor_heap.c3:1030-1045`).

## No application can request contiguity

An application can *observe* slots after the fact: `TextureView.index` is
public, and live `TextureIndex` values are documented as the zero-based backend
slot plus one (`gpu.c3i:28-30`). A detect-and-fall-back prototype is therefore
constructible — publish a batch, check whether the returned indices happen to be
contiguous and ascending, and store explicit indices when they are not.

What no public entry point offers is a way to *reserve* or *request*
contiguity, or to observe free-list state before publishing.
`DeviceCaps.texture_heap_capacity` is documented as "Configured semantic heap
capacity, not the hardware maximum" (`gpu.c3i:468-469`), and `MemoryStats`
(`gpu.c3i:776-781`) carries no descriptor occupancy. A guaranteed range is
therefore necessarily a change to core descriptor allocation, and the
fragmentation half of the question cannot be prototyped on top of the existing
batch API at all.

## Material record size

The library defines no material record; `TextureIndex` appears in neither
`abi/core.abi` nor `include/shaders/generated/shader_abi.glsl`. Sizes below are
`c3c` 0.8.0 results for the two application-side shapes:

| Record | C3 size |
| --- | --- |
| four `TextureIndex` fields | 16 B |
| one base `TextureIndex` | 4 B |
| `float[4]` factor + four `TextureIndex` | 32 B |
| `float[4]` factor + one base `TextureIndex` | 20 B |

Shared data is std430 (`docs/shader_abi.md:34-43`). Four bare `uint`s have
4-byte alignment, so the raw saving is 12 B per material. Once the record
carries a 16-byte-aligned member — a `vec4`, or a matrix — std430 rounds the
struct and its array stride up to that alignment, and both variants above become
32 B: the saving is **zero**. A less strictly aligned member keeps part of it;
with a `vec2` factor the shapes are 24 B and 16 B, an 8 B saving. The issue's own
four-texture material with a colour factor falls in the zero case.

## The case for ranges: descriptor-write batching

`accumulate_texture_descriptor_write` (`descriptor_heap.c3:957-999`) emits one
`vk::WriteDescriptorSet` per binding per view — sampled and/or storage, so up to
2 per view. Publishing `N` views costs `sampled_count + storage_count` writes, up
to `2N`. A contiguous range of uniform usage and binding collapses to one write
per binding with `dst_array_element = base` and `descriptorCount = N`: at most 2
writes regardless of `N`.

The saving is the write array only. A merged write still needs one
`vk::DescriptorImageInfo` per descriptor, so the image-info array keeps its
`descs.len * 2` sizing (`descriptor_heap.c3:1082-1084`) and the point at which a
batch spills past `TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP = 64`
(`descriptor_heap.c3:21`) is unchanged. `create_texture_views` also allocates
from `talloc` unconditionally inside `@pool()` (`descriptor_heap.c3:1048-1050`),
so merging avoids no temporary allocation. Whether the reduced write count is
measurable requires a device.

## Not collected

| Item | Reason |
| --- | --- |
| CPU descriptor-publication cost | needs a representative GPU; a software driver dominates library-side cost |
| GPU timing and output correctness of both representations | same |
| Observed fragmentation rate under churn | needs a range allocator, which does not exist |
| Failed-contiguous-allocation frequency | same |
| Prototype allocator host cost | no allocator was built; a guaranteed range is not constructible above the public API |
| Shader data loads or instructions per material | needs the base-plus-offset representation and a device; this is where base-plus-offset plausibly wins, and it is untested |

## Recommendation

Retain independent `TextureIndex` values. This rests on structural analysis, not
measurement:

1. an application can detect contiguity but never request it, so a *guaranteed*
   range is unavoidably a core descriptor change;
2. post-churn contiguity **and** ascending order require an allocation policy
   the heap does not have, plus a distinct "free slots exist but none
   contiguous" fault alongside `DESCRIPTOR_HEAP_FULL`;
3. the record-size saving is 12 B raw and typically 0 B after std430 padding.

The one untested benefit is descriptor-write batching. Revisit if a consuming
project shows descriptor-publication cost, or GPU-authored material data,
as an actual constraint — measured on real hardware.
