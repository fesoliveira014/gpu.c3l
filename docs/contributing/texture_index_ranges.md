# Contiguous texture-index ranges

Whether base-plus-offset bindless addressing is worth a descriptor range
allocator, compared with storing explicit `TextureIndex` values in application
material data.

## Addressing model

`TextureIndex` is a `bitstruct ... : uint` (`gpu/gpu.c3i:31-33`), so it is a
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
  (`descriptor_heap.c3:649`) and commit pops them, so a recovered contiguous set
  arrives in reverse push order — **descending**. Base-plus-offset requires
  ascending, so contiguity alone is not sufficient; ordering is a second
  requirement the heap does not provide.

The capacity check is a total-count check, not a contiguity check: headroom is
`texture_free_count + capacity - texture_next_new` and only `DESCRIPTOR_HEAP_FULL`
(`gpu.c3i:177`) is raised (`descriptor_heap.c3:1030-1045`).

## No application-local prototype is constructible

No public entry point lets an application choose, reserve, or observe descriptor
slots. `DeviceCaps.texture_heap_capacity` is documented as "Configured semantic
heap capacity, not the hardware maximum" (`gpu.c3i:468-469`), and `MemoryStats`
(`gpu.c3i:776-781`) carries no descriptor occupancy. Any range feature is
therefore necessarily a change to core descriptor allocation, not something an
application can prototype on top of the existing batch API.

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
carries any vector member its alignment is 16, and std430 rounds the struct — and
its array stride — up to that alignment: both vector variants become 32 B. For a
realistic material the saving is **zero**.

## The case for ranges: descriptor-write batching

`accumulate_texture_descriptor_write` (`descriptor_heap.c3:957-998`) emits one
`VkWriteDescriptorSet` per binding per view — sampled and/or storage, so up to 2
per view. Publishing `N` views costs `sampled_count + storage_count` writes, up
to `2N`. A contiguous range of uniform usage and binding collapses to one write
per binding with `dst_array_element = base` and `descriptorCount = N`: at most 2
writes regardless of `N`.

That also bounds scratch. `max_writes = descs.len * 2` against
`TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP = 64` (`descriptor_heap.c3:21`,
`:1076-1084`) means batches above 32 views spill to `talloc`; a merged range
write would stay on the stack. Whether either effect is measurable requires a
device.

## Not collected

| Item | Reason |
| --- | --- |
| CPU descriptor-publication cost | needs a representative GPU; a software driver dominates library-side cost |
| GPU timing and output correctness of both representations | same |
| Observed fragmentation rate under churn | needs a range allocator, which does not exist |
| Failed-contiguous-allocation frequency | same |
| Prototype allocator host cost | no allocator was built; none is constructible above the public API |

## Recommendation

Retain independent `TextureIndex` values. This rests on structural analysis, not
measurement:

1. no application-local prototype is constructible, so the feature is
   unavoidably a core descriptor change;
2. post-churn contiguity **and** ascending order require an allocation policy
   the heap does not have, plus a distinct "free slots exist but none
   contiguous" fault alongside `DESCRIPTOR_HEAP_FULL`;
3. the record-size saving is 12 B raw and typically 0 B after std430 padding.

The one untested benefit is descriptor-write batching. Revisit if a consuming
project shows descriptor-publication cost, or GPU-authored material data,
as an actual constraint — measured on real hardware.
