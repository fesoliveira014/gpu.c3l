# Contiguous texture-index ranges

**Question.** Should the descriptor heap offer contiguous index ranges so
material data can store one base `TextureIndex` plus offsets instead of one
index per texture?

**Decision.** No. Keep independent `TextureIndex` values.

## Addressing model

`TextureIndex` is a 4-byte shader value: backend slot plus one, zero
invalid. `GPU_HEAP_SLOT(index)` in `descriptor_heap.glsl` undoes the offset.
Index space is affine to slot space, so `base + offset` needs no shader
change if contiguity holds.

## Slot allocation today

`commit_texture_descriptor_item` (`gpu/internal/vk/descriptor_heap.c3`)
pops a LIFO free stack when it has entries, otherwise bumps a pointer.

- Fresh heap: a `create_texture_views` batch returns ascending contiguous
  indices. Base-plus-offset works today for load-once asset sets.
- After churn: recycled slots come back in reverse free order. Contiguity is
  not enough; ascending order is a second requirement the heap does not
  provide.

The capacity check counts free slots in total and raises only
`DESCRIPTOR_HEAP_FULL`. No public call can request or reserve contiguity, and
no public value exposes free-list state. A guaranteed range is therefore a
core allocator change and cannot be prototyped above the public API.

## Record size

The library defines no material record. For an application record:

| Record | std430 size |
|---|---:|
| four `TextureIndex` | 16 B |
| one base `TextureIndex` | 4 B |
| `vec4` factor plus four `TextureIndex` | 32 B |
| `vec4` factor plus one base `TextureIndex` | 32 B |

Raw saving is 12 B. Any 16-byte-aligned member (`vec4`, matrix columns)
rounds both shapes to the same size, so the saving is zero for the common
material with a color factor. A `vec2` factor keeps 8 B of it.

## The untested benefit

`create_texture_views` emits one descriptor write per binding per view, up
to `2N`. A contiguous range of uniform usage collapses to at most two writes
with `descriptorCount = N`. The image-info array stays at `2N` entries and the
temporary allocation is unchanged, so only the write array shrinks. Whether
that is measurable needs a device.

## Not collected

CPU publication cost, GPU timing of both representations, fragmentation and
failed-range rates under churn, and shader loads per material. All need a
representative GPU or an allocator that does not exist.

## Reasoning

1. A guaranteed range is unavoidably a core descriptor change.
2. Post-churn contiguity and ordering need a new allocation policy plus a new
   "free slots exist but none contiguous" fault.
3. The record-size saving is typically zero after std430 padding.

## Revisit when

A consuming project shows descriptor-publication cost or GPU-authored
material data as a measured constraint on real hardware.
