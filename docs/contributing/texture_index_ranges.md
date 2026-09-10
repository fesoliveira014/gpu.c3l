# Contiguous texture-index ranges

[Documentation](../index.md) › Contributing › Contiguous texture-index ranges

**Question.** Should the descriptor heap offer contiguous index ranges so
material data can store one base `TextureIndex` plus offsets, and so
row-addressed layouts (a double-buffered G-buffer at
`row * texture_count + i`) can pick their own slots?

**Decision.** Reservation region only; the general allocator is unchanged.
`reserve_texture_indices` carves `count` contiguous ascending slots from a
region at the top of the heap that grows downward. `create_texture_view_at`
publishes into one reserved slot; `update_texture_view` rewrites a slot's
descriptor without changing its index. `create_texture_view` never returns a
reserved slot.

## Addressing model

`TextureIndex` is a 4-byte shader value: backend slot plus one, zero
invalid. `GPU_HEAP_SLOT(index)` in `descriptor_heap.glsl` undoes the offset.
Index space is affine to slot space, so `base + offset` needs no shader
change inside a reservation.

## Slot allocation

`commit_texture_descriptor_item` (`gpu/internal/vk/descriptor_heap.c3`)
pops a LIFO free stack when it has entries, otherwise bumps
`texture_next_new`. The bump pointer stops at `texture_reserved_low`, the
boundary of the reservation region. Headroom is
`free_count + (texture_reserved_low - texture_next_new)`.

Reservations follow stack discipline. Releasing the lowest reservation
moves the boundary up past it and past any released reservations above it.
Releasing a higher reservation clears its cells and leaves a hole until the
reservations below it go. No reordering of the free stack, no compaction,
no new fault type: exhaustion is `DESCRIPTOR_HEAP_FULL` on either side.

A reserved slot keeps its `reserved` bit across `destroy_texture_view`; the
slot does not enter the free stack and accepts a later
`create_texture_view_at`. The heap uses `PARTIALLY_BOUND`, so a destroyed
slot's descriptor is not rewritten; reading a destroyed slot is the same
hazard as on the general path.

## In-place update

`update_texture_view` resolves the new texture's cached native view, writes
the descriptor at the existing slot, moves the live-view count between the
textures, and advances the cell generation. The caller's token is rewritten;
older copies fault `INVALID_HANDLE`. The old native view stays in its
texture's cache, which is released with the texture.

## What was rejected

- A range allocator with best-fit or compaction over the general region:
  needs a new allocation policy and a "free slots exist but none contiguous"
  fault, for a record-size saving that std430 padding usually erases.
- Sampler reservations: samplers are interned and never freed.

## Revisit when

A consuming project needs reservations to outlive the stack discipline
(interleaved lifetimes with large holes), or shows descriptor-publication
cost as a measured constraint on real hardware.
