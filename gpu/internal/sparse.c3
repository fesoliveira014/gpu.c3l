module gpu::internal @private;

import gpu;

fn bool sparse_ranges_overlap(
    usz a_offset,
    usz a_size,
    usz b_offset,
    usz b_size,
) @private {
    if (a_size == 0 || b_size == 0) return false;
    if (a_offset <= b_offset) return b_offset - a_offset < a_size;
    return a_offset - b_offset < b_size;
}

fn bool sparse_tile_boxes_overlap(
    SparseTextureTileBind* a,
    SparseTextureTileBind* b,
) @private {
    if (a.aspect != b.aspect || a.mip_level != b.mip_level) return false;
    return sparse_ranges_overlap(a.offset.x, a.extent.x, b.offset.x, b.extent.x)
        && sparse_ranges_overlap(a.offset.y, a.extent.y, b.offset.y, b.extent.y)
        && sparse_ranges_overlap(a.offset.z, a.extent.z, b.offset.z, b.extent.z);
}

fn bool sparse_unbind_offset_valid(
    GpuAllocation allocation,
    usz allocation_offset,
) @private {
    bool unbind = allocation.owner == 0
        && allocation.index == 0
        && allocation.generation == 0;
    return !unbind || allocation_offset == 0;
}

<* A sparse texture description: valid single-layer, uncompressed, non-depth
   texture with no attachment usage. *>
fn bool sparse_texture_desc_valid(TextureDesc* desc) @private {
    if (!texture_desc_valid(desc)) return false;
    return texture_array_layers(desc) == 1
        && !format_is_depth_stencil(desc.format)
        && !format_is_block_compressed(desc.format)
        && !desc.usage.color_attach
        && !desc.usage.depth_attach;
}

<* Rules on a sparse bind description that read no texture or device state:
   slices with a length have a pointer; tile, opaque and wait counts stay within
   `MAX_SPARSE_TILE_BINDS`, `MAX_SPARSE_OPAQUE_BINDS` and
   `MAX_SPARSE_BIND_WAITS`; at least one tile or opaque bind; tile binds name
   the color aspect, a nonempty extent and an offset that fits int; opaque
   binds name a defined aspect and a nonempty range; an unbind has offset 0;
   tile boxes of one aspect and mip do not intersect; and opaque ranges of one
   aspect do not intersect. *>
fn bool sparse_bind_desc_valid(SparseTextureBindDesc* desc) @private {
    if ((desc.tiles.len > 0 && desc.tiles.ptr == null)
        || (desc.opaque.len > 0 && desc.opaque.ptr == null)
        || (desc.waits.len > 0 && desc.waits.ptr == null)
        || desc.tiles.len > gpu::MAX_SPARSE_TILE_BINDS
        || desc.opaque.len > gpu::MAX_SPARSE_OPAQUE_BINDS
        || desc.waits.len > gpu::MAX_SPARSE_BIND_WAITS
        || desc.tiles.len + desc.opaque.len == 0) {
        return false;
    }
    foreach (i, &bind : desc.tiles) {
        if (bind.aspect != SparseTextureAspect.COLOR
            || bind.extent.x == 0
            || bind.extent.y == 0
            || bind.extent.z == 0
            || bind.offset.x > (uint)int::max
            || bind.offset.y > (uint)int::max
            || bind.offset.z > (uint)int::max
            || !sparse_unbind_offset_valid(
                bind.allocation,
                bind.allocation_offset,
            )) {
            return false;
        }
        for (sz prior = 0; prior < i; prior++) {
            if (sparse_tile_boxes_overlap(&desc.tiles[prior], bind)) {
                return false;
            }
        }
    }
    foreach (i, &bind : desc.opaque) {
        if ((bind.aspect != SparseTextureAspect.COLOR
                && bind.aspect != SparseTextureAspect.METADATA)
            || bind.size == 0
            || !sparse_unbind_offset_valid(
                bind.allocation,
                bind.allocation_offset,
            )) {
            return false;
        }
        for (sz prior = 0; prior < i; prior++) {
            if (desc.opaque[prior].aspect == bind.aspect
                && sparse_ranges_overlap(
                    desc.opaque[prior].resource_offset,
                    desc.opaque[prior].size,
                    bind.resource_offset,
                    bind.size,
                )) {
                return false;
            }
        }
    }
    return true;
}
