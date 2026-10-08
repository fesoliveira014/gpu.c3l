module gpu::internal @private;

import gpu;

<* Texel block footprint of a format. Uncompressed formats are 1x1 blocks. *>
struct FormatBlock {
    uint width;
    uint height;
    usz  bytes;
}

fn FormatBlock format_block(Format format) @private {
    switch (format) {
        case Format.R8_UNORM:
        case Format.R8_UINT:
            return { 1, 1, 1 };
        case Format.RG8_UNORM:
        case Format.R16_UINT:
        case Format.R16_FLOAT:
            return { 1, 1, 2 };
        case Format.RGBA8_UNORM:
        case Format.RGBA8_SRGB:
        case Format.BGRA8_UNORM:
        case Format.BGRA8_SRGB:
        case Format.RG16_FLOAT:
        case Format.R32_UINT:
        case Format.R32_FLOAT:
        case Format.D32_FLOAT:
            return { 1, 1, 4 };
        case Format.S8_UINT:
            return { 1, 1, 1 };
        case Format.RGBA16_FLOAT:
        case Format.RG32_FLOAT:
            return { 1, 1, 8 };
        case Format.RGBA32_FLOAT:
            return { 1, 1, 16 };
        case Format.BC1_RGBA_UNORM:
        case Format.BC1_RGBA_SRGB:
        case Format.BC4_UNORM:
            return { 4, 4, 8 };
        case Format.BC3_UNORM:
        case Format.BC3_SRGB:
        case Format.BC5_UNORM:
        case Format.BC6H_UFLOAT:
        case Format.BC7_UNORM:
        case Format.BC7_SRGB:
            return { 4, 4, 16 };
        default:
            return { 0, 0, 0 };
    }
}

fn bool format_has_depth(Format format) @private {
    return format == Format.D32_FLOAT
        || format == Format.D24_UNORM_S8_UINT
        || format == Format.D32_FLOAT_S8_UINT;
}

fn bool format_has_stencil(Format format) @private {
    return format == Format.D24_UNORM_S8_UINT
        || format == Format.D32_FLOAT_S8_UINT
        || format == Format.S8_UINT;
}

fn bool format_is_depth_stencil(Format format) @private {
    return format_has_depth(format) || format_has_stencil(format);
}

fn bool format_is_combined_depth_stencil(Format format) @private {
    return format_has_depth(format) && format_has_stencil(format);
}

<* Footprint of one aspect as the copy commands consume it: 4 bytes for the
   depth aspect of a combined format, 1 for stencil, else the format block. *>
fn FormatBlock format_aspect_block(Format format, TextureAspect aspect) @private {
    if (!format_is_combined_depth_stencil(format)) return format_block(format);
    switch (aspect) {
        case TextureAspect.DEPTH:   return { 1, 1, 4 };
        case TextureAspect.STENCIL: return { 1, 1, 1 };
        default:                    return { 0, 0, 0 };
    }
}

fn bool texture_aspect_valid(TextureAspect aspect) @private {
    switch (aspect) {
        case TextureAspect.AUTO:
        case TextureAspect.DEPTH:
        case TextureAspect.STENCIL:
            return true;
        default:
            return false;
    }
}

fn bool format_is_block_compressed(Format format) @private {
    FormatBlock block = format_block(format);
    return block.width > 1 || block.height > 1;
}

const uint CUBE_FACE_COUNT = 6;

fn ulong texture_compatibility_owner(TextureCompatibility compatibility) @private {
    return (ulong)((uint128)compatibility >> 64);
}

fn uint texture_compatibility_value(TextureCompatibility compatibility) @private {
    return (uint)(uint128)compatibility;
}

fn uint texture_mip_levels(TextureDesc* desc) @private {
    return desc.mip_levels == 0 ? 1 : desc.mip_levels;
}

fn uint texture_array_layers(TextureDesc* desc) @private {
    return desc.array_layers == 0 ? 1 : desc.array_layers;
}

fn bool texture_desc_is_3d(TextureDesc* desc) @private {
    return desc.depth > 0;
}

fn uint texture_desc_native_depth(TextureDesc* desc) @private {
    return texture_desc_is_3d(desc) ? desc.depth : 1;
}

fn uint texture_full_mip_chain(
    uint width,
    uint height,
    uint depth,
) @private {
    uint levels = 1;
    uint size = width > height ? width : height;
    if (depth > size) size = depth;
    while (size > 1) {
        size >>= 1;
        levels++;
    }
    return levels;
}

fn bool texture_format_in_backend_profile(Format format) @private {
    switch (format) {
        case Format.R8_UNORM:
        case Format.R8_UINT:
        case Format.RG8_UNORM:
        case Format.RGBA8_UNORM:
        case Format.RGBA8_SRGB:
        case Format.BGRA8_UNORM:
        case Format.BGRA8_SRGB:
        case Format.R16_UINT:
        case Format.R16_FLOAT:
        case Format.RG16_FLOAT:
        case Format.RGBA16_FLOAT:
        case Format.R32_UINT:
        case Format.R32_FLOAT:
        case Format.RG32_FLOAT:
        case Format.RGBA32_FLOAT:
        case Format.D32_FLOAT:
        case Format.BC1_RGBA_UNORM:
        case Format.BC1_RGBA_SRGB:
        case Format.BC3_UNORM:
        case Format.BC3_SRGB:
        case Format.BC4_UNORM:
        case Format.BC5_UNORM:
        case Format.BC6H_UFLOAT:
        case Format.BC7_UNORM:
        case Format.BC7_SRGB:
        case Format.D24_UNORM_S8_UINT:
        case Format.D32_FLOAT_S8_UINT:
        case Format.S8_UINT:
            return true;
        default:
            return false;
    }
}

fn bool sample_count_valid(SampleCount sample_count) @private {
    switch (sample_count) {
        case SampleCount.ONE:
        case SampleCount.TWO:
        case SampleCount.FOUR:
        case SampleCount.EIGHT:
        case SampleCount.SIXTEEN:
        case SampleCount.THIRTY_TWO:
        case SampleCount.SIXTY_FOUR:
            return true;
        default:
            return false;
    }
}

fn bool texture_desc_valid(TextureDesc* desc) @private {
    if (desc.width == 0 || desc.height == 0) return false;
    if (texture_mip_levels(desc) > texture_full_mip_chain(
        desc.width,
        desc.height,
        texture_desc_native_depth(desc),
    )) return false;
    if (!texture_format_in_backend_profile(desc.format)) return false;

    bool has_usage = desc.usage.sampled || desc.usage.storage || desc.usage.color_attach
        || desc.usage.depth_attach || desc.usage.transfer_src || desc.usage.transfer_dst;
    if (!has_usage) return false;

    if (!sample_count_valid(desc.sample_count)) return false;
    if (texture_desc_is_3d(desc)
        && (texture_array_layers(desc) != 1
            || desc.usage.color_attach
            || desc.usage.depth_attach)) {
        return false;
    }
    if (desc.sample_count != SampleCount.ONE
        && (texture_mip_levels(desc) != 1
            || desc.usage.sampled
            || desc.usage.storage
            || desc.usage.transfer_src
            || desc.usage.transfer_dst
            || (!desc.usage.color_attach && !desc.usage.depth_attach))) {
        return false;
    }

    bool is_depth = format_is_depth_stencil(desc.format);
    if (is_depth
        && (desc.usage.color_attach
            || desc.usage.storage
            || desc.cube_compatible
            || texture_desc_is_3d(desc))) {
        return false;
    }
    if (!is_depth && desc.usage.depth_attach) return false;

    if (desc.cube_compatible
        && (texture_desc_is_3d(desc)
            || desc.width != desc.height
            || texture_array_layers(desc) % CUBE_FACE_COUNT != 0
            || desc.sample_count != SampleCount.ONE)) {
        return false;
    }

    if (format_is_block_compressed(desc.format)
        && (texture_desc_is_3d(desc)
            || desc.usage.storage
            || desc.usage.color_attach
            || desc.usage.depth_attach
            || desc.sample_count != SampleCount.ONE)) {
        return false;
    }

    return true;
}

<* Shape of one texture as a texture-to-texture copy sees it. A positive
   depth marks a three-dimensional texture; counts are normalized. *>
struct TextureCopyShape {
    Format       format;
    TextureUsage usage;
    SampleCount  sample_count;
    uint         width;
    uint         height;
    uint         depth;
    uint         mip_levels;
    uint         array_layers;
}

<* Copy extents after zero-value defaults; two-dimensional copies have depth one. *>
struct TextureCopyExtent {
    uint width;
    uint height;
    uint depth;
    uint layers;
}

fn bool texture_copy_axis_valid(
    uint offset,
    uint size,
    uint block,
    uint mip_size,
) @private {
    if (offset > mip_size || size > mip_size - offset) return false;
    if (offset % block != 0) return false;
    return size % block == 0 || offset + size == mip_size;
}

fn bool texture_copy_ranges_overlap(
    uint first,
    uint first_size,
    uint second,
    uint second_size,
) @private {
    return first < second + second_size && second < first + first_size;
}

<* Validate a texture-to-texture copy region against both texture shapes.
   Regions inside one texture must not overlap. *>
fn TextureCopyExtent? resolve_texture_copy(
    TextureCopyShape* src,
    TextureCopyShape* dst,
    TextureCopyDesc* desc,
    bool same_texture,
) @private {
    if (src.format != dst.format || src.sample_count != dst.sample_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!src.usage.transfer_src || !dst.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (desc.src_mip >= src.mip_levels || desc.dst_mip >= dst.mip_levels) {
        return gpu::INVALID_ARGUMENT~;
    }

    TextureAspect aspect = desc.aspect;
    if (!texture_aspect_valid(aspect)) return gpu::INVALID_ARGUMENT~;
    if (aspect == TextureAspect.DEPTH && !format_has_depth(src.format)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (aspect == TextureAspect.STENCIL && !format_has_stencil(src.format)) {
        return gpu::INVALID_ARGUMENT~;
    }
    FormatBlock block = format_aspect_block(src.format, aspect);
    if (block.bytes == 0) return gpu::INVALID_ARGUMENT~;

    bool is_3d = src.depth > 0;
    if (is_3d != (dst.depth > 0)) return gpu::INVALID_ARGUMENT~;

    uint src_width = gpu::texture_mip_dimension(src.width, desc.src_mip);
    uint src_height = gpu::texture_mip_dimension(src.height, desc.src_mip);
    uint dst_width = gpu::texture_mip_dimension(dst.width, desc.dst_mip);
    uint dst_height = gpu::texture_mip_dimension(dst.height, desc.dst_mip);
    uint src_depth = is_3d ? gpu::texture_mip_dimension(src.depth, desc.src_mip) : 1;
    uint dst_depth = is_3d ? gpu::texture_mip_dimension(dst.depth, desc.dst_mip) : 1;

    if (desc.src_x > src_width || desc.src_y > src_height) return gpu::INVALID_ARGUMENT~;
    uint width = desc.width == 0 ? src_width - desc.src_x : desc.width;
    uint height = desc.height == 0 ? src_height - desc.src_y : desc.height;

    uint layers = 1;
    uint depth = 1;
    if (is_3d) {
        if (desc.src_base_layer != 0 || desc.dst_base_layer != 0 || desc.layer_count > 1) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (desc.src_z > src_depth) return gpu::INVALID_ARGUMENT~;
        depth = desc.depth == 0 ? src_depth - desc.src_z : desc.depth;
        if (depth == 0 || depth > src_depth - desc.src_z
            || desc.dst_z > dst_depth || depth > dst_depth - desc.dst_z) {
            return gpu::INVALID_ARGUMENT~;
        }
    } else {
        if (desc.src_z != 0 || desc.dst_z != 0 || desc.depth > 1) return gpu::INVALID_ARGUMENT~;
        layers = desc.layer_count == 0 ? 1 : desc.layer_count;
        if (desc.src_base_layer > src.array_layers
            || layers > src.array_layers - desc.src_base_layer
            || desc.dst_base_layer > dst.array_layers
            || layers > dst.array_layers - desc.dst_base_layer) {
            return gpu::INVALID_ARGUMENT~;
        }
    }

    if (width == 0 || height == 0) return gpu::INVALID_ARGUMENT~;
    if (!texture_copy_axis_valid(desc.src_x, width, block.width, src_width)
        || !texture_copy_axis_valid(desc.dst_x, width, block.width, dst_width)
        || !texture_copy_axis_valid(desc.src_y, height, block.height, src_height)
        || !texture_copy_axis_valid(desc.dst_y, height, block.height, dst_height)) {
        return gpu::INVALID_ARGUMENT~;
    }

    if (same_texture && desc.src_mip == desc.dst_mip
        && texture_copy_ranges_overlap(desc.src_x, width, desc.dst_x, width)
        && texture_copy_ranges_overlap(desc.src_y, height, desc.dst_y, height)
        && texture_copy_ranges_overlap(desc.src_z, depth, desc.dst_z, depth)
        && texture_copy_ranges_overlap(desc.src_base_layer, layers, desc.dst_base_layer, layers)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return { .width = width, .height = height, .depth = depth, .layers = layers };
}
