module gpu::internal @private;

import gpu;


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
            || desc.sample_count != SampleCount.ONE
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

    bool is_depth = desc.format == Format.D32_FLOAT;
    if (is_depth && (desc.usage.color_attach || desc.usage.storage)) return false;
    if (!is_depth && desc.usage.depth_attach) return false;

    return true;
}
