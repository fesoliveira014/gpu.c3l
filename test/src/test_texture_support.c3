module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;

fn gpu::TextureDesc valid_texture_desc() {
    return {
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
}

fn void texture_desc_profile_accepts_normalized_defaults() @test {
    gpu::TextureDesc desc = valid_texture_desc();
    assert(gpu::internal::texture_desc_valid(&desc), "zero mip and layer counts normalize to one");
    assert(gpu::internal::texture_mip_levels(&desc) == 1, "zero mip count normalizes to one");
    assert(gpu::internal::texture_array_layers(&desc) == 1, "zero layer count normalizes to one");
    assert(!gpu::internal::texture_desc_is_3d(&desc),
        "zero depth preserves two-dimensional descriptors");
    assert(gpu::internal::texture_desc_native_depth(&desc) == 1,
        "two-dimensional descriptors lower to native depth one");
}

fn void texture_desc_profile_validates_three_dimensional_shape() @test {
    gpu::TextureDesc desc = valid_texture_desc();
    desc.width = 1;
    desc.height = 1;
    desc.depth = 8;
    desc.mip_levels = 4;
    assert(gpu::internal::texture_desc_is_3d(&desc)
        && gpu::internal::texture_desc_native_depth(&desc) == 8
        && gpu::internal::texture_desc_valid(&desc),
        "positive depth selects a valid three-dimensional mip chain");

    desc.depth = 1;
    assert(gpu::internal::texture_desc_is_3d(&desc)
        && gpu::internal::texture_desc_native_depth(&desc) == 1,
        "depth one remains a genuine three-dimensional descriptor");

    desc.depth = 8;
    desc.mip_levels = 5;
    assert(!gpu::internal::texture_desc_valid(&desc),
        "three-dimensional mip validation includes depth");
    desc.mip_levels = 4;
    desc.array_layers = 2;
    assert(!gpu::internal::texture_desc_valid(&desc),
        "three-dimensional textures cannot be arrays");
    desc.array_layers = 1;
    desc.sample_count = gpu::SampleCount.TWO;
    assert(!gpu::internal::texture_desc_valid(&desc),
        "three-dimensional textures are single-sampled");
    desc.sample_count = gpu::SampleCount.ONE;
    desc.usage = { .color_attach };
    assert(!gpu::internal::texture_desc_valid(&desc),
        "three-dimensional color attachments are not exposed");
    desc.format = gpu::Format.D32_FLOAT;
    desc.usage = { .depth_attach };
    assert(!gpu::internal::texture_desc_valid(&desc),
        "three-dimensional depth attachments are not exposed");
}

fn void texture_desc_profile_rejects_invalid_shape() @test {
    gpu::TextureDesc desc = valid_texture_desc();

    desc.width = 0;
    assert(!gpu::internal::texture_desc_valid(&desc), "zero width is invalid");
    desc = valid_texture_desc();
    desc.height = 0;
    assert(!gpu::internal::texture_desc_valid(&desc), "zero height is invalid");
    desc = valid_texture_desc();
    desc.mip_levels = 6;
    assert(!gpu::internal::texture_desc_valid(&desc), "mip count beyond the full chain is invalid");
}

fn void texture_desc_profile_rejects_invalid_usage_roles() @test {
    gpu::TextureDesc desc = valid_texture_desc();

    desc.usage = {};
    assert(!gpu::internal::texture_desc_valid(&desc), "empty usage is invalid");
    desc = valid_texture_desc();
    desc.format = gpu::Format.D32_FLOAT;
    desc.usage = { .storage };
    assert(!gpu::internal::texture_desc_valid(&desc), "depth storage is invalid");
    desc.usage = { .color_attach };
    assert(!gpu::internal::texture_desc_valid(&desc), "depth color attachment is invalid");
    desc = valid_texture_desc();
    desc.usage = { .depth_attach };
    assert(!gpu::internal::texture_desc_valid(&desc), "color depth attachment is invalid");
}

fn void texture_desc_profile_rejects_unsupported_formats() @test {
    gpu::TextureDesc desc = valid_texture_desc();
    desc.format = gpu::Format.UNDEFINED;
    assert(!gpu::internal::texture_desc_valid(&desc), "undefined format is invalid");
}

fn void format_blocks_are_texel_or_bc_sized() @test {
    gpu::internal::FormatBlock rgba8 = gpu::internal::format_block(gpu::Format.RGBA8_UNORM);
    assert(rgba8.width == 1 && rgba8.height == 1 && rgba8.bytes == 4,
        "uncompressed formats are one-texel blocks");
    gpu::internal::FormatBlock bc1 = gpu::internal::format_block(gpu::Format.BC1_RGBA_SRGB);
    assert(bc1.width == 4 && bc1.height == 4 && bc1.bytes == 8,
        "BC1 packs a 4x4 block in 8 bytes");
    gpu::internal::FormatBlock bc7 = gpu::internal::format_block(gpu::Format.BC7_UNORM);
    assert(bc7.width == 4 && bc7.height == 4 && bc7.bytes == 16,
        "BC7 packs a 4x4 block in 16 bytes");
    assert(gpu::internal::format_block(gpu::Format.UNDEFINED).bytes == 0,
        "undefined formats have no block footprint");
    assert(!gpu::internal::format_is_block_compressed(gpu::Format.D32_FLOAT)
            && gpu::internal::format_is_block_compressed(gpu::Format.BC6H_UFLOAT),
        "block compression is derived from the block extent");
}

fn void texture_desc_profile_validates_block_compressed_usage() @test {
    gpu::TextureDesc desc = valid_texture_desc();
    desc.width = 6;
    desc.height = 6;
    desc.format = gpu::Format.BC7_SRGB;
    assert(gpu::internal::texture_desc_valid(&desc),
        "sampled BC textures accept edges that are not block multiples");
    desc.mip_levels = 3;
    assert(gpu::internal::texture_desc_valid(&desc),
        "BC mip chains extend down to the 1x1 tail");

    desc.usage = { .sampled, .storage };
    assert(!gpu::internal::texture_desc_valid(&desc),
        "BC storage images are not exposed");
    desc.usage = { .color_attach };
    desc.mip_levels = 1;
    assert(!gpu::internal::texture_desc_valid(&desc),
        "BC color attachments are not exposed");
    desc.usage = { .sampled, .transfer_dst };
    desc.depth = 2;
    desc.width = 8;
    desc.height = 8;
    assert(!gpu::internal::texture_desc_valid(&desc),
        "BC textures are two-dimensional");
}

fn void texture_mip_helpers_count_whole_blocks() @test {
    assert(gpu::texture_mip_dimension(30, 0) == 30
            && gpu::texture_mip_dimension(30, 3) == 3
            && gpu::texture_mip_dimension(30, 9) == 1
            && gpu::texture_mip_dimension(1, 40) == 1,
        "mip axes halve per level and clamp at one");

    gpu::TextureDesc rgba = valid_texture_desc();
    rgba.mip_levels = 5;
    assert(gpu::texture_mip_bytes(&rgba, 0)!! == 16 * 8 * 4
            && gpu::texture_mip_bytes(&rgba, 4)!! == 4,
        "uncompressed mips are texel counts times texel bytes");

    gpu::TextureDesc bc7 = valid_texture_desc();
    bc7.width = 30;
    bc7.height = 18;
    bc7.mip_levels = 5;
    bc7.format = gpu::Format.BC7_SRGB;
    assert(gpu::texture_mip_bytes(&bc7, 0)!! == 8 * 5 * 16
            && gpu::texture_mip_bytes(&bc7, 2)!! == 2 * 1 * 16
            && gpu::texture_mip_bytes(&bc7, 4)!! == 16,
        "block-compressed mips round every edge up to whole blocks");

    gpu::TextureDesc volume = valid_texture_desc();
    volume.width = 4;
    volume.height = 4;
    volume.depth = 4;
    volume.mip_levels = 3;
    assert(gpu::texture_mip_bytes(&volume, 1)!! == 2 * 2 * 2 * 4,
        "3D mips include every depth slice");

    assert(@catch(gpu::texture_mip_bytes(&bc7, 5)) == gpu::INVALID_ARGUMENT,
        "a mip past the chain faults");
    bc7.format = gpu::Format.UNDEFINED;
    assert(@catch(gpu::texture_mip_bytes(&bc7, 0)) == gpu::INVALID_ARGUMENT,
        "an undefined format has no footprint");
    assert(@catch(gpu::texture_mip_bytes(null, 0)) == gpu::INVALID_ARGUMENT,
        "a null description faults");
}
