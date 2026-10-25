module gpu_test;

import gpu::internal @public;
import gpu @public;

fn gpu::internal::TextureCopyShape copy_shape(
    gpu::Format format,
    uint size,
    uint mip_levels,
    gpu::TextureUsage usage,
) {
    return {
        .format       = format,
        .usage        = usage,
        .sample_count = gpu::SampleCount.ONE,
        .width        = size,
        .height       = size,
        .mip_levels   = mip_levels,
        .array_layers = 1,
    };
}

fn void texture_copy_defaults_cover_the_source_mip() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 4, 3, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.RGBA8_UNORM, 2, 2, { .transfer_dst });
    gpu::TextureCopyDesc desc = { .src_mip = 1 };
    gpu::internal::TextureCopyExtent extent = gpu::internal::resolve_texture_copy(&src, &dst, &desc, false);
    assert(extent.width == 2 && extent.height == 2 && extent.depth == 1 && extent.layers == 1,
        "zero extents select the remaining source mip");
}

fn void texture_copy_accepts_block_region_at_sub_block_mip_edge() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.BC1_RGBA_UNORM, 64, 7, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.BC1_RGBA_UNORM, 8, 4, { .transfer_dst });

    for (uint level = 0; level < 4; level++) {
        gpu::TextureCopyDesc desc = { .src_mip = level + 3, .dst_mip = level };
        gpu::internal::TextureCopyExtent extent =
            gpu::internal::resolve_texture_copy(&src, &dst, &desc, false);
        assert(extent.width == (8u >> level) && extent.height == (8u >> level),
            "whole mips down to one texel copy between block-compressed chains");
    }
}

fn void texture_copy_accepts_disjoint_regions_within_one_texture() @test {
    gpu::internal::TextureCopyShape shape = copy_shape(
        gpu::Format.RGBA8_UNORM,
        8,
        3,
        { .transfer_src, .transfer_dst },
    );
    shape.array_layers = 2;

    gpu::TextureCopyDesc other_mip = { .src_mip = 0, .dst_mip = 1, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &other_mip, true);

    gpu::TextureCopyDesc other_layer = { .dst_base_layer = 1, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &other_layer, true);

    gpu::TextureCopyDesc disjoint = { .dst_x = 4, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &disjoint, true);
}
