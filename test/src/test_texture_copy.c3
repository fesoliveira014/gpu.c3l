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

fn void texture_copy_rejects_format_mismatch() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 4, 1, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.BGRA8_UNORM, 4, 1, { .transfer_dst });
    gpu::TextureCopyDesc desc = {};
    fault mismatch = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &desc, false));
    assert(mismatch == gpu::INVALID_ARGUMENT, "unequal formats fault INVALID_ARGUMENT");
}

fn void texture_copy_rejects_sample_count_mismatch() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 4, 1, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.RGBA8_UNORM, 4, 1, { .transfer_dst });
    dst.sample_count = gpu::SampleCount.FOUR;
    gpu::TextureCopyDesc desc = {};
    fault mismatch = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &desc, false));
    assert(mismatch == gpu::INVALID_ARGUMENT, "unequal sample counts fault INVALID_ARGUMENT");
}

fn void texture_copy_rejects_missing_transfer_usage() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 4, 1, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.RGBA8_UNORM, 4, 1, { .transfer_dst });
    gpu::TextureCopyDesc desc = {};

    src.usage = { .sampled };
    fault no_source = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &desc, false));
    assert(no_source == gpu::INVALID_ARGUMENT, "a source without transfer_src faults");

    src.usage = { .transfer_src };
    dst.usage = { .sampled };
    fault no_destination = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &desc, false));
    assert(no_destination == gpu::INVALID_ARGUMENT, "a destination without transfer_dst faults");
}

fn void texture_copy_rejects_out_of_bounds_regions() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 8, 4, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.RGBA8_UNORM, 4, 3, { .transfer_dst });

    gpu::TextureCopyDesc source_beyond = { .src_mip = 1, .src_x = 2, .width = 3, .height = 2 };
    fault on_source = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &source_beyond, false));
    assert(on_source == gpu::INVALID_ARGUMENT, "a region past the source mip edge faults");

    gpu::TextureCopyDesc destination_beyond = { .dst_mip = 1, .width = 3, .height = 2 };
    fault on_destination = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &destination_beyond, false));
    assert(on_destination == gpu::INVALID_ARGUMENT, "a region past the destination mip edge faults");

    gpu::TextureCopyDesc missing_mip = { .src_mip = 4 };
    fault no_mip = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &missing_mip, false));
    assert(no_mip == gpu::INVALID_ARGUMENT, "a mip level past the chain faults");

    gpu::TextureCopyDesc layer_beyond = { .dst_base_layer = 1, .width = 1, .height = 1 };
    fault on_layer = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &layer_beyond, false));
    assert(on_layer == gpu::INVALID_ARGUMENT, "a layer past the destination array faults");
}

fn void texture_copy_defaults_cover_the_source_mip() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.RGBA8_UNORM, 4, 3, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.RGBA8_UNORM, 2, 2, { .transfer_dst });
    gpu::TextureCopyDesc desc = { .src_mip = 1 };
    gpu::internal::TextureCopyExtent extent = gpu::internal::resolve_texture_copy(&src, &dst, &desc, false)!!;
    assert(extent.width == 2 && extent.height == 2 && extent.depth == 1 && extent.layers == 1,
        "zero extents select the remaining source mip");
}

fn void texture_copy_rejects_unaligned_block_offset() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.BC1_RGBA_UNORM, 16, 5, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.BC1_RGBA_UNORM, 16, 5, { .transfer_dst });

    gpu::TextureCopyDesc source_offset = { .src_x = 2, .width = 4, .height = 4 };
    fault on_source = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &source_offset, false));
    assert(on_source == gpu::INVALID_ARGUMENT, "a source offset inside a block faults");

    gpu::TextureCopyDesc destination_offset = { .dst_y = 1, .width = 4, .height = 4 };
    fault on_destination = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &destination_offset, false));
    assert(on_destination == gpu::INVALID_ARGUMENT, "a destination offset inside a block faults");

    gpu::TextureCopyDesc partial = { .width = 6, .height = 4 };
    fault on_extent = @catch(gpu::internal::resolve_texture_copy(&src, &dst, &partial, false));
    assert(on_extent == gpu::INVALID_ARGUMENT, "a partial block short of the mip edge faults");
}

fn void texture_copy_accepts_block_region_at_sub_block_mip_edge() @test {
    gpu::internal::TextureCopyShape src = copy_shape(gpu::Format.BC1_RGBA_UNORM, 64, 7, { .transfer_src });
    gpu::internal::TextureCopyShape dst = copy_shape(gpu::Format.BC1_RGBA_UNORM, 8, 4, { .transfer_dst });

    for (uint level = 0; level < 4; level++) {
        gpu::TextureCopyDesc desc = { .src_mip = level + 3, .dst_mip = level };
        gpu::internal::TextureCopyExtent extent =
            gpu::internal::resolve_texture_copy(&src, &dst, &desc, false)!!;
        assert(extent.width == (8u >> level) && extent.height == (8u >> level),
            "whole mips down to one texel copy between block-compressed chains");
    }
}

fn void texture_copy_rejects_overlap_within_one_texture() @test {
    gpu::internal::TextureCopyShape shape = copy_shape(
        gpu::Format.RGBA8_UNORM,
        8,
        3,
        { .transfer_src, .transfer_dst },
    );
    shape.array_layers = 2;

    gpu::TextureCopyDesc overlapping = { .src_x = 0, .dst_x = 2, .width = 4, .height = 4 };
    fault overlap = @catch(gpu::internal::resolve_texture_copy(&shape, &shape, &overlapping, true));
    assert(overlap == gpu::INVALID_ARGUMENT, "overlapping regions of one subresource fault");

    gpu::TextureCopyDesc other_mip = { .src_mip = 0, .dst_mip = 1, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &other_mip, true)!!;

    gpu::TextureCopyDesc other_layer = { .dst_base_layer = 1, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &other_layer, true)!!;

    gpu::TextureCopyDesc disjoint = { .dst_x = 4, .width = 4, .height = 4 };
    gpu::internal::resolve_texture_copy(&shape, &shape, &disjoint, true)!!;
}
