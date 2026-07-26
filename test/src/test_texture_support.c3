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
