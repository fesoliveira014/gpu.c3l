module gpu;

fn TextureDesc valid_texture_desc() {
    return {
        .dimension = TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 8,
        .format    = Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
}

fn void texture_desc_profile_accepts_normalized_defaults() @test {
    TextureDesc desc = valid_texture_desc();
    assert(texture_desc_valid(&desc), "zero mip and layer counts normalize to one");
    assert(texture_mip_levels(&desc) == 1, "zero mip count normalizes to one");
    assert(texture_array_layers(&desc) == 1, "zero layer count normalizes to one");
}

fn void texture_desc_profile_rejects_invalid_shape() @test {
    TextureDesc desc = valid_texture_desc();

    desc.width = 0;
    assert(!texture_desc_valid(&desc), "zero width is invalid");
    desc = valid_texture_desc();
    desc.height = 0;
    assert(!texture_desc_valid(&desc), "zero height is invalid");
    desc = valid_texture_desc();
    desc.depth = 2;
    assert(!texture_desc_valid(&desc), "2D depth above one is invalid");
    desc = valid_texture_desc();
    desc.dimension = TextureDimension.TEX_3D;
    assert(!texture_desc_valid(&desc), "unimplemented dimensions are invalid");
    desc = valid_texture_desc();
    desc.mip_levels = 6;
    assert(!texture_desc_valid(&desc), "mip count beyond the full chain is invalid");
}

fn void texture_desc_profile_rejects_invalid_usage_roles() @test {
    TextureDesc desc = valid_texture_desc();

    desc.usage = {};
    assert(!texture_desc_valid(&desc), "empty usage is invalid");
    desc.usage = { .shared_queues };
    assert(!texture_desc_valid(&desc), "shared queues alone is invalid");
    desc = valid_texture_desc();
    desc.format = Format.D32_FLOAT;
    desc.usage = { .storage };
    assert(!texture_desc_valid(&desc), "depth storage is invalid");
    desc.usage = { .color_attach };
    assert(!texture_desc_valid(&desc), "depth color attachment is invalid");
    desc = valid_texture_desc();
    desc.usage = { .depth_attach };
    assert(!texture_desc_valid(&desc), "color depth attachment is invalid");
}

fn void texture_desc_profile_rejects_unsupported_formats() @test {
    TextureDesc desc = valid_texture_desc();
    desc.format = Format.UNDEFINED;
    assert(!texture_desc_valid(&desc), "undefined format is invalid");
    desc.format = Format.D24_UNORM_S8_UINT;
    desc.usage = { .depth_attach };
    assert(!texture_desc_valid(&desc), "D24S8 is outside the end-to-end backend profile");
    desc.usage = { .color_attach };
    assert(!texture_desc_valid(&desc), "D24S8 color usage remains invalid");
    desc.usage = { .transfer_dst };
    assert(!texture_desc_valid(&desc), "D24S8 transfer usage remains invalid");
}
