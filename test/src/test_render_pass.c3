module gpu_test;

import gpu @public;

fn gpu::TextureHandle render_test_texture(uint index) {
    return {
        .owner      = 1,
        .index      = index,
        .generation = 1,
    };
}

fn gpu::RenderPassDesc valid_render_pass_desc(gpu::ColorTargetDesc[] colors) {
    return {
        .colors = colors,
        .width  = 8,
        .height = 8,
    };
}

fn void render_pass_descriptor_accepts_zero_value_operations() @test {
    gpu::ColorTargetDesc[1] colors = {{ .texture = render_test_texture(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);
    assert(gpu::render_pass_desc_valid(&desc),
        "zero-value load/store operations form a valid color pass");
}

fn void render_pass_descriptor_rejects_invalid_structure() @test {
    gpu::ColorTargetDesc[1] colors = {{ .texture = render_test_texture(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);

    desc.width = 0;
    assert(!gpu::render_pass_desc_valid(&desc), "zero width is invalid");
    desc = valid_render_pass_desc(colors[..]);
    desc.colors = {};
    assert(!gpu::render_pass_desc_valid(&desc), "a pass needs an attachment");
}

fn void render_pass_descriptor_rejects_invalid_operations() @test {
    gpu::ColorTargetDesc[1] colors = {{ .texture = render_test_texture(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);

    *((int*)&colors[0].load_op) = 99;
    assert(!gpu::render_pass_desc_valid(&desc), "invalid color load operation is rejected");
    colors[0].load_op = gpu::LoadOp.LOAD;
    *((int*)&colors[0].store_op) = 99;
    assert(!gpu::render_pass_desc_valid(&desc), "invalid color store operation is rejected");

    gpu::DepthTargetDesc depth = {
        .texture = render_test_texture(1),
    };
    *((int*)&depth.load_op) = 99;
    desc = {
        .depth  = &depth,
        .width  = 8,
        .height = 8,
    };
    assert(!gpu::render_pass_desc_valid(&desc), "invalid depth load operation is rejected");
}

fn void render_pass_descriptor_validates_resolve_identity() @test {
    gpu::ColorTargetDesc[1] colors = {{ .texture = render_test_texture(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);
    desc.colors[0].resolve_texture = render_test_texture(1);
    assert(gpu::render_pass_desc_valid(&desc),
        "a distinct resolve attachment is structurally valid");

    desc.colors[0].resolve_texture = desc.colors[0].texture;
    assert(!gpu::render_pass_desc_valid(&desc),
        "an attachment cannot resolve into itself");
}

fn void render_pass_descriptor_validates_depth_clear_range() @test {
    gpu::DepthTargetDesc depth = {
        .texture = render_test_texture(2),
        .load_op = gpu::LoadOp.CLEAR,
        .clear   = { .depth = 1.0f },
    };
    gpu::RenderPassDesc desc = {
        .depth  = &depth,
        .width  = 8,
        .height = 8,
    };
    assert(gpu::render_pass_desc_valid(&desc), "unit depth clear is valid");

    depth.clear.depth = -0.1f;
    assert(!gpu::render_pass_desc_valid(&desc), "negative depth clear is invalid");
    depth.clear.depth = 1.1f;
    assert(!gpu::render_pass_desc_valid(&desc), "depth clear above one is invalid");
}

fn void render_sample_count_is_shared_by_textures_and_pipelines() @test {
    gpu::TextureDesc texture = valid_texture_desc();
    gpu::GraphicsPipelineDesc pipeline;
    assert(texture.sample_count == gpu::SampleCount.ONE,
        "zero-value textures are single-sampled");
    assert(pipeline.sample_count == gpu::SampleCount.ONE,
        "zero-value graphics pipelines are single-sampled");

    texture.usage = { .color_attach };
    texture.sample_count = gpu::SampleCount.FOUR;
    assert(gpu::texture_desc_valid(&texture),
        "multisample color attachments are structurally valid");

    texture.mip_levels = 2;
    assert(!gpu::texture_desc_valid(&texture),
        "multisample textures cannot have mip chains");
    texture.mip_levels = 1;
    texture.usage = { .sampled };
    assert(!gpu::texture_desc_valid(&texture),
        "multisample textures stay attachment-only in strict core");
}