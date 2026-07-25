module gpu_test;

import gpu::internal @public;
import gpu @public;

fn gpu::AttachmentViewHandle render_test_view(uint index) {
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

fn void full_render_graphics_state_is_explicit_and_representable() @test {
    gpu::GraphicsState state = gpu::full_render_graphics_state(320, 240)!!;
    assert(state.viewport.x == 0.0f && state.viewport.y == 0.0f);
    assert(state.viewport.width == 320.0f
        && state.viewport.height == 240.0f);
    assert(state.viewport.min_depth == 0.0f
        && state.viewport.max_depth == 1.0f);
    assert(state.scissor.x == 0 && state.scissor.y == 0);
    assert(state.scissor.width == 320 && state.scissor.height == 240);
    assert(state.raster.topology == gpu::PrimitiveTopology.TRIANGLES);
    assert(state.raster.cull_mode == gpu::CullMode.NONE);
    assert(state.raster.front_face == gpu::FrontFace.COUNTER_CLOCKWISE);
    assert(!state.raster.depth_bias_enable);
    assert(state.raster.depth_bias_constant == 0.0f);
    assert(state.raster.depth_bias_slope == 0.0f);
    assert(state.raster.depth_bias_clamp == 0.0f);
    assert(!state.depth.test_enable && !state.depth.write_enable);
    assert(state.depth.compare == gpu::CompareOp.NEVER);
    assert(state.color.targets.len == 0);

    assert(@catch(gpu::full_render_graphics_state(0, 1))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::full_render_graphics_state(1, 0))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::full_render_graphics_state(
        (uint)int::max + 1,
        1,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::full_render_graphics_state(
        1,
        (uint)int::max + 1,
    )) == gpu::INVALID_ARGUMENT);
}

fn void common_color_states_are_exact_and_caller_owned() @test {
    gpu::ColorWriteMask alpha_only = { .alpha };
    gpu::ColorTargetState disabled =
        gpu::color_blend_disabled(alpha_only);
    assert(!disabled.blend.enable);
    assert(disabled.write_mask.alpha && (uint)disabled.write_mask == 8);

    gpu::ColorTargetState alpha = gpu::alpha_blend();
    assert(alpha.blend.enable);
    assert(alpha.blend.src_color == gpu::BlendFactor.SRC_ALPHA);
    assert(alpha.blend.dst_color == gpu::BlendFactor.ONE_MINUS_SRC_ALPHA);
    assert(alpha.blend.src_alpha == gpu::BlendFactor.ONE);
    assert(alpha.blend.dst_alpha == gpu::BlendFactor.ONE_MINUS_SRC_ALPHA);
    assert((uint)alpha.write_mask == (uint)gpu::COLOR_WRITE_ALL);

    gpu::ColorTargetState premultiplied =
        gpu::premultiplied_alpha_blend();
    assert(premultiplied.blend.src_color == gpu::BlendFactor.ONE);
    assert(
        premultiplied.blend.dst_color
            == gpu::BlendFactor.ONE_MINUS_SRC_ALPHA
    );

    gpu::ColorTargetState additive = gpu::additive_blend();
    assert(additive.blend.src_color == gpu::BlendFactor.ONE);
    assert(additive.blend.dst_color == gpu::BlendFactor.ONE);
    assert(additive.blend.src_alpha == gpu::BlendFactor.ONE);
    assert(additive.blend.dst_alpha == gpu::BlendFactor.ONE);

    gpu::ColorTargetState[3] targets;
    gpu::ColorState packet =
        gpu::uniform_color_state(targets[..], additive);
    assert(packet.targets.ptr == targets[..].ptr && packet.targets.len == 3);
    foreach (&target : targets) {
        assert(target.blend.src_color == gpu::BlendFactor.ONE);
        assert(target.blend.dst_color == gpu::BlendFactor.ONE);
        assert((uint)target.write_mask == (uint)gpu::COLOR_WRITE_ALL);
    }
}

fn void render_pass_descriptor_accepts_zero_value_operations() @test {
    gpu::ColorTargetDesc[1] colors = {{ .view = render_test_view(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);
    assert(gpu::internal::render_pass_desc_valid(&desc),
        "zero-value load/store operations form a valid color pass");
}

fn void render_pass_descriptor_rejects_invalid_structure() @test {
    gpu::ColorTargetDesc[1] colors = {{ .view = render_test_view(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);

    desc.width = 0;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "zero width is invalid");
    desc = valid_render_pass_desc(colors[..]);
    desc.colors = {};
    assert(!gpu::internal::render_pass_desc_valid(&desc), "a pass needs an attachment");
}

fn void render_pass_descriptor_rejects_invalid_operations() @test {
    gpu::ColorTargetDesc[1] colors = {{ .view = render_test_view(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);

    *((int*)&colors[0].load_op) = 99;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "invalid color load operation is rejected");
    colors[0].load_op = gpu::LoadOp.LOAD;
    *((int*)&colors[0].store_op) = 99;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "invalid color store operation is rejected");

    gpu::DepthTargetDesc depth = {
        .view = render_test_view(1),
    };
    *((int*)&depth.load_op) = 99;
    desc = {
        .depth  = &depth,
        .width  = 8,
        .height = 8,
    };
    assert(!gpu::internal::render_pass_desc_valid(&desc), "invalid depth load operation is rejected");
}

fn void render_pass_descriptor_validates_resolve_identity() @test {
    gpu::ColorTargetDesc[1] colors = {{ .view = render_test_view(0) }};
    gpu::RenderPassDesc desc = valid_render_pass_desc(colors[..]);
    desc.colors[0].resolve_view = render_test_view(1);
    assert(gpu::internal::render_pass_desc_valid(&desc),
        "a distinct resolve attachment is structurally valid");

    desc.colors[0].resolve_view = desc.colors[0].view;
    assert(!gpu::internal::render_pass_desc_valid(&desc),
        "an attachment cannot resolve into itself");
}

fn void render_pass_descriptor_validates_depth_clear_range() @test {
    gpu::DepthTargetDesc depth = {
        .view    = render_test_view(2),
        .load_op = gpu::LoadOp.CLEAR,
        .clear   = { .depth = 1.0f },
    };
    gpu::RenderPassDesc desc = {
        .depth  = &depth,
        .width  = 8,
        .height = 8,
    };
    assert(gpu::internal::render_pass_desc_valid(&desc), "unit depth clear is valid");

    depth.clear.depth = -0.1f;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "negative depth clear is invalid");
    depth.clear.depth = 1.1f;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "depth clear above one is invalid");
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
    assert(gpu::internal::texture_desc_valid(&texture),
        "multisample color attachments are structurally valid");

    texture.mip_levels = 2;
    assert(!gpu::internal::texture_desc_valid(&texture),
        "multisample textures cannot have mip chains");
    texture.mip_levels = 1;
    texture.usage = { .sampled };
    assert(!gpu::internal::texture_desc_valid(&texture),
        "multisample textures stay attachment-only in strict core");
}
