module gpu_test;

import gpu::internal @public;
import gpu @public;

fn void render_geometry_state_is_explicit_and_representable() @test {
    gpu::GraphicsState state = gpu::render_geometry_state(320, 240)!!;
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

    assert(@catch(gpu::render_geometry_state(0, 1))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::render_geometry_state(1, 0))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::render_geometry_state(
        (uint)int::max + 1,
        1,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::render_geometry_state(
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
        "multisample textures stay attachment-only in the device baseline");
}
