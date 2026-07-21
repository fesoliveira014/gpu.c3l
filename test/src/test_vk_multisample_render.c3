module gpu_test;

import gpu @public;

fn gpu::SampleCount common_render_sample_count(
    gpu::TextureSampleCountSupport color,
    gpu::TextureSampleCountSupport depth,
) {
    if (color.four && depth.four) return gpu::SampleCount.FOUR;
    if (color.two && depth.two) return gpu::SampleCount.TWO;
    if (color.eight && depth.eight) return gpu::SampleCount.EIGHT;
    if (color.sixteen && depth.sixteen) {
        return gpu::SampleCount.SIXTEEN;
    }
    if (color.thirty_two && depth.thirty_two) {
        return gpu::SampleCount.THIRTY_TWO;
    }
    if (color.sixty_four && depth.sixty_four) {
        return gpu::SampleCount.SIXTY_FOUR;
    }
    return gpu::SampleCount.ONE;
}

fn gpu::TextureHandle make_multisample_target(
    gpu::Device* device,
    gpu::Format format,
    gpu::TextureUsage usage,
    gpu::SampleCount sample_count,
    ZString debug_name,
) {
    gpu::TextureDesc desc = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .access       = { .graphics },
        .width        = DP_RT,
        .height       = DP_RT,
        .format       = format,
        .usage        = usage,
        .sample_count = sample_count,
        .debug_name   = debug_name,
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void multisample_color_depth_resolves_to_readback() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureFormatSupport color_support =
        gpu::get_texture_format_support(
            &device,
            gpu::Format.RGBA8_UNORM,
        )!!;
    gpu::TextureFormatSupport depth_support =
        gpu::get_texture_format_support(
            &device,
            gpu::Format.D32_FLOAT,
        )!!;
    gpu::SampleCount sample_count = common_render_sample_count(
        color_support.sample_counts,
        depth_support.sample_counts,
    );
    if (sample_count == gpu::SampleCount.ONE) return;

    gpu::TextureHandle color = make_multisample_target(
        device:       &device,
        format:       gpu::Format.RGBA8_UNORM,
        usage:        { .color_attach },
        sample_count: sample_count,
        debug_name:   "multisample_color",
    );
    defer gpu::destroy_texture(&device, color)!!;
    gpu::AttachmentViewHandle color_view = create_test_attachment_view(
        &device,
        color,
    )!!;
    defer destroy_test_attachment_view(&device, &color_view)!!;
    gpu::TextureHandle depth = make_multisample_target(
        device:       &device,
        format:       gpu::Format.D32_FLOAT,
        usage:        { .depth_attach },
        sample_count: sample_count,
        debug_name:   "multisample_depth",
    );
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;
    gpu::TextureHandle resolve = make_multisample_target(
        device:       &device,
        format:       gpu::Format.RGBA8_UNORM,
        usage:        { .color_attach, .transfer_src },
        sample_count: gpu::SampleCount.ONE,
        debug_name:   "multisample_resolve",
    );
    defer gpu::destroy_texture(&device, resolve)!!;
    gpu::AttachmentViewHandle resolve_view = create_test_attachment_view(
        &device,
        resolve,
    )!!;
    defer destroy_test_attachment_view(&device, &resolve_view)!!;

    DepthCtx shaders = make_depth_ctx();
    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .colors          = pipeline_colors[..],
        .depth_format    = gpu::Format.D32_FLOAT,
        .sample_count    = sample_count,
        .debug_name      = "multisample_color_depth",
    };
    gpu::PipelineHandle pipeline =
        gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    OwnedSpan vertices_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &vertices_owned)!!;
    OwnedSpan color_owned = write_color(
        &device,
        { 0.25f, 0.5f, 0.75f, 1.0f },
    );
    defer free_owned_span(&device, &color_owned)!!;
    OwnedSpan readback = make_readback(&device, DP_COLOR_BYTES);
    defer free_owned_span(&device, &readback)!!;

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    record_color_to_attach(&commands, color);
    record_color_to_attach(&commands, resolve);
    record_depth_to_attach(&commands, depth);

    gpu::ColorTargetDesc[1] colors = {{
        .view         = color_view,
        .resolve_view = resolve_view,
        .load_op      = gpu::LoadOp.CLEAR,
        .store_op     = gpu::StoreOp.DONT_CARE,
        .clear        = {
            .rgba = {
                8.0f / 255.0f,
                16.0f / 255.0f,
                32.0f / 255.0f,
                1.0f,
            },
        },
    }};
    gpu::DepthTargetDesc depth_target = {
        .view     = depth_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.DONT_CARE,
        .clear    = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .depth  = &depth_target,
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::DepthState depth_state = {
        .test_enable  = true,
        .write_enable = true,
        .compare      = gpu::CompareOp.LESS,
    };
    bind_test_graphics_pipeline(
        &commands,
        pipeline,
        depth_state,
    );
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(
            &device,
            vertices_owned.span,
        ),
        fragment_root:  test_span_address(
            &device,
            color_owned.span,
        ),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::GpuSpan readback_span = readback.span;
    record_color_readback(
        &commands,
        resolve,
        &readback_span,
    );
    submit_and_finish(&device, &commands);
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    char* pixels = (char*)test_span_mapping(
        &device,
        readback.span,
    ).ptr;
    char* center =
        &pixels[(DP_RT / 2 * DP_RT + DP_RT / 2) * 4];
    uint center_r = (uint)center[0];
    uint center_g = (uint)center[1];
    uint center_b = (uint)center[2];
    assert(center_r >= 63 && center_r <= 65
        && center_g >= 127 && center_g <= 129
        && center_b >= 190 && center_b <= 192
        && center[3] == (char)255,
        "multisample color/depth draw resolves its center pixel");
    assert(pixels[0] == (char)8
        && pixels[1] == (char)16
        && pixels[2] == (char)32
        && pixels[3] == (char)255,
        "multisample resolve preserves the clear color outside the draw");
}
