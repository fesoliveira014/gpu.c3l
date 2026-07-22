module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

fn gpu::TextureHandle make_render_validation_target(
    gpu::Device* device,
    gpu::Format format,
) {
    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = VAL_RT,
        .height    = VAL_RT,
        .format    = format,
        .usage     = { .color_attach },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn vk::TextureSlot* render_validation_slot(
    gpu::Device* device,
    gpu::TextureHandle texture,
) {
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(*device);
    return state.textures.get(texture)!!;
}

fn vk::AttachmentViewSlot* render_validation_view_slot(
    gpu::Device* device,
    gpu::AttachmentViewHandle view,
) {
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(*device);
    return state.attachment_views.get(view)!!;
}

fn void texture_create_info_uses_requested_sample_count() @test {
    gpu::TextureDesc desc = {
        .access       = { .graphics },
        .width        = 16,
        .height       = 16,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .color_attach },
        .sample_count = gpu::SampleCount.FOUR,
    };
    uint[1] families;
    vk::ImageCreateInfo info = vk::make_texture_create_info(
        desc:           &desc,
        queue_families: families[:0],
    );
    assert(info.samples == vk::SampleCountFlagBits.COUNT_4_BIT,
        "texture creation uses the requested native sample count");
}

fn void color_resolve_mode_matches_numeric_format() @test {
    assert(vk::color_resolve_mode(gpu::Format.RGBA8_UNORM)
        == vk::ResolveModeFlagBits.AVERAGE_BIT,
        "normalized color resolves average samples");
    assert(vk::color_resolve_mode(gpu::Format.RGBA16_FLOAT)
        == vk::ResolveModeFlagBits.AVERAGE_BIT,
        "floating-point color resolves average samples");
    assert(vk::color_resolve_mode(gpu::Format.R8_UINT)
        == vk::ResolveModeFlagBits.SAMPLE_ZERO_BIT,
        "integer color resolves select sample zero");
}

fn void clear_color_uses_attachment_numeric_type() @test {
    gpu::ClearColor normalized = {
        .rgba = { 0.25f, 0.5f, 0.75f, 1.0f },
    };
    vk::ClearColorValue normalized_value = vk::clear_color_to_vk(
        gpu::Format.RGBA8_UNORM,
        normalized,
    );
    assert(normalized_value.float32 == normalized.rgba,
        "normalized attachment clears use floating-point values");

    gpu::ClearColor integer = {
        .uint_rgba = { 1, 2, 3, 4 },
    };
    vk::ClearColorValue integer_value = vk::clear_color_to_vk(
        gpu::Format.R32_UINT,
        integer,
    );
    assert(integer_value.uint32 == integer.uint_rgba,
        "unsigned-integer attachment clears use unsigned values");
}

fn void graphics_pipeline_rejects_unsupported_sample_count() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureFormatSupport support = gpu::get_texture_format_support(
        &device,
        gpu::Format.RGBA8_UNORM,
    )!!;
    gpu::SampleCount unsupported = gpu::SampleCount.ONE;
    if (!support.sample_counts.sixty_four) {
        unsupported = gpu::SampleCount.SIXTY_FOUR;
    } else if (!support.sample_counts.thirty_two) {
        unsupported = gpu::SampleCount.THIRTY_TWO;
    } else if (!support.sample_counts.sixteen) {
        unsupported = gpu::SampleCount.SIXTEEN;
    } else if (!support.sample_counts.eight) {
        unsupported = gpu::SampleCount.EIGHT;
    } else if (!support.sample_counts.four) {
        unsupported = gpu::SampleCount.FOUR;
    } else if (!support.sample_counts.two) {
        unsupported = gpu::SampleCount.TWO;
    }
    if (unsupported == gpu::SampleCount.ONE) return;

    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc desc = {
        .colors       = pipeline_colors[..],
        .sample_count = unsupported,
    };
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    assert(@catch(vk::validate_graphics_pipeline_desc(state, &desc))
        == gpu::UNSUPPORTED_FEATURE,
        "pipeline creation preflights attachment sample support");
}

fn void graphics_pipeline_key_distinguishes_sample_count() @test {
    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc one = {
        .vertex_shader   = { .digest = 1 },
        .fragment_shader = { .digest = 2 },
        .colors          = pipeline_colors[..],
    };
    gpu::GraphicsPipelineDesc four = one;
    four.sample_count = gpu::SampleCount.FOUR;

    vk::PipelineKey one_key = vk::build_graphics_key(
        &one,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    vk::PipelineKey four_key = vk::build_graphics_key(
        &four,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(one_key.sample_count == (int)gpu::SampleCount.ONE,
        "zero-initialized pipeline keys select one sample");
    assert(four_key.sample_count == (int)gpu::SampleCount.FOUR,
        "pipeline keys retain the requested sample count");
    assert(!vk::key_equals(&one_key, &four_key),
        "pipeline cache identity includes sample count");
}

fn void graphics_pipeline_key_zeroes_unobservable_and_unused_target_state() @test {
    gpu::ColorTargetState[1] first_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .blend      = {
            .src_color = gpu::BlendFactor.DST_ALPHA,
            .dst_color = gpu::BlendFactor.SRC_ALPHA,
            .color_op  = gpu::BlendOp.MAX,
        },
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc first = {
        .vertex_shader   = { .digest = 1 },
        .fragment_shader = { .digest = 2 },
        .colors          = first_colors[..],
    };
    gpu::ColorTargetState[1] second_colors = first_colors;
    second_colors[0].blend = {};
    gpu::GraphicsPipelineDesc second = first;
    second.colors = second_colors[..];

    vk::PipelineKey first_key = vk::build_graphics_key(
        &first,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    vk::PipelineKey second_key = vk::build_graphics_key(
        &second,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(vk::key_equals(&first_key, &second_key),
        "disabled blend bytes are absent from canonical identity");
    assert(vk::hash_key(&first_key) == vk::hash_key(&second_key),
        "canonical aliases have equal key hashes");
    assert(first_key.reserved == 0 && second_key.reserved == 0,
        "explicit key tail padding stays zero");

    for (uint i = 1; i < gpu::MAX_COLOR_ATTACHMENTS; i++) {
        vk::ColorTargetKey* unused = &first_key.color_targets[i];
        assert(unused.format == 0
                && unused.blend_enable == 0
                && (uint)unused.write_mask == 0,
            "unused target key fields are zero");
        assert(unused.blend.src_color == 0
                && unused.blend.dst_color == 0
                && unused.blend.color_op == 0
                && unused.blend.src_alpha == 0
                && unused.blend.dst_alpha == 0
                && unused.blend.alpha_op == 0,
            "unused target blend fields are zero");
    }
}

fn void render_pass_rejects_mixed_samples_before_mutation() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle first = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, first)!!;
    gpu::TextureHandle second = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, second)!!;
    render_validation_slot(&device, second).sample_count =
        gpu::SampleCount.FOUR;
    gpu::AttachmentViewHandle first_view = create_test_attachment_view(
        &device,
        first,
    )!!;
    defer destroy_test_attachment_view(&device, &first_view)!!;
    gpu::AttachmentViewHandle second_view = create_test_attachment_view(
        &device,
        second,
    )!!;
    defer destroy_test_attachment_view(&device, &second_view)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    record_val_target_to_attach(&commands, first);
    record_val_target_to_attach(&commands, second);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint references_before = record.scratch.reference_count;
    gpu::ColorTargetDesc[2] colors = {
        { .view = first_view },
        { .view = second_view },
    };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };

    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "all render attachments must use the same sample count");
    assert(record.state == vk::CommandState.RECORDING,
        "sample mismatch records no native render pass");
    assert(record.scratch.reference_count == references_before,
        "sample mismatch retains no attachment references");
}

fn void render_pass_rejects_invalid_resolve_before_mutation() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle source = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, source)!!;
    gpu::TextureHandle resolve = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, resolve)!!;
    gpu::AttachmentViewHandle source_view = create_test_attachment_view(
        &device,
        source,
    )!!;
    defer destroy_test_attachment_view(&device, &source_view)!!;
    gpu::AttachmentViewHandle resolve_view = create_test_attachment_view(
        &device,
        resolve,
    )!!;
    defer destroy_test_attachment_view(&device, &resolve_view)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    record_val_target_to_attach(&commands, source);
    record_val_target_to_attach(&commands, resolve);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint references_before = record.scratch.reference_count;
    gpu::ColorTargetDesc[1] colors = {{
        .view         = source_view,
        .resolve_view = resolve_view,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };

    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "a single-sample color attachment cannot have a resolve target");
    assert(record.state == vk::CommandState.RECORDING,
        "invalid resolve records no native render pass");
    assert(record.scratch.reference_count == references_before,
        "invalid resolve retains no attachment references");

    vk::AttachmentViewSlot* source_slot = render_validation_view_slot(
        &device,
        source_view,
    );
    vk::AttachmentViewSlot* resolve_slot = render_validation_view_slot(
        &device,
        resolve_view,
    );
    source_slot.sample_count = gpu::SampleCount.FOUR;
    resolve_slot.format = gpu::Format.BGRA8_UNORM;
    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "resolve source and destination formats must match");
    assert(record.scratch.reference_count == references_before,
        "resolve format mismatch retains no attachment references");

    resolve_slot.format = gpu::Format.RGBA8_UNORM;
    resolve_slot.sample_count = gpu::SampleCount.TWO;
    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "resolve destinations must be single-sampled");
    assert(record.scratch.reference_count == references_before,
        "resolve sample mismatch retains no attachment references");
}

fn void pipeline_render_compatibility_rejects_before_mutation() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_render_validation_target(
        &device,
        gpu::Format.BGRA8_UNORM,
    );
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList prebound = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&prebound)!!;
    record_val_target_to_attach(&prebound, target);
    gpu::cmd_bind_pipeline(&prebound, draw.graphics)!!;
    vk::CommandRecord* prebound_record = vk::resolve_command(&prebound)!!;
    uint prebound_references = prebound_record.scratch.reference_count;
    assert(@catch(gpu::cmd_begin_render_pass(&prebound, &pass))
        == gpu::INVALID_ARGUMENT,
        "begin rejects a prebound pipeline with incompatible color formats");
    assert(prebound_record.state == vk::CommandState.RECORDING,
        "prebound incompatibility records no native render pass");
    assert(prebound_record.scratch.reference_count == prebound_references,
        "prebound incompatibility retains no attachment references");

    gpu::CommandList in_pass = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&in_pass)!!;
    record_val_target_to_attach(&in_pass, target);
    gpu::cmd_begin_render_pass(&in_pass, &pass)!!;
    vk::CommandRecord* in_pass_record = vk::resolve_command(&in_pass)!!;
    uint in_pass_references = in_pass_record.scratch.reference_count;
    assert(@catch(gpu::cmd_bind_pipeline(&in_pass, draw.graphics))
        == gpu::INVALID_ARGUMENT,
        "binding rejects a pipeline incompatible with the active render pass");
    assert(in_pass_record.bound_pipeline.handle == gpu::PIPELINE_HANDLE_INVALID,
        "in-pass incompatibility leaves the active pipeline unchanged");
    assert(in_pass_record.scratch.reference_count == in_pass_references,
        "in-pass incompatibility retains no pipeline reference");
    gpu::cmd_end_render_pass(&in_pass)!!;
}

fn void end_render_pass_scopes_graphics_binding_to_pass() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle first = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, first)!!;
    gpu::TextureHandle second = make_render_validation_target(
        &device,
        gpu::Format.BGRA8_UNORM,
    );
    defer gpu::destroy_texture(&device, second)!!;
    gpu::AttachmentViewHandle first_view = create_test_attachment_view(
        &device,
        first,
    )!!;
    defer destroy_test_attachment_view(&device, &first_view)!!;
    gpu::AttachmentViewHandle second_view = create_test_attachment_view(
        &device,
        second,
    )!!;
    defer destroy_test_attachment_view(&device, &second_view)!!;
    gpu::ColorTargetDesc[1] first_colors = {{ .view = first_view }};
    gpu::RenderPassDesc first_pass = {
        .colors = first_colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::ColorTargetDesc[1] second_colors = {{ .view = second_view }};
    gpu::RenderPassDesc second_pass = {
        .colors = second_colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    record_val_target_to_attach(&commands, first);
    record_val_target_to_attach(&commands, second);
    gpu::cmd_begin_render_pass(&commands, &first_pass)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.bound_pipeline.handle == gpu::PIPELINE_HANDLE_INVALID,
        "ending a render pass releases the graphics pipeline binding");
    gpu::cmd_begin_render_pass(&commands, &second_pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
}

fn void pipeline_sample_mismatch_rejects_before_mutation() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, target)!!;
    render_validation_slot(&device, target).sample_count =
        gpu::SampleCount.FOUR;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    record_val_target_to_attach(&commands, target);
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint references_before = record.scratch.reference_count;

    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "pipeline and render attachments must use the same sample count");
    assert(record.state == vk::CommandState.RECORDING,
        "pipeline sample mismatch records no native render pass");
    assert(record.scratch.reference_count == references_before,
        "pipeline sample mismatch retains no attachment references");
}
