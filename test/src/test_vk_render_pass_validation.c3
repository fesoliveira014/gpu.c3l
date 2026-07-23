module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

struct DynamicStateFixture {
    gpu::Device               device;
    gpu::TextureHandle        target;
    gpu::AttachmentViewHandle view;
    gpu::CommandAllocator     allocator;
    gpu::CommandList          commands;
    vk::VkDeviceState*        state;
}

fn DynamicStateFixture make_dynamic_state_fixture(
    gpu::ContractValidation contract,
    gpu::DebugMessageCallback callback = null,
    void* user_data = null,
) {
    gpu::RuntimeDesc runtime_desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = contract,
        .track_resource_lifetimes = true,
        .application_name         = "gpu_dynamic_state_contract_test",
        .debug_callback           = callback,
        .debug_user_data          = user_data,
    };
    DynamicStateFixture fixture;
    fixture.device = create_test_device(&runtime_desc)!!;
    fixture.target = make_render_validation_target(
        &fixture.device,
        gpu::Format.RGBA8_UNORM,
    );
    fixture.view = create_test_attachment_view(
        &fixture.device,
        fixture.target,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &fixture.device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    fixture.allocator = gpu::create_command_allocator(
        &fixture.device,
        queue,
    )!!;
    fixture.commands = gpu::begin_commands(&fixture.allocator)!!;
    record_val_target_to_attach(&fixture.commands, fixture.target);
    gpu::ColorTargetDesc[1] colors = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::cmd_begin_render_pass(&fixture.commands, &pass)!!;
    fixture.state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(fixture.device);
    vk::reset_dynamic_state_observation(fixture.state);
    return fixture;
}

fn void destroy_dynamic_state_fixture(
    DynamicStateFixture* fixture,
) {
    gpu::cmd_end_render_pass(&fixture.commands)!!;
    discard_test_commands(&fixture.commands)!!;
    gpu::destroy_command_allocator(&fixture.allocator)!!;
    destroy_test_attachment_view(&fixture.device, &fixture.view)!!;
    gpu::destroy_texture(&fixture.device, fixture.target)!!;
    destroy_test_device(&fixture.device)!!;
}

fn bool native_viewport_matches(
    vk::Viewport* native,
    gpu::Viewport* expected,
) {
    return native.x == expected.x
        && native.y == expected.y
        && native.width == expected.width
        && native.height == expected.height
        && native.min_depth == expected.min_depth
        && native.max_depth == expected.max_depth;
}

fn bool native_scissor_matches(
    vk::Rect2D* native,
    gpu::ScissorRect* expected,
) {
    return native.offset.x == expected.x
        && native.offset.y == expected.y
        && native.extent.width == (uint)expected.width
        && native.extent.height == (uint)expected.height;
}

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

fn void viewport_device_bounds_accept_native_shapes_exactly() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    fixture.state.viewport_limits = {
        .max_width  = 100,
        .max_height = 80,
        .min_bound  = -50.0f,
        .max_bound  = 100.0f,
    };

    gpu::Viewport inside = {
        .width     = (float)VAL_RT,
        .height    = (float)VAL_RT,
        .max_depth = 1.0f,
    };
    gpu::cmd_set_viewport(&fixture.commands, &inside)!!;
    gpu::Viewport device_edges = {
        .y         = 80.0f,
        .width     = 100.0f,
        .height    = -80.0f,
        .min_depth = 1.0f,
        .max_depth = 0.0f,
    };
    gpu::cmd_set_viewport(&fixture.commands, &device_edges)!!;
    gpu::Viewport negative_origin = {
        .x         = -50.0f,
        .y         = -50.0f,
        .width     = 10.0f,
        .height    = 10.0f,
        .max_depth = 1.0f,
    };
    gpu::cmd_set_viewport(&fixture.commands, &negative_origin)!!;

    vk::DynamicStateObservation observation =
        vk::dynamic_state_observation(fixture.state);
    assert(observation.viewport_emissions == 3,
        "inside, negative-height overscan, and negative-origin viewports emit");
    assert(native_viewport_matches(
        &observation.viewport,
        &negative_origin,
    ), "the final native viewport preserves every accepted field");
}

fn void viewport_rejections_preserve_previous_native_state() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    fixture.state.viewport_limits = {
        .max_width  = 100,
        .max_height = 80,
        .min_bound  = -50.0f,
        .max_bound  = 100.0f,
    };

    gpu::Viewport accepted = {
        .width     = 16.0f,
        .height    = 16.0f,
        .max_depth = 1.0f,
    };
    gpu::cmd_set_viewport(&fixture.commands, &accepted)!!;
    gpu::Viewport[17] rejected = {
        { .width = 0.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 0.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -0.0f, .max_depth = 1.0f },
        { .width = 101.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 81.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -81.0f, .max_depth = 1.0f },
        { .x = -51.0f, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .x = 100.0f, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = -51.0f, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = 100.0f, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .x = float::nan, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = float::inf, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = float::nan, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -float::inf, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .min_depth = float::nan, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .max_depth = float::inf },
        { .width = 1.0f, .height = 1.0f, .max_depth = 1.1f },
    };
    foreach (&viewport : rejected) {
        assert(@catch(gpu::cmd_set_viewport(
            &fixture.commands,
            viewport,
        )) == gpu::INVALID_ARGUMENT, "invalid viewport faults before emission");
    }

    vk::DynamicStateObservation observation =
        vk::dynamic_state_observation(fixture.state);
    assert(observation.viewport_emissions == 1,
        "rejected viewports add no native emission");
    assert(native_viewport_matches(&observation.viewport, &accepted),
        "rejected viewports preserve the previous native state");
}

fn void scissor_representation_accepts_overscan_and_empty_clip() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);

    gpu::ScissorRect overscan = {
        .x      = (int)VAL_RT,
        .y      = (int)VAL_RT,
        .width  = 32,
        .height = 32,
    };
    gpu::cmd_set_scissor(&fixture.commands, &overscan)!!;
    gpu::ScissorRect empty = {
        .x = int::max,
        .y = int::max,
    };
    gpu::cmd_set_scissor(&fixture.commands, &empty)!!;
    vk::DynamicStateObservation accepted =
        vk::dynamic_state_observation(fixture.state);
    assert(accepted.scissor_emissions == 2,
        "off-pass and zero-extent scissors emit");
    assert(native_scissor_matches(&accepted.scissor, &empty),
        "empty scissor preserves exact native offset and extent");

    gpu::ScissorRect[6] rejected = {
        { .x = -1, .width = 1, .height = 1 },
        { .y = -1, .width = 1, .height = 1 },
        { .width = -1, .height = 1 },
        { .width = 1, .height = -1 },
        { .x = int::max, .width = 1 },
        { .y = int::max, .height = 1 },
    };
    foreach (&scissor : rejected) {
        assert(@catch(gpu::cmd_set_scissor(
            &fixture.commands,
            scissor,
        )) == gpu::INVALID_ARGUMENT, "invalid scissor faults before emission");
    }
    vk::DynamicStateObservation after =
        vk::dynamic_state_observation(fixture.state);
    assert(after.scissor_emissions == accepted.scissor_emissions,
        "rejected scissors add no native emission");
    assert(native_scissor_matches(&after.scissor, &empty),
        "rejected scissors preserve the previous native state");
}

fn void dynamic_state_diagnostics_name_exact_rejected_fields() @test {
    SyncDiagnosticCapture capture = {};
    DynamicStateFixture fixture = make_dynamic_state_fixture(
        contract:  gpu::ContractValidation.FULL,
        callback:  &capture_sync_diagnostic,
        user_data: &capture,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    fixture.state.viewport_limits = {
        .max_width  = 100,
        .max_height = 80,
        .min_bound  = -50.0f,
        .max_bound  = 100.0f,
    };

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_set_viewport",
        debug_name:     null,
        rejected_field: "Viewport.width",
        invariant:      "width must not exceed the selected device maximum",
    );
    gpu::Viewport viewport = {
        .width     = 101.0f,
        .height    = 1.0f,
        .max_depth = 1.0f,
    };
    assert(@catch(gpu::cmd_set_viewport(
        &fixture.commands,
        &viewport,
    )) == gpu::INVALID_ARGUMENT, "over-limit viewport keeps INVALID_ARGUMENT");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_set_scissor",
        debug_name:     null,
        rejected_field: "ScissorRect.x + ScissorRect.width",
        invariant:      "endpoint must fit signed native coordinate range",
    );
    gpu::ScissorRect scissor = {
        .x     = int::max,
        .width = 1,
    };
    assert(@catch(gpu::cmd_set_scissor(
        &fixture.commands,
        &scissor,
    )) == gpu::INVALID_ARGUMENT, "overflowing scissor keeps INVALID_ARGUMENT");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );
}

fn void all_policies_lower_dynamic_state_identically() @test {
    DynamicStateFixture trusted = make_dynamic_state_fixture(
        gpu::ContractValidation.TRUSTED,
    );
    defer destroy_dynamic_state_fixture(&trusted);
    DynamicStateFixture boundaries = make_dynamic_state_fixture(
        gpu::ContractValidation.OBJECT_BOUNDARIES,
    );
    defer destroy_dynamic_state_fixture(&boundaries);
    DynamicStateFixture checked = make_dynamic_state_fixture(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&checked);

    gpu::Viewport viewport = {
        .x         = -1.0f,
        .y         = (float)VAL_RT,
        .width     = (float)VAL_RT + 2.0f,
        .height    = -(float)VAL_RT,
        .min_depth = 1.0f,
        .max_depth = 0.0f,
    };
    gpu::ScissorRect scissor = {
        .x      = (int)VAL_RT,
        .y      = (int)VAL_RT,
        .width  = 1,
        .height = 1,
    };
    gpu::cmd_set_viewport(&trusted.commands, &viewport)!!;
    gpu::cmd_set_scissor(&trusted.commands, &scissor)!!;
    gpu::cmd_set_viewport(&boundaries.commands, &viewport)!!;
    gpu::cmd_set_scissor(&boundaries.commands, &scissor)!!;
    gpu::cmd_set_viewport(&checked.commands, &viewport)!!;
    gpu::cmd_set_scissor(&checked.commands, &scissor)!!;

    vk::DynamicStateObservation trusted_observation =
        vk::dynamic_state_observation(trusted.state);
    vk::DynamicStateObservation boundaries_observation =
        vk::dynamic_state_observation(boundaries.state);
    vk::DynamicStateObservation checked_observation =
        vk::dynamic_state_observation(checked.state);
    assert(trusted_observation.viewport_emissions == 1
        && boundaries_observation.viewport_emissions == 1
        && checked_observation.viewport_emissions == 1
        && trusted_observation.scissor_emissions == 1
        && boundaries_observation.scissor_emissions == 1
        && checked_observation.scissor_emissions == 1,
        "each policy emits each accepted dynamic state exactly once");
    assert(native_viewport_matches(
        &trusted_observation.viewport,
        &viewport,
    ) && native_viewport_matches(
        &boundaries_observation.viewport,
        &viewport,
    ) && native_viewport_matches(
        &checked_observation.viewport,
        &viewport,
    ), "trusted, boundary, and checked viewport lowering is identical");
    assert(native_scissor_matches(
        &trusted_observation.scissor,
        &scissor,
    ) && native_scissor_matches(
        &boundaries_observation.scissor,
        &scissor,
    ) && native_scissor_matches(
        &checked_observation.scissor,
        &scissor,
    ), "trusted, boundary, and checked scissor lowering is identical");
}
