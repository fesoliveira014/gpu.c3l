module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::core::mem;

const char[*] REVIEW_ALT_COMP_SPIRV =
    $embed("shaders/heap_write.comp.spv");

struct ReviewValidationCapture {
    uint errors;
}

struct DynamicColorCapture {
    uint blend_enable_calls;
    uint blend_equation_calls;
    uint write_mask_calls;
    uint first_attachment;
    uint target_count;
    uint[gpu::MAX_COLOR_ATTACHMENTS] blend_enables;
    vk::ColorBlendEquationEXT[gpu::MAX_COLOR_ATTACHMENTS] blend_equations;
    vk::ColorComponentFlags[gpu::MAX_COLOR_ATTACHMENTS] write_masks;
}

DynamicColorCapture* dynamic_color_capture;

fn void capture_color_blend_enable(
    vk::CommandBuffer command_buffer,
    uint first_attachment,
    uint target_count,
    uint* blend_enables,
) {
    (void)command_buffer;
    DynamicColorCapture* capture = dynamic_color_capture;
    capture.blend_enable_calls++;
    capture.first_attachment = first_attachment;
    capture.target_count = target_count;
    for (uint i = 0; i < target_count; i++) {
        capture.blend_enables[i] = blend_enables[i];
    }
}

fn void capture_color_blend_equation(
    vk::CommandBuffer command_buffer,
    uint first_attachment,
    uint target_count,
    vk::ColorBlendEquationEXT* blend_equations,
) {
    (void)command_buffer;
    DynamicColorCapture* capture = dynamic_color_capture;
    capture.blend_equation_calls++;
    assert(capture.first_attachment == first_attachment
        && capture.target_count == target_count);
    for (uint i = 0; i < target_count; i++) {
        capture.blend_equations[i] = blend_equations[i];
    }
}

fn void capture_color_write_mask(
    vk::CommandBuffer command_buffer,
    uint first_attachment,
    uint target_count,
    vk::ColorComponentFlags* write_masks,
) {
    (void)command_buffer;
    DynamicColorCapture* capture = dynamic_color_capture;
    capture.write_mask_calls++;
    assert(capture.first_attachment == first_attachment
        && capture.target_count == target_count);
    for (uint i = 0; i < target_count; i++) {
        capture.write_masks[i] = write_masks[i];
    }
}

fn void capture_review_validation_error(
    gpu::DebugMessage* message,
    void* user_data,
) {
    if (message.category.backend
        && message.category.validation
        && message.severity == gpu::DebugMessageSeverity.ERROR) {
        ReviewValidationCapture* capture =
            (ReviewValidationCapture*)user_data;
        capture.errors++;
    }
}

fn gpu::Device make_review_validation_device(
    ReviewValidationCapture* capture,
) {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_pipeline_binding_review_test",
        .debug_callback           = &capture_review_validation_error,
        .debug_user_data           = capture,
    };
    return create_test_device(&desc)!!;
}

struct DynamicStateFixture {
    gpu::Device               device;
    gpu::TextureHandle        target;
    gpu::AttachmentViewHandle view;
    gpu::CommandAllocator     allocator;
    gpu::CommandList          commands;
    vk::VkDeviceState*        state;
}

fn gpu::PipelineHandle review_distinct_graphics_pipeline(
    gpu::Device* device,
    gpu::Format format,
) {
    gpu::ShaderDesc vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = VAL_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = VAL_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::Format[1] color_formats = { format };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = gpu::prepare_shader_code(&vertex_desc)!!,
        .fragment_shader = gpu::prepare_shader_code(&fragment_desc)!!,
        .color_formats   = color_formats[..],
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::PipelineHandle review_distinct_compute_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = REVIEW_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc desc = {
        .shader = gpu::prepare_shader_code(&shader_desc)!!,
    };
    return gpu::create_compute_pipeline(device, &desc)!!;
}

fn DynamicStateFixture make_dynamic_state_fixture_before_pass(
    gpu::ContractValidation contract,
    gpu::DebugMessageCallback callback = null,
    void* user_data = null,
) {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = contract,
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
    fixture.state = gpu::internal::device_slots[fixture.device.index].data.state;
    vk::reset_dynamic_state_observation(fixture.state);
    return fixture;
}

fn DynamicStateFixture make_dynamic_state_fixture(
    gpu::ContractValidation contract,
    gpu::DebugMessageCallback callback = null,
    void* user_data = null,
) {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        contract,
        callback,
        user_data,
    );
    gpu::ColorTargetDesc[1] colors = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(
        vk::resolve_command(&fixture.commands)!!,
        color_formats[..],
    );
    gpu::cmd_begin_render_pass_with_state(
        &fixture.commands,
        &pass,
        &graphics_state,
    )!!;
    vk::reset_dynamic_state_observation(fixture.state);
    return fixture;
}

fn void destroy_dynamic_state_fixture(
    DynamicStateFixture* fixture,
) {
    vk::CommandRecord* record = vk::resolve_command(&fixture.commands)!!;
    if (record.state == gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        gpu::cmd_end_render_pass(&fixture.commands)!!;
    }
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

fn bool raster_state_matches(
    gpu::DynamicRasterState* actual,
    gpu::DynamicRasterState* expected,
) {
    return actual.topology == expected.topology
        && actual.cull_mode == expected.cull_mode
        && actual.front_face == expected.front_face
        && actual.depth_bias_enable == expected.depth_bias_enable
        && actual.depth_bias_constant == expected.depth_bias_constant
        && actual.depth_bias_slope == expected.depth_bias_slope
        && actual.depth_bias_clamp == expected.depth_bias_clamp;
}

fn bool depth_state_matches(
    gpu::DepthState* actual,
    gpu::DepthState* expected,
) {
    return actual.test_enable == expected.test_enable
        && actual.write_enable == expected.write_enable
        && actual.compare == expected.compare;
}

fn bool observation_matches_graphics_state(
    vk::DynamicStateObservation* observation,
    gpu::GraphicsState* expected,
) {
    return native_viewport_matches(
        &observation.viewport,
        &expected.viewport,
    ) && native_scissor_matches(
        &observation.scissor,
        &expected.scissor,
    ) && raster_state_matches(
        &observation.raster,
        &expected.raster,
    ) && depth_state_matches(
        &observation.depth,
        &expected.depth,
    );
}

fn bool observation_has_no_emissions(
    vk::DynamicStateObservation* observation,
) {
    return observation.begin_rendering_emissions == 0
        && observation.viewport_emissions == 0
        && observation.scissor_emissions == 0
        && observation.raster_state_emissions == 0
        && observation.depth_state_emissions == 0;
}

fn vk::VkDynamicColorDispatch install_dynamic_color_capture(
    vk::VkDeviceState* state,
    DynamicColorCapture* capture,
) {
    vk::VkDynamicColorDispatch original =
        state.device_dispatch.dynamic_color;
    dynamic_color_capture = capture;
    state.device_dispatch.dynamic_color = {
        .set_blend_enable   = &capture_color_blend_enable,
        .set_blend_equation = &capture_color_blend_equation,
        .set_write_mask     = &capture_color_write_mask,
    };
    return original;
}

fn void select_graphics_color_pipeline(
    vk::CommandRecord* record,
    gpu::Format[] formats,
) {
    record.logical_pipeline = {
        .handle = {
            .owner      = 1,
            .index      = 1,
            .generation = 1,
        },
        .kind = vk::PipelineKind.GRAPHICS,
    };
    record.logical_pipeline.render.color_count = (uint)formats.len;
    foreach (i, format : formats) {
        record.logical_pipeline.render.color_formats[i] = format;
    }
}

fn bool dynamic_color_captures_match(
    DynamicColorCapture* first,
    DynamicColorCapture* second,
) {
    if (first.blend_enable_calls != second.blend_enable_calls
        || first.blend_equation_calls != second.blend_equation_calls
        || first.write_mask_calls != second.write_mask_calls
        || first.first_attachment != second.first_attachment
        || first.target_count != second.target_count) return false;
    for (uint i = 0; i < first.target_count; i++) {
        vk::ColorBlendEquationEXT* a = &first.blend_equations[i];
        vk::ColorBlendEquationEXT* b = &second.blend_equations[i];
        if (first.blend_enables[i] != second.blend_enables[i]
            || first.write_masks[i] != second.write_masks[i]
            || a.src_color_blend_factor != b.src_color_blend_factor
            || a.dst_color_blend_factor != b.dst_color_blend_factor
            || a.color_blend_op != b.color_blend_op
            || a.src_alpha_blend_factor != b.src_alpha_blend_factor
            || a.dst_alpha_blend_factor != b.dst_alpha_blend_factor
            || a.alpha_blend_op != b.alpha_blend_op) return false;
    }
    return true;
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
        gpu::internal::device_slots[device.index].data.state;
    return state.textures.get(texture)!!;
}

fn vk::AttachmentViewSlot* render_validation_view_slot(
    gpu::Device* device,
    gpu::AttachmentViewHandle view,
) {
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
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

    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc = {
        .color_formats = color_formats[..],
        .sample_count = unsupported,
    };
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    assert(@catch(vk::validate_graphics_pipeline_desc(state, &desc))
        == gpu::UNSUPPORTED_FEATURE,
        "pipeline creation preflights attachment sample support");
}

fn void graphics_pipeline_key_distinguishes_sample_count() @test {
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc one = {
        .vertex_shader   = { .digest = 1 },
        .fragment_shader = { .digest = 2 },
        .color_formats   = color_formats[..],
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
    assert(one_key.sample_count == gpu::SampleCount.ONE,
        "zero-initialized pipeline keys select one sample");
    assert(four_key.sample_count == gpu::SampleCount.FOUR,
        "pipeline keys retain the requested sample count");
    assert(!vk::key_equals(&one_key, &four_key),
        "pipeline cache identity includes sample count");
}

fn void graphics_pipeline_key_zeroes_unused_formats() @test {
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc first = {
        .vertex_shader   = { .digest = 1 },
        .fragment_shader = { .digest = 2 },
        .color_formats   = color_formats[..],
        .debug_name      = "first",
    };
    gpu::GraphicsPipelineDesc second = first;
    second.debug_name = "alias";

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
        "debug names are absent from canonical identity");
    assert(vk::hash_key(&first_key) == vk::hash_key(&second_key),
        "canonical aliases have equal key hashes");
    for (uint i = 1; i < gpu::MAX_COLOR_ATTACHMENTS; i++) {
        assert(first_key.color_formats[i] == gpu::Format.UNDEFINED,
            "unused target formats are zero");
    }
}

fn void graphics_pipeline_key_is_compact_and_format_only() @test {
    gpu::Format[2] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.BGRA8_UNORM,
    };
    gpu::GraphicsPipelineDesc first = {
        .color_formats = formats[..],
        .depth_format  = gpu::Format.D32_FLOAT,
        .debug_name = "first",
    };
    gpu::GraphicsPipelineDesc same_desc = first;
    same_desc.debug_name = "alias";

    vk::PipelineKey first_key = vk::build_graphics_key(
        &first,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    vk::PipelineKey alias_key = vk::build_graphics_key(
        &same_desc,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(vk::key_equals(&first_key, &alias_key));
    assert(vk::hash_key(&first_key) == vk::hash_key(&alias_key));
    assert(first_key.color_formats[0] == gpu::Format.RGBA8_UNORM);
    assert(vk::PipelineKey::size == 60,
        "the shared pipeline key remains compact");

    gpu::GraphicsPipelineDesc changed = first;
    changed.sample_count = gpu::SampleCount.FOUR;
    vk::PipelineKey sample_key = vk::build_graphics_key(
        &changed,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &sample_key));

    changed = first;
    changed.polygon_mode = gpu::PolygonMode.LINE;
    vk::PipelineKey polygon_key = vk::build_graphics_key(
        &changed,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &polygon_key));

    gpu::Format[1] alternate_formats = {
        gpu::Format.BGRA8_UNORM,
    };
    changed = first;
    changed.color_formats = alternate_formats[..];
    vk::PipelineKey format_key = vk::build_graphics_key(
        &changed,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &format_key));

    vk::PipelineKey vertex_key = vk::build_graphics_key(
        &first,
        (vk::ShaderId)3,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &vertex_key));

    vk::PipelineKey fragment_key = vk::build_graphics_key(
        &first,
        (vk::ShaderId)1,
        (vk::ShaderId)3,
    );
    assert(!vk::key_equals(&first_key, &fragment_key));

    changed = first;
    changed.depth_format = gpu::Format.UNDEFINED;
    vk::PipelineKey depth_key = vk::build_graphics_key(
        &changed,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &depth_key));

    gpu::Format[2] reordered_formats = {
        gpu::Format.BGRA8_UNORM,
        gpu::Format.RGBA8_UNORM,
    };
    changed = first;
    changed.color_formats = reordered_formats[..];
    vk::PipelineKey order_key = vk::build_graphics_key(
        &changed,
        (vk::ShaderId)1,
        (vk::ShaderId)2,
    );
    assert(!vk::key_equals(&first_key, &order_key));
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
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[2] color_targets = {
        gpu::color_blend_disabled(),
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::Format[2] color_formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(record, color_formats[..]);

    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    ))
        == gpu::INVALID_ARGUMENT,
        "all render attachments must use the same sample count");
    assert(record.state == gpu::internal::CommandState.RECORDING,
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
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(record, color_formats[..]);

    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    ))
        == gpu::INVALID_ARGUMENT,
        "a single-sample color attachment cannot have a resolve target");
    assert(record.state == gpu::internal::CommandState.RECORDING,
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
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    ))
        == gpu::INVALID_ARGUMENT,
        "resolve source and destination formats must match");
    assert(record.scratch.reference_count == references_before,
        "resolve format mismatch retains no attachment references");

    resolve_slot.format = gpu::Format.RGBA8_UNORM;
    resolve_slot.sample_count = gpu::SampleCount.TWO;
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    ))
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
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
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
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::PipelineHandle compatible = review_distinct_graphics_pipeline(
        &device,
        gpu::Format.BGRA8_UNORM,
    );
    defer gpu::destroy_pipeline(&device, compatible)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList prebound = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&prebound)!!;
    record_val_target_to_attach(&prebound, target);
    gpu::cmd_bind_pipeline(&prebound, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&prebound, draw.graphics)!!;
    vk::CommandRecord* prebound_record = vk::resolve_command(&prebound)!!;
    uint prebound_references = prebound_record.scratch.reference_count;
    vk::NativePipelineBinding native_graphics_before =
        prebound_record.native_graphics_pipeline;
    vk::NativePipelineBinding native_compute_before =
        prebound_record.native_compute_pipeline;
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &prebound,
        &pass,
        &graphics_state,
    ))
        == gpu::INVALID_ARGUMENT,
        "begin rejects a prebound pipeline with incompatible color formats");
    assert(prebound_record.state == gpu::internal::CommandState.RECORDING,
        "prebound incompatibility records no native render pass");
    assert(prebound_record.scratch.reference_count == prebound_references,
        "prebound incompatibility retains no attachment references");
    assert(prebound_record.logical_pipeline.handle == draw.graphics
        && prebound_record.native_graphics_pipeline.pipeline
            == native_graphics_before.pipeline
        && prebound_record.native_graphics_pipeline.cache_entry
            == native_graphics_before.cache_entry
        && prebound_record.native_compute_pipeline.pipeline
            == native_compute_before.pipeline
        && prebound_record.native_compute_pipeline.cache_entry
            == native_compute_before.cache_entry,
        "prebound incompatibility preserves both native pipeline caches");

    gpu::CommandList in_pass = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&in_pass)!!;
    record_val_target_to_attach(&in_pass, target);
    gpu::cmd_bind_pipeline(&in_pass, compatible)!!;
    gpu::cmd_begin_render_pass_with_state(&in_pass, &pass, &graphics_state)!!;
    vk::CommandRecord* in_pass_record = vk::resolve_command(&in_pass)!!;
    uint in_pass_references = in_pass_record.scratch.reference_count;
    assert(@catch(gpu::cmd_bind_pipeline(&in_pass, draw.graphics))
        == gpu::INVALID_ARGUMENT,
        "binding rejects a pipeline incompatible with the active render pass");
    assert(in_pass_record.logical_pipeline.handle == compatible,
        "in-pass incompatibility leaves the active pipeline unchanged");
    assert(in_pass_record.scratch.reference_count == in_pass_references,
        "in-pass incompatibility retains no pipeline reference");
    gpu::cmd_end_render_pass(&in_pass)!!;
}

fn void preserves_graphics_binding_across_render_passes() @test {
    ReviewValidationCapture capture = {};
    gpu::Device device = make_review_validation_device(&capture);
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
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
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];

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
    // Scope the validation-error window to the cross-pass/cross-bind recording
    // under test; device/resource setup is out of scope for this assertion.
    capture.errors = 0;
    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_begin_render_pass_with_state(&commands, &first_pass, &graphics_state)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.logical_pipeline.handle == draw.graphics,
        "ending a render pass preserves the logical graphics binding");
    assert(record.native_graphics_pipeline.pipeline != null,
        "ending a render pass preserves the native graphics binding");
    gpu::CommandResolutionStats before_reuse =
        gpu::command_resolution_stats();
    gpu::cmd_begin_render_pass_with_state(&commands, &second_pass, &graphics_state)!!;
    assert(gpu::command_resolution_stats().pipeline_bind_commands
            == before_reuse.pipeline_bind_commands,
        "a compatible pass reuses the selected graphics binding");
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_begin_render_pass_with_state(&commands, &second_pass, &graphics_state)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 2,
        "graphics and compute bind points retain independent native bindings");
    assert(capture.errors == 0,
        "cross-pass and cross-bind recording emits no Vulkan validation errors");
}

fn void distinct_graphics_cache_entries_emit_distinct_binds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::PipelineHandle alternate = review_distinct_graphics_pipeline(
        &device,
        gpu::Format.BGRA8_UNORM,
    );
    defer gpu::destroy_pipeline(&device, alternate)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_bind_pipeline(&commands, alternate)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 2,
        "distinct graphics cache entries emit two native binds");
    assert_shader_heap_bind_work(stats, 1);
}

fn void same_graphics_pipeline_across_passes_emits_one_bind() @test {
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
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, second)!!;
    gpu::TextureHandle third = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_texture(&device, third)!!;
    gpu::AttachmentViewHandle[3] views = {
        create_test_attachment_view(&device, first)!!,
        create_test_attachment_view(&device, second)!!,
        create_test_attachment_view(&device, third)!!,
    };
    defer destroy_test_attachment_view(&device, &views[0])!!;
    defer destroy_test_attachment_view(&device, &views[1])!!;
    defer destroy_test_attachment_view(&device, &views[2])!!;
    gpu::ColorTargetDesc[1] colors = {{ .view = views[0] }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
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
    record_val_target_to_attach(&commands, third);

    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::reset_dynamic_state_observation(state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    for (uint i = 0; i < views.len; i++) {
        colors[0].view = views[i];
        gpu::cmd_begin_render_pass(&commands, &pass)!!;
        gpu::cmd_end_render_pass(&commands)!!;
    }
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    vk::DynamicStateObservation observation =
        vk::dynamic_state_observation(state);
    assert(stats.pipeline_bind_commands == 1,
        "the same graphics pipeline across passes emits one native bind");
    assert(observation.begin_rendering_emissions == views.len
        && observation.viewport_emissions == 1
        && observation.scissor_emissions == 1
        && observation.raster_state_emissions == 1
        && observation.depth_state_emissions == 1,
        "compatible passes reuse one complete graphics-state packet");
}

fn void distinct_compute_cache_entries_emit_distinct_binds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    gpu::PipelineHandle alternate = review_distinct_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, alternate)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&commands, alternate)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 2,
        "distinct compute cache entries emit two native binds");
}

fn void compute_graphics_compute_reuses_compute_binding() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 2,
        "compute binding survives an intervening graphics bind");
}

fn void graphics_alias_updates_logical_handle_without_rebinding() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::PipelineHandle alias_pipeline = review_distinct_graphics_pipeline(
        &device,
        gpu::Format.RGBA8_UNORM,
    );
    defer gpu::destroy_pipeline(&device, alias_pipeline)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_bind_pipeline(&commands, alias_pipeline)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 1,
        "graphics aliases sharing one cache entry emit one native bind");
    assert(record.logical_pipeline.handle == alias_pipeline,
        "graphics aliases still advance the logical public handle");
}

fn void command_buffer_reuse_resets_both_pipeline_bindings() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_render_validation_target(
        &device,
        gpu::Format.RGBA8_UNORM,
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
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics.color.targets = color_targets[..];
    gpu::CommandList first = gpu::begin_commands(&allocator)!!;
    gpu::cmd_bind_pipeline(&first, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&first, draw.graphics)!!;
    gpu::cmd_set_graphics_state(&first, &graphics)!!;
    vk::CommandRecord* first_record = vk::resolve_command(&first)!!;
    assert(first_record.graphics_state_initialized
            && first_record.color_state_initialized,
        "complete state initializes both readiness domains");
    assert(first_record.descriptor_heap_binding.graphics_set_bound
            && first_record.descriptor_heap_binding.compute_set_bound,
        "the first record caches both descriptor-set bind points");
    gpu::cmd_bind_pipeline(&first, compute.pipeline)!!;
    assert(first_record.color_state_initialized
            && first_record.color_state.count == 1
            && first_record.color_state.formats[0]
                == gpu::Format.RGBA8_UNORM,
        "a compute bind preserves initialized graphics color state");
    discard_test_commands(&first)!!;

    gpu::CommandList reused = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&reused)!!;
    vk::CommandRecord* reused_record = vk::resolve_command(&reused)!!;
    assert(!reused_record.graphics_state_initialized
        && !reused_record.color_state_initialized
        && reused_record.color_state.count == 0,
        "command-buffer reuse clears graphics and color initialization");
    assert(!reused_record.descriptor_heap_binding.graphics_set_bound
            && !reused_record.descriptor_heap_binding.compute_set_bound,
        "command-buffer reuse clears both descriptor-set bind points");
    record_val_target_to_attach(&reused, target);
    gpu::reset_command_resolution_stats();
    gpu::cmd_bind_pipeline(&reused, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&reused, draw.graphics)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.pipeline_bind_commands == 2,
        "command-buffer reuse resets both native pipeline caches");
    assert_shader_heap_bind_work(stats, 2);
    gpu::cmd_begin_render_pass(&reused, &pass)!!;
    assert(@catch(gpu::cmd_draw(
        commands:       &reused,
        vertex_root:    {},
        fragment_root:  {},
        vertex_count:   0,
        instance_count: 0,
    )) == gpu::COMMAND_RECORDING_ERROR,
        "FULL draw after reuse requires graphics-state reinitialization");
    gpu::cmd_end_render_pass(&reused)!!;
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
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
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

    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    ))
        == gpu::INVALID_ARGUMENT,
        "pipeline and render attachments must use the same sample count");
    assert(record.state == gpu::internal::CommandState.RECORDING,
        "pipeline sample mismatch records no native render pass");
    assert(record.scratch.reference_count == references_before,
        "pipeline sample mismatch retains no attachment references");
}

fn void graphics_state_packet_emission_is_exact() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(
        vk::resolve_command(&fixture.commands)!!,
        color_formats[..],
    );
    gpu::ColorTargetDesc[1] colors = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    gpu::GraphicsState graphics = {
        .viewport = {
            .y         = (float)VAL_RT,
            .width     = (float)VAL_RT,
            .height    = -(float)VAL_RT,
            .min_depth = 1.0f,
            .max_depth = 0.0f,
        },
        .scissor = {
            .x      = 1,
            .y      = 2,
            .width  = (int)VAL_RT - 1,
            .height = (int)VAL_RT - 2,
        },
        .raster = {
            .topology            = gpu::PrimitiveTopology.LINES,
            .cull_mode           = gpu::CullMode.BACK,
            .front_face          = gpu::FrontFace.CLOCKWISE,
            .depth_bias_enable   = true,
            .depth_bias_constant = 1.25f,
            .depth_bias_slope    = 0.5f,
            .depth_bias_clamp    = 0.125f,
        },
        .depth = {
            .compare = gpu::CompareOp.ALWAYS,
        },
        .color = {
            .targets = color_targets[..],
        },
    };

    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_set_viewport(&fixture.commands, &graphics.viewport)!!;
    gpu::CommandResolutionStats partial_stats =
        gpu::command_resolution_stats();
    vk::DynamicStateObservation partial =
        vk::dynamic_state_observation(fixture.state);
    vk::CommandRecord* record = vk::resolve_command(&fixture.commands)!!;
    assert(partial_stats.native_commands == 1
        && partial.viewport_emissions == 1
        && partial.scissor_emissions == 0
        && partial.raster_state_emissions == 0
        && partial.depth_state_emissions == 0,
        "a pre-pass partial setter emits only its documented subset");
    assert(!record.graphics_state_initialized,
        "a partial setter does not initialize complete graphics state");

    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_render_pass(&fixture.commands, &pass)!!;
    gpu::CommandResolutionStats minimal_stats =
        gpu::command_resolution_stats();
    vk::DynamicStateObservation minimal =
        vk::dynamic_state_observation(fixture.state);
    assert(minimal_stats.recording_commands == 1
        && minimal_stats.native_commands == 1,
        "minimal begin crosses one boundary and emits one native command");
    assert(minimal.begin_rendering_emissions == 1
        && minimal.viewport_emissions == 0
        && minimal.scissor_emissions == 0
        && minimal.raster_state_emissions == 0
        && minimal.depth_state_emissions == 0,
        "minimal begin emits no graphics-state command");
    assert(!record.graphics_state_initialized,
        "minimal begin does not initialize graphics state");
    gpu::cmd_end_render_pass(&fixture.commands)!!;

    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_set_graphics_state(&fixture.commands, &graphics)!!;
    gpu::CommandResolutionStats packet_stats =
        gpu::command_resolution_stats();
    vk::DynamicStateObservation packet =
        vk::dynamic_state_observation(fixture.state);
    assert(packet_stats.recording_commands == 1
        && packet_stats.native_commands == 13,
        "complete state emits one thirteen-command packet");
    assert(packet.begin_rendering_emissions == 0
        && packet.viewport_emissions == 1
        && packet.scissor_emissions == 1
        && packet.raster_state_emissions == 1
        && packet.depth_state_emissions == 1,
        "complete state emits each supplied group exactly once");
    assert(observation_matches_graphics_state(&packet, &graphics),
        "complete state preserves every supplied value");
    assert(record.graphics_state_initialized
            && record.color_state_initialized,
        "complete state initializes both readiness domains");

    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_render_pass(&fixture.commands, &pass)!!;
    vk::DynamicStateObservation reuse =
        vk::dynamic_state_observation(fixture.state);
    assert(gpu::command_resolution_stats().native_commands == 1
        && reuse.begin_rendering_emissions == 1
        && reuse.viewport_emissions == 0
        && reuse.scissor_emissions == 0
        && reuse.raster_state_emissions == 0
        && reuse.depth_state_emissions == 0,
        "a later minimal pass reuses state without replay");
    assert(record.graphics_state_initialized,
        "pass begin preserves complete initialization");
    gpu::cmd_end_render_pass(&fixture.commands)!!;

    gpu::GraphicsState replacement =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    replacement.color.targets = color_targets[..];
    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_render_pass_with_state(
        &fixture.commands,
        &pass,
        &replacement,
    )!!;
    vk::DynamicStateObservation combined =
        vk::dynamic_state_observation(fixture.state);
    assert(gpu::command_resolution_stats().recording_commands == 1
        && gpu::command_resolution_stats().native_commands == 14,
        "convenience begin is one boundary and fourteen native commands");
    assert(combined.begin_rendering_emissions == 1
        && combined.viewport_emissions == 1
        && combined.scissor_emissions == 1
        && combined.raster_state_emissions == 1
        && combined.depth_state_emissions == 1,
        "convenience begin emits one begin and one complete packet");
    assert(observation_matches_graphics_state(&combined, &replacement),
        "convenience begin preserves every supplied value");
}

fn void invalid_graphics_state_packet_is_transactional() @test {
    SyncDiagnosticCapture capture = {};
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        contract:  gpu::ContractValidation.FULL,
        callback:  &capture_sync_diagnostic,
        user_data: &capture,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(
        vk::resolve_command(&fixture.commands)!!,
        color_formats[..],
    );
    gpu::ColorTargetDesc[1] colors = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics.color.targets = color_targets[..];
    int invalid_compare = 99;
    mem::copy(
        &graphics.depth.compare,
        &invalid_compare,
        int::size,
    );
    vk::CommandRecord* record = vk::resolve_command(&fixture.commands)!!;
    uint references_before = record.scratch.reference_count;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass_with_state",
        debug_name:     null,
        rejected_field: "GraphicsState.depth.compare",
        invariant:      "compare operation must be defined",
    );
    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &fixture.commands,
        &pass,
        &graphics,
    )) == gpu::INVALID_ARGUMENT,
        "a late-invalid packet faults before pass begin");
    gpu::CommandResolutionStats begin_stats =
        gpu::command_resolution_stats();
    vk::DynamicStateObservation begin_observation =
        vk::dynamic_state_observation(fixture.state);
    assert(begin_stats.native_commands == 0,
        "invalid initial state emits no native command");
    assert(observation_has_no_emissions(&begin_observation),
        "invalid initial state emits no partial packet");
    assert(record.state == gpu::internal::CommandState.RECORDING,
        "invalid initial state leaves the command outside a pass");
    assert(record.scratch.reference_count == references_before,
        "invalid initial state retains no attachment reference");
    assert(!record.graphics_state_initialized,
        "invalid initial state does not initialize graphics state");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    graphics.depth.compare = gpu::CompareOp.LESS;
    gpu::cmd_begin_render_pass_with_state(
        &fixture.commands,
        &pass,
        &graphics,
    )!!;
    mem::copy(
        &graphics.depth.compare,
        &invalid_compare,
        int::size,
    );
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_set_graphics_state",
        debug_name:     null,
        rejected_field: "GraphicsState.depth.compare",
        invariant:      "compare operation must be defined",
    );
    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_set_graphics_state(
        &fixture.commands,
        &graphics,
    )) == gpu::INVALID_ARGUMENT,
        "a late-invalid replacement faults before emission");
    gpu::CommandResolutionStats replacement_stats =
        gpu::command_resolution_stats();
    vk::DynamicStateObservation replacement_observation =
        vk::dynamic_state_observation(fixture.state);
    assert(replacement_stats.native_commands == 0,
        "invalid replacement emits no native command");
    assert(observation_has_no_emissions(&replacement_observation),
        "invalid replacement emits no partial state");
    assert(record.state == gpu::internal::CommandState.RECORDING_RENDER_PASS,
        "invalid replacement preserves the active pass");
    assert(record.graphics_state_initialized,
        "invalid replacement preserves prior complete initialization");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );
}

fn void late_invalid_color_in_complete_graphics_state_is_transactional() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    gpu::Format[1] color_formats = {
        gpu::Format.R8_UINT,
    };
    select_graphics_color_pipeline(record, color_formats[..]);
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::alpha_blend(),
    };
    graphics.color.targets = color_targets[..];

    vk::reset_dynamic_state_observation(fixture.state);
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_set_graphics_state(
        &fixture.commands,
        &graphics,
    )) == gpu::INVALID_ARGUMENT,
        "integer blending rejects the complete packet");
    vk::DynamicStateObservation observation =
        vk::dynamic_state_observation(fixture.state);
    assert(gpu::command_resolution_stats().native_commands == 0
        && observation_has_no_emissions(&observation),
        "late color validation rejects before any ordinary or color emission");
    assert(record.state == gpu::internal::CommandState.RECORDING
        && !record.graphics_state_initialized
        && !record.color_state_initialized,
        "late color rejection preserves command phase and readiness");
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
    gpu::Viewport[18] rejected = {
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
        { .width = 1.0f, .height = 1.0f, .min_depth = -0.1f, .max_depth = 1.0f },
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

fn void both_modes_lower_dynamic_state_identically() @test {
    DynamicStateFixture trusted = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.TRUSTED,
    );
    defer destroy_dynamic_state_fixture(&trusted);
    DynamicStateFixture full = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&full);

    gpu::ColorTargetDesc[1] trusted_colors = {{ .view = trusted.view }};
    gpu::ColorTargetDesc[1] full_colors = {{ .view = full.view }};
    gpu::RenderPassDesc trusted_pass = {
        .colors = trusted_colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::RenderPassDesc full_pass = {
        .colors = full_colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    gpu::GraphicsState graphics = {
        .viewport = {
            .x         = -1.0f,
            .y         = (float)VAL_RT,
            .width     = (float)VAL_RT + 2.0f,
            .height    = -(float)VAL_RT,
            .min_depth = 1.0f,
            .max_depth = 0.0f,
        },
        .scissor = {
            .x      = (int)VAL_RT,
            .y      = (int)VAL_RT,
            .width  = 1,
            .height = 1,
        },
        .raster = {
            .topology   = gpu::PrimitiveTopology.POINTS,
            .cull_mode  = gpu::CullMode.FRONT,
            .front_face = gpu::FrontFace.CLOCKWISE,
        },
        .depth = {
            .compare = gpu::CompareOp.GREATER_EQUAL,
        },
        .color = {
            .targets = color_targets[..],
        },
    };
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    select_graphics_color_pipeline(
        vk::resolve_command(&trusted.commands)!!,
        color_formats[..],
    );
    select_graphics_color_pipeline(
        vk::resolve_command(&full.commands)!!,
        color_formats[..],
    );
    gpu::cmd_begin_render_pass_with_state(
        &trusted.commands,
        &trusted_pass,
        &graphics,
    )!!;
    gpu::cmd_begin_render_pass_with_state(
        &full.commands,
        &full_pass,
        &graphics,
    )!!;

    vk::DynamicStateObservation trusted_observation =
        vk::dynamic_state_observation(trusted.state);
    vk::DynamicStateObservation full_observation =
        vk::dynamic_state_observation(full.state);
    assert(trusted_observation.begin_rendering_emissions == 1
        && full_observation.begin_rendering_emissions == 1,
        "each mode begins rendering exactly once");
    assert(trusted_observation.viewport_emissions == 1
        && trusted_observation.scissor_emissions == 1
        && trusted_observation.raster_state_emissions == 1
        && trusted_observation.depth_state_emissions == 1
        && full_observation.viewport_emissions == 1
        && full_observation.scissor_emissions == 1
        && full_observation.raster_state_emissions == 1
        && full_observation.depth_state_emissions == 1,
        "each mode emits one complete packet");
    assert(observation_matches_graphics_state(
        &trusted_observation,
        &graphics,
    ) && observation_matches_graphics_state(
        &full_observation,
        &graphics,
    ), "trusted and full packet lowering is identical");
}

fn void trusted_dynamic_state_bypasses_full_device_bounds() @test {
    DynamicStateFixture trusted = make_dynamic_state_fixture(
        gpu::ContractValidation.TRUSTED,
    );
    defer destroy_dynamic_state_fixture(&trusted);
    DynamicStateFixture full = make_dynamic_state_fixture(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&full);
    full.state.viewport_limits = {
        .max_width  = 1,
        .max_height = 1,
        .min_bound  = 0.0f,
        .max_bound  = 1.0f,
    };

    gpu::Viewport over_limit = {
        .width     = 2.0f,
        .height    = 1.0f,
        .max_depth = 1.0f,
    };
    gpu::cmd_set_viewport(&trusted.commands, &over_limit)!!;
    assert(@catch(gpu::cmd_set_viewport(
        &full.commands,
        &over_limit,
    )) == gpu::INVALID_ARGUMENT,
        "FULL rejects a device-over-limit viewport");

    vk::DynamicStateObservation trusted_observation =
        vk::dynamic_state_observation(trusted.state);
    vk::DynamicStateObservation full_observation =
        vk::dynamic_state_observation(full.state);
    assert(trusted_observation.viewport_emissions == 1
        && full_observation.viewport_emissions == 0,
        "TRUSTED forwards while FULL rejects before emission");
    assert(native_viewport_matches(
        &trusted_observation.viewport,
        &over_limit,
    ), "TRUSTED preserves the over-limit native viewport");
}

fn void dynamic_color_packet_emits_three_heterogeneous_arrays() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    DynamicColorCapture capture;
    vk::VkDynamicColorDispatch original =
        install_dynamic_color_capture(fixture.state, &capture);
    defer {
        fixture.state.device_dispatch.dynamic_color = original;
        dynamic_color_capture = null;
    }
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    gpu::Format[2] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA16_FLOAT,
    };
    select_graphics_color_pipeline(record, formats[..]);
    gpu::ColorWriteMask red_alpha = { .red, .alpha };
    gpu::ColorTargetState[2] targets = {
        gpu::color_blend_disabled(red_alpha),
        gpu::alpha_blend(),
    };
    gpu::ColorWriteMask green_blue = { .green, .blue };
    targets[1].write_mask = green_blue;
    gpu::ColorState color = { .targets = targets[..] };

    gpu::reset_command_resolution_stats();
    gpu::cmd_set_color_state(&fixture.commands, &color)!!;

    assert(capture.blend_enable_calls == 1
        && capture.blend_equation_calls == 1
        && capture.write_mask_calls == 1
        && capture.first_attachment == 0
        && capture.target_count == 2,
        "one packet emits exactly three arrays over the complete target range");
    assert(capture.blend_enables[0] == vk::FALSE
        && capture.blend_enables[1] == vk::TRUE,
        "blend enables preserve heterogeneous target order");
    vk::ColorBlendEquationEXT* disabled = &capture.blend_equations[0];
    assert(disabled.src_color_blend_factor == vk::BlendFactor.ONE
        && disabled.dst_color_blend_factor == vk::BlendFactor.ZERO
        && disabled.color_blend_op == vk::BlendOp.ADD
        && disabled.src_alpha_blend_factor == vk::BlendFactor.ONE
        && disabled.dst_alpha_blend_factor == vk::BlendFactor.ZERO
        && disabled.alpha_blend_op == vk::BlendOp.ADD,
        "disabled blend lowers to the canonical opaque equation");
    vk::ColorBlendEquationEXT* alpha = &capture.blend_equations[1];
    assert(alpha.src_color_blend_factor == vk::BlendFactor.SRC_ALPHA
        && alpha.dst_color_blend_factor
            == vk::BlendFactor.ONE_MINUS_SRC_ALPHA
        && alpha.color_blend_op == vk::BlendOp.ADD
        && alpha.src_alpha_blend_factor == vk::BlendFactor.ONE
        && alpha.dst_alpha_blend_factor
            == vk::BlendFactor.ONE_MINUS_SRC_ALPHA
        && alpha.alpha_blend_op == vk::BlendOp.ADD,
        "enabled blend factors and operations lower exactly");
    vk::ColorComponentFlags red_alpha_native =
        (vk::ColorComponentFlags)(
            (uint)vk::ColorComponentFlagBits.R_BIT
            | (uint)vk::ColorComponentFlagBits.A_BIT
        );
    vk::ColorComponentFlags green_blue_native =
        (vk::ColorComponentFlags)(
            (uint)vk::ColorComponentFlagBits.G_BIT
            | (uint)vk::ColorComponentFlagBits.B_BIT
        );
    assert(capture.write_masks[0] == red_alpha_native
        && capture.write_masks[1] == green_blue_native,
        "write masks preserve heterogeneous target order");
    assert(gpu::command_resolution_stats().native_commands == 3,
        "one nonempty packet records exactly three native commands");
}

fn void trusted_and_full_color_packets_lower_identically() @test {
    DynamicStateFixture trusted = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.TRUSTED,
    );
    defer destroy_dynamic_state_fixture(&trusted);
    DynamicStateFixture full = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&full);
    gpu::Format[2] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA16_FLOAT,
    };
    select_graphics_color_pipeline(
        vk::resolve_command(&trusted.commands)!!,
        formats[..],
    );
    select_graphics_color_pipeline(
        vk::resolve_command(&full.commands)!!,
        formats[..],
    );
    gpu::ColorTargetState[2] targets = {
        gpu::premultiplied_alpha_blend(),
        gpu::additive_blend(),
    };
    gpu::ColorState color = { .targets = targets[..] };
    DynamicColorCapture trusted_capture;
    vk::VkDynamicColorDispatch trusted_original =
        install_dynamic_color_capture(
            trusted.state,
            &trusted_capture,
        );
    gpu::cmd_set_color_state(&trusted.commands, &color)!!;
    trusted.state.device_dispatch.dynamic_color = trusted_original;

    DynamicColorCapture full_capture;
    vk::VkDynamicColorDispatch full_original =
        install_dynamic_color_capture(
            full.state,
            &full_capture,
        );
    defer {
        full.state.device_dispatch.dynamic_color = full_original;
        dynamic_color_capture = null;
    }
    gpu::cmd_set_color_state(&full.commands, &color)!!;

    assert(dynamic_color_captures_match(
        &trusted_capture,
        &full_capture,
    ), "TRUSTED and FULL emit identical native color arrays");
}

fn void zero_target_color_packet_emits_no_native_command() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    DynamicColorCapture capture;
    vk::VkDynamicColorDispatch original =
        install_dynamic_color_capture(fixture.state, &capture);
    defer {
        fixture.state.device_dispatch.dynamic_color = original;
        dynamic_color_capture = null;
    }
    gpu::Format[] formats;
    select_graphics_color_pipeline(
        vk::resolve_command(&fixture.commands)!!,
        formats[..],
    );
    gpu::ColorState color;

    gpu::reset_command_resolution_stats();
    gpu::cmd_set_color_state(&fixture.commands, &color)!!;

    assert(capture.blend_enable_calls == 0
        && capture.blend_equation_calls == 0
        && capture.write_mask_calls == 0
        && gpu::command_resolution_stats().native_commands == 0,
        "a depth-only dynamic domain emits no color command");

    gpu::GraphicsState graphics =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::reset_command_resolution_stats();
    gpu::cmd_set_graphics_state(&fixture.commands, &graphics)!!;
    assert(gpu::command_resolution_stats().native_commands == 10
            && capture.blend_enable_calls == 0
            && capture.blend_equation_calls == 0
            && capture.write_mask_calls == 0,
        "a complete depth-only packet emits ten ordinary commands and no color commands");
}

fn void full_color_validation_rejects_before_mutation() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    DynamicColorCapture capture;
    vk::VkDynamicColorDispatch original =
        install_dynamic_color_capture(fixture.state, &capture);
    defer {
        fixture.state.device_dispatch.dynamic_color = original;
        dynamic_color_capture = null;
    }
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    gpu::Format[2] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.R8_UINT,
    };
    select_graphics_color_pipeline(record, formats[..]);
    vk::BoundPipeline pipeline_before = record.logical_pipeline;
    gpu::internal::CommandState phase_before = record.state;
    gpu::ColorTargetState[2] targets = {
        gpu::color_blend_disabled(),
        gpu::alpha_blend(),
    };
    gpu::ColorState color = { .targets = targets[..] };

    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_set_color_state(
        &fixture.commands,
        &color,
    )) == gpu::INVALID_ARGUMENT,
        "FULL rejects blending on an integer target");
    assert(capture.blend_enable_calls == 0
        && capture.blend_equation_calls == 0
        && capture.write_mask_calls == 0
        && gpu::command_resolution_stats().native_commands == 0,
        "late-target validation completes before the first native call");
    assert(record.state == phase_before
        && record.logical_pipeline.handle == pipeline_before.handle
        && record.logical_pipeline.kind == pipeline_before.kind
        && record.logical_pipeline.render.color_count
            == pipeline_before.render.color_count
        && record.logical_pipeline.render.color_formats[0]
            == pipeline_before.render.color_formats[0]
        && record.logical_pipeline.render.color_formats[1]
            == pipeline_before.render.color_formats[1],
        "rejection preserves command phase and selected pipeline state");

    targets[1] = gpu::color_blend_disabled();
    uint invalid_write_mask = 1u << 8;
    mem::copy(
        &targets[0].write_mask,
        &invalid_write_mask,
        uint::size,
    );
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_set_color_state(
        &fixture.commands,
        &color,
    )) == gpu::INVALID_ARGUMENT,
        "FULL rejects write-mask bits outside RGBA");
    assert(capture.blend_enable_calls == 0
            && capture.blend_equation_calls == 0
            && capture.write_mask_calls == 0
            && gpu::command_resolution_stats().native_commands == 0,
        "invalid write masks reject before native mutation");

    targets[0] = gpu::color_blend_disabled();
    gpu::ColorState short_color = {
        .targets = targets[:1],
    };
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_set_color_state(
        &fixture.commands,
        &short_color,
    )) == gpu::INVALID_ARGUMENT,
        "FULL rejects color target counts that differ from the bound pipeline");
    assert(capture.blend_enable_calls == 0
            && capture.blend_equation_calls == 0
            && capture.write_mask_calls == 0
            && gpu::command_resolution_stats().native_commands == 0,
        "target-count rejection occurs before native mutation");
}

fn void complete_graphics_state_requires_the_graphics_bind_point() @test {
    SyncDiagnosticCapture capture = {};
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        contract:  gpu::ContractValidation.FULL,
        callback:  &capture_sync_diagnostic,
        user_data: &capture,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics.color.targets = color_targets[..];

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_set_graphics_state",
        debug_name:     null,
        rejected_field: "pipeline",
        invariant:      "the required pipeline bind point must be active",
    );
    assert(@catch(gpu::cmd_set_graphics_state(
        &fixture.commands,
        &graphics,
    )) == gpu::INVALID_ARGUMENT,
        "complete graphics state rejects when no pipeline is bound");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.PIPELINE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    gpu::ColorTargetDesc[1] colors = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    record.logical_pipeline = {
        .handle = {
            .owner      = 1,
            .index      = 2,
            .generation = 3,
        },
        .kind = vk::PipelineKind.COMPUTE,
    };
    uint references_before = record.scratch.reference_count;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass_with_state",
        debug_name:     null,
        rejected_field: "pipeline",
        invariant:      "pipeline kind must match the GRAPHICS bind point",
    );
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &fixture.commands,
        &pass,
        &graphics,
    )) == gpu::INVALID_ARGUMENT,
        "combined begin rejects when only a compute pipeline is bound");
    assert(record.state == gpu::internal::CommandState.RECORDING
            && record.scratch.reference_count == references_before,
        "wrong bind-point rejection records no pass or references");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.PIPELINE,
        has_public_identity: true,
        index:               2,
        generation:          3,
    );
}

fn void dynamic_color_domain_persists_until_incompatible_bind() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    DynamicColorCapture capture;
    vk::VkDynamicColorDispatch original =
        install_dynamic_color_capture(fixture.state, &capture);
    defer {
        fixture.state.device_dispatch.dynamic_color = original;
        dynamic_color_capture = null;
    }
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    select_graphics_color_pipeline(record, formats[..]);
    gpu::ColorTargetState[1] targets = {
        gpu::premultiplied_alpha_blend(),
    };
    gpu::ColorState color = { .targets = targets[..] };

    gpu::cmd_set_color_state(&fixture.commands, &color)!!;
    assert(record.color_state_initialized
        && record.color_state.count == 1
        && record.color_state.formats[0] == gpu::Format.RGBA8_UNORM
        && !record.graphics_state_initialized,
        "a color packet records its domain without initializing graphics");

    gpu::ColorTargetDesc[1] attachments = {{ .view = fixture.view }};
    gpu::RenderPassDesc pass = {
        .colors = attachments[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::cmd_begin_render_pass(&fixture.commands, &pass)!!;
    gpu::cmd_end_render_pass(&fixture.commands)!!;
    assert(record.color_state_initialized,
        "a compatible pass boundary preserves color initialization");

    vk::BoundPipeline compatible_alias = record.logical_pipeline;
    compatible_alias.handle.index++;
    vk::update_full_color_state_for_bind(
        record,
        &compatible_alias,
    );
    assert(record.color_state_initialized,
        "a compatible graphics pipeline alias preserves color initialization");

    vk::BoundPipeline incompatible = compatible_alias;
    incompatible.render.color_formats[0] = gpu::Format.R8_UNORM;
    vk::update_full_color_state_for_bind(record, &incompatible);
    assert(!record.color_state_initialized
        && record.color_state.count == 0,
        "an incompatible format domain clears initialization precisely");
    assert(capture.blend_enable_calls == 1
        && capture.blend_equation_calls == 1
        && capture.write_mask_calls == 1,
        "pass and bind changes never replay the explicit color packet");
}

fn void dynamic_color_initialization_gates_direct_and_generated_draws() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
        gpu::ContractValidation.FULL,
    );
    defer destroy_dynamic_state_fixture(&fixture);
    DynamicColorCapture capture;
    vk::VkDynamicColorDispatch original =
        install_dynamic_color_capture(fixture.state, &capture);
    defer {
        fixture.state.device_dispatch.dynamic_color = original;
        dynamic_color_capture = null;
    }
    vk::CommandRecord* record =
        vk::resolve_command(&fixture.commands)!!;
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    select_graphics_color_pipeline(record, formats[..]);
    record.graphics_state_initialized = true;

    assert(@catch(vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        false,
    )) == gpu::COMMAND_RECORDING_ERROR
        && @catch(vk::validate_color_execution_state_for_test(
            fixture.state,
            record,
            true,
        )) == gpu::COMMAND_RECORDING_ERROR,
        "direct and generated draws both require initialized dynamic color");

    gpu::ColorTargetState[1] targets = {
        gpu::additive_blend(),
    };
    gpu::ColorState color = { .targets = targets[..] };
    gpu::cmd_set_color_state(&fixture.commands, &color)!!;
    vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        false,
    )!!;
    vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        true,
    )!!;

    record.color_state.formats[0] = gpu::Format.R8_UNORM;
    assert(@catch(vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        false,
    )) == gpu::COMMAND_RECORDING_ERROR
        && @catch(vk::validate_color_execution_state_for_test(
            fixture.state,
            record,
            true,
        )) == gpu::COMMAND_RECORDING_ERROR,
        "both draw paths reject a stale incompatible color domain");

    gpu::Format[] no_formats;
    select_graphics_color_pipeline(record, no_formats[..]);
    record.color_state_initialized = false;
    record.color_state = {};
    vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        false,
    )!!;
    vk::validate_color_execution_state_for_test(
        fixture.state,
        record,
        true,
    )!!;
    assert(capture.blend_enable_calls == 1
        && capture.blend_equation_calls == 1
        && capture.write_mask_calls == 1,
        "depth-only validation requires no packet and emits no hidden replay");
}
