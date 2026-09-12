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

struct PipelineBindCapture {
    vk::Pipeline[4] pipelines;
    uint            calls;
    uint            graphics_calls;
    uint            compute_calls;
    uint            topology_calls;
    vk::PrimitiveTopology topology;
}

fn void capture_pipeline_topology(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PrimitiveTopology topology,
) {
    (void)command_buffer;
    PipelineBindCapture* capture = (PipelineBindCapture*)user_data;
    capture.topology_calls++;
    capture.topology = topology;
}

fn void capture_pipeline_bind(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineBindPoint bind_point,
    vk::Pipeline pipeline,
) {
    (void)command_buffer;
    PipelineBindCapture* capture = (PipelineBindCapture*)user_data;
    if (capture.calls < capture.pipelines.len) {
        capture.pipelines[capture.calls] = pipeline;
    }
    capture.calls++;
    if (bind_point == vk::PipelineBindPoint.COMPUTE) {
        capture.compute_calls++;
    } else if (bind_point == vk::PipelineBindPoint.GRAPHICS) {
        capture.graphics_calls++;
    }
}

fn void? bind_pipeline_with_capture(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    PipelineBindCapture* capture,
) {
    vk::PipelineBindOps ops = {
        .user_data = capture,
        .bind      = &capture_pipeline_bind,
        .set_topology = &capture_pipeline_topology,
    };
    return vk::vk_cmd_bind_pipeline_with_ops(
        vk::resolve_command(commands)!,
        pipeline,
        &ops,
    );
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
        .spirv       = VAL_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .spirv       = VAL_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::Format[1] color_formats = { format };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = vertex_desc,
        .fragment_shader = fragment_desc,
        .color_formats   = color_formats[..],
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::PipelineHandle review_distinct_compute_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc shader_desc = {
        .spirv       = REVIEW_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc desc = {
        .shader = shader_desc,
    };
    return gpu::create_compute_pipeline(device, &desc)!!;
}

fn DynamicStateFixture make_dynamic_state_fixture_before_pass(
    gpu::DebugMessageCallback callback = null,
    void* user_data = null,
) {
    gpu::RuntimeDesc runtime_desc = {
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
    return fixture;
}

fn DynamicStateFixture make_dynamic_state_fixture(
    gpu::DebugMessageCallback callback = null,
    void* user_data = null,
) {
    DynamicStateFixture fixture = make_dynamic_state_fixture_before_pass(
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
    gpu::cmd_begin_render_pass(&fixture.commands, &pass)!!;
    select_graphics_color_pipeline(
        vk::resolve_command(&fixture.commands)!!,
        color_formats[..],
    );
    gpu::cmd_set_graphics_state(&fixture.commands, &graphics_state)!!;
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

fn void texture_create_info_and_default_view_preserve_dimension() @test {
    gpu::TextureDesc three_dimensional = {
        .access       = { .compute },
        .width        = 8,
        .height       = 4,
        .depth        = 4,
        .mip_levels   = 3,
        .array_layers = 1,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled, .storage },
    };
    uint[1] families;
    vk::ImageCreateInfo image_info = vk::make_texture_create_info(
        desc:           &three_dimensional,
        queue_families: families[:0],
    );
    assert(image_info.image_type == vk::ImageType.TYPE_3D
        && image_info.extent.width == three_dimensional.width
        && image_info.extent.height == three_dimensional.height
        && image_info.extent.depth == three_dimensional.depth
        && image_info.array_layers == 1
        && image_info.samples == vk::SampleCountFlagBits.COUNT_1_BIT,
        "positive depth lowers to one exact three-dimensional image");

    gpu::TextureDesc unvalidated_three_dimensional = three_dimensional;
    unvalidated_three_dimensional.array_layers = 2;
    image_info = vk::make_texture_create_info(
        desc:           &unvalidated_three_dimensional,
        queue_families: families[:0],
    );
    assert(image_info.array_layers == 1,
        "private three-dimensional lowering clamps native array layers");

    vk::ImageViewCreateInfo view_info =
        vk::default_texture_view_create_info(
            desc:         &three_dimensional,
            image:        (vk::Image)(uptr)1,
            mip_levels:   three_dimensional.mip_levels,
            array_layers: 1,
        );
    assert(view_info.view_type == vk::ImageViewType.TYPE_3D
        && view_info.subresource_range.base_array_layer == 0
        && view_info.subresource_range.layer_count == 1,
        "three-dimensional default views cover one native array layer");

    gpu::TextureDesc two_dimensional = three_dimensional;
    two_dimensional.depth = 0;
    image_info = vk::make_texture_create_info(
        desc:           &two_dimensional,
        queue_families: families[:0],
    );
    assert(image_info.image_type == vk::ImageType.TYPE_2D
        && image_info.extent.depth == 1,
        "zero depth preserves two-dimensional native lowering");
}

fn void texture_copy_regions_preserve_dimension_and_safe_span_bounds() @test {
    vk::ResolvedSpan span = {
        .size = 4096,
    };
    vk::TextureSlot two_dimensional = {
        .format       = gpu::Format.RGBA8_UNORM,
        .width        = 8,
        .height       = 6,
        .mip_levels   = 2,
        .array_layers = 4,
    };
    vk::BufferImageCopy two_d = vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &two_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        1,
        layer_count:       2,
        x:                 3,
        y:                 1,
        z:                 0,
        width:             0,
        height:            0,
        depth:             0,
    )!!;
    assert(two_d.image_offset.x == 3
        && two_d.image_offset.y == 1
        && two_d.image_offset.z == 0
        && two_d.image_extent.width == 5
        && two_d.image_extent.height == 5
        && two_d.image_extent.depth == 1
        && two_d.image_subresource.base_array_layer == 1
        && two_d.image_subresource.layer_count == 2,
        "two-dimensional zero extents select the remaining x/y region");

    two_d = vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &two_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        0,
        layer_count:       0,
        x:                 0,
        y:                 0,
        z:                 0,
        width:             1,
        height:            1,
        depth:             1,
    )!!;
    assert(two_d.image_extent.depth == 1,
        "explicit depth one preserves two-dimensional copies");

    fault two_d_z = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &two_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        0,
        layer_count:       0,
        x:                 0,
        y:                 0,
        z:                 1,
        width:             1,
        height:            1,
        depth:             1,
    ));
    fault two_d_depth = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &two_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        0,
        layer_count:       0,
        x:                 0,
        y:                 0,
        z:                 0,
        width:             1,
        height:            1,
        depth:             2,
    ));
    assert(two_d_z == gpu::INVALID_ARGUMENT
        && two_d_depth == gpu::INVALID_ARGUMENT,
        "two-dimensional copies reject volume coordinates");

    vk::TextureSlot three_dimensional = {
        .format       = gpu::Format.RGBA8_UNORM,
        .width        = 8,
        .height       = 4,
        .depth        = 8,
        .mip_levels   = 4,
        .array_layers = 1,
    };
    vk::BufferImageCopy three_d = vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 0,
        mip:               1,
        base_layer:        0,
        layer_count:       0,
        x:                 1,
        y:                 0,
        z:                 1,
        width:             0,
        height:            0,
        depth:             0,
    )!!;
    assert(three_d.image_offset.x == 1
        && three_d.image_offset.y == 0
        && three_d.image_offset.z == 1
        && three_d.image_extent.width == 3
        && three_d.image_extent.height == 2
        && three_d.image_extent.depth == 3
        && three_d.image_subresource.base_array_layer == 0
        && three_d.image_subresource.layer_count == 1,
        "three-dimensional zero extents select the remaining mip volume");

    fault three_d_layer = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        1,
        layer_count:       1,
        x:                 0,
        y:                 0,
        z:                 0,
        width:             1,
        height:            1,
        depth:             1,
    ));
    fault three_d_layers = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 0,
        mip:               0,
        base_layer:        0,
        layer_count:       2,
        x:                 0,
        y:                 0,
        z:                 0,
        width:             1,
        height:            1,
        depth:             1,
    ));
    fault three_d_z = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 0,
        mip:               1,
        base_layer:        0,
        layer_count:       1,
        x:                 0,
        y:                 0,
        z:                 5,
        width:             1,
        height:            1,
        depth:             1,
    ));
    fault three_d_depth = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 0,
        mip:               1,
        base_layer:        0,
        layer_count:       1,
        x:                 0,
        y:                 0,
        z:                 1,
        width:             1,
        height:            1,
        depth:             4,
    ));
    assert(three_d_layer == gpu::INVALID_ARGUMENT
        && three_d_layers == gpu::INVALID_ARGUMENT
        && three_d_z == gpu::INVALID_ARGUMENT
        && three_d_depth == gpu::INVALID_ARGUMENT,
        "three-dimensional copies reject array layers and out-of-range depth");

    span.size = 95;
    fault short_padded_span = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 4,
        mip:               1,
        base_layer:        0,
        layer_count:       1,
        x:                 1,
        y:                 0,
        z:                 1,
        width:             3,
        height:            2,
        depth:             3,
    ));
    span.size = 4096;
    fault short_row = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &three_dimensional,
        row_length_texels: 2,
        mip:               1,
        base_layer:        0,
        layer_count:       1,
        x:                 1,
        y:                 0,
        z:                 1,
        width:             3,
        height:            2,
        depth:             3,
    ));
    vk::TextureSlot overflow_texture = {
        .format       = gpu::Format.RGBA8_UNORM,
        .width        = uint::max,
        .height       = uint::max,
        .depth        = 1,
        .mip_levels   = 1,
        .array_layers = 1,
    };
    span.size = usz::max;
    fault byte_overflow = @catch(vk::resolve_texture_copy_region(
        resolved:          &span,
        slot:              &overflow_texture,
        row_length_texels: uint::max,
        mip:               0,
        base_layer:        0,
        layer_count:       1,
        x:                 0,
        y:                 0,
        z:                 0,
        width:             1,
        height:            uint::max,
        depth:             1,
    ));
    assert(short_padded_span == gpu::INVALID_ARGUMENT
        && short_row == gpu::INVALID_ARGUMENT
        && byte_overflow == gpu::INVALID_ARGUMENT,
        "multi-slice span sizing rejects padding, row, and arithmetic faults");
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
        .color_formats = color_formats[..],
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
        .color_formats = color_formats[..],
        .debug_name    = "first",
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
    assert(vk::PipelineKey::size == 64,
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
    gpu::cmd_begin_render_pass(&commands, &first_pass)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.logical_pipeline.handle == draw.graphics,
        "ending a render pass preserves the logical graphics binding");
    assert(record.native_graphics_pipeline.pipeline != null,
        "ending a render pass preserves the native graphics binding");
    gpu::cmd_begin_render_pass(&commands, &second_pass)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_begin_render_pass(&commands, &second_pass)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    assert(capture.errors == 0,
        "cross-pass and cross-bind recording emits no Vulkan validation errors");
}

fn void graphics_alias_updates_logical_handle() @test {
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

    PipelineBindCapture capture;
    bind_pipeline_with_capture(
        &commands,
        draw.graphics,
        &capture,
    )!!;
    bind_pipeline_with_capture(
        &commands,
        alias_pipeline,
        &capture,
    )!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(capture.calls == 1 && capture.graphics_calls == 1,
        "graphics aliases sharing one cache entry emit one native bind");
    assert(record.logical_pipeline.handle == alias_pipeline,
        "graphics aliases still advance the logical public handle");
}

fn void mesh_state_replays_topology_on_graphics_bind() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    record.logical_pipeline.kind = vk::PipelineKind.MESH;
    gpu::GraphicsState state = gpu::render_geometry_state(VAL_RT, VAL_RT)!!;
    state.raster.topology = gpu::PrimitiveTopology.LINES;
    gpu::cmd_set_graphics_state(&commands, &state)!!;
    assert(record.graphics_state_initialized && !record.raster_topology_emitted);

    PipelineBindCapture capture;
    bind_pipeline_with_capture(&commands, draw.graphics, &capture)!!;
    assert(capture.topology_calls == 1
        && capture.topology == vk::PrimitiveTopology.LINE_LIST
        && record.raster_topology_emitted);
    bind_pipeline_with_capture(&commands, draw.graphics, &capture)!!;
    assert(capture.topology_calls == 1);
}

fn void alternating_bind_points_preserve_native_caches() @test {
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

    PipelineBindCapture capture;
    bind_pipeline_with_capture(
        &commands,
        compute.pipeline,
        &capture,
    )!!;
    bind_pipeline_with_capture(
        &commands,
        draw.graphics,
        &capture,
    )!!;
    bind_pipeline_with_capture(
        &commands,
        compute.pipeline,
        &capture,
    )!!;
    bind_pipeline_with_capture(
        &commands,
        draw.graphics,
        &capture,
    )!!;

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(capture.calls == 2
            && capture.compute_calls == 1
            && capture.graphics_calls == 1,
        "alternating bind points retain independent native suppression caches");
    assert(record.native_compute_pipeline.pipeline == capture.pipelines[0]
            && record.native_graphics_pipeline.pipeline == capture.pipelines[1],
        "both native caches preserve their last emitted pipelines");
    assert(record.logical_pipeline.handle == draw.graphics,
        "suppressed binds still publish the complete logical snapshot");
}

fn void distinct_pipelines_on_one_bind_point_emit_distinct_binds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    gpu::PipelineHandle other = review_distinct_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, other)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    PipelineBindCapture capture;
    bind_pipeline_with_capture(&commands, compute.pipeline, &capture)!!;
    bind_pipeline_with_capture(&commands, other, &capture)!!;

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(capture.calls == 2 && capture.compute_calls == 2,
        "a different pipeline on one bind point is never suppressed");
    assert(capture.pipelines[0] != capture.pipelines[1],
        "the two binds name distinct native pipelines");
    assert(record.native_compute_pipeline.pipeline == capture.pipelines[1],
        "the native cache retains the last emitted pipeline");
    assert(record.logical_pipeline.handle == other,
        "the logical snapshot follows the last bind");
}

fn void command_buffer_reuse_resets_graphics_and_descriptor_state() @test {
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
    assert(first_record.graphics_state_initialized,
        "complete state initializes topology persistence");
    assert(first_record.descriptor_heap_binding.graphics_set_bound
            && first_record.descriptor_heap_binding.compute_set_bound,
        "the first record caches both descriptor-set bind points");
    gpu::cmd_bind_pipeline(&first, compute.pipeline)!!;
    discard_test_commands(&first)!!;

    gpu::CommandList reused = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&reused)!!;
    vk::CommandRecord* reused_record = vk::resolve_command(&reused)!!;
    assert(!reused_record.logical_pipeline.handle.is_valid()
            && reused_record.logical_pipeline.pipeline == null
            && reused_record.logical_pipeline.layout == null
            && reused_record.logical_pipeline.generated_dispatch_layout == null
            && reused_record.logical_pipeline.cache_entry == 0
            && reused_record.native_graphics_pipeline.pipeline == null
            && reused_record.native_graphics_pipeline.cache_entry == 0
            && reused_record.native_compute_pipeline.pipeline == null
            && reused_record.native_compute_pipeline.cache_entry == 0,
        "command-buffer reuse clears logical and both native pipeline caches");
    assert(!reused_record.graphics_state_initialized
,
        "command-buffer reuse clears topology initialization");
    assert(!reused_record.descriptor_heap_binding.graphics_set_bound
            && !reused_record.descriptor_heap_binding.compute_set_bound,
        "command-buffer reuse clears both descriptor-set bind points");
    record_val_target_to_attach(&reused, target);
    gpu::cmd_bind_pipeline(&reused, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&reused, draw.graphics)!!;
    gpu::cmd_begin_render_pass(&reused, &pass)!!;
    gpu::cmd_set_graphics_state(&reused, &graphics)!!;
    gpu::cmd_end_render_pass(&reused)!!;
}

fn void graphics_state_lowering_is_exact() @test {
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
    vk::LoweredGraphicsState lowered =
        vk::lower_graphics_state(&graphics)!!;
    assert(native_viewport_matches(&lowered.viewport, &graphics.viewport)
        && native_scissor_matches(&lowered.scissor, &graphics.scissor),
        "viewport and scissor lowering preserve every supplied value");
    assert(lowered.raster.topology == graphics.raster.topology
        && lowered.raster.cull_mode == graphics.raster.cull_mode
        && lowered.raster.front_face == graphics.raster.front_face
        && lowered.raster.depth_bias_enable
            == graphics.raster.depth_bias_enable
        && lowered.raster.depth_bias_constant
            == graphics.raster.depth_bias_constant
        && lowered.raster.depth_bias_slope
            == graphics.raster.depth_bias_slope
        && lowered.raster.depth_bias_clamp
            == graphics.raster.depth_bias_clamp,
        "raster lowering preserves every supplied value");
    assert(lowered.depth.test_enable == graphics.depth.test_enable
        && lowered.depth.write_enable == graphics.depth.write_enable
        && lowered.depth.compare == graphics.depth.compare,
        "depth lowering preserves every supplied value");
    assert(lowered.color.count == 1
        && lowered.color.blend_enables[0] == vk::FALSE
        && lowered.color.blend_equations[0].src_color_blend_factor
            == vk::BlendFactor.ONE
        && lowered.color.blend_equations[0].dst_color_blend_factor
            == vk::BlendFactor.ZERO
        && lowered.color.write_masks[0]
            == (vk::ColorComponentFlags)(
                (uint)vk::ColorComponentFlagBits.R_BIT
                | (uint)vk::ColorComponentFlagBits.G_BIT
                | (uint)vk::ColorComponentFlagBits.B_BIT
                | (uint)vk::ColorComponentFlagBits.A_BIT
            ),
        "color lowering uses the canonical opaque native state");
}

fn void viewport_device_bounds_accept_native_shapes_exactly() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture(
    );
    defer destroy_dynamic_state_fixture(&fixture);

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

    vk::Viewport native = vk::lower_viewport(&negative_origin);
    assert(native_viewport_matches(&native, &negative_origin),
        "viewport lowering preserves every accepted field");
}

fn void scissor_representation_accepts_overscan_and_empty_clip() @test {
    DynamicStateFixture fixture = make_dynamic_state_fixture(
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
    vk::Rect2D native = vk::lower_scissor(&empty);
    assert(native_scissor_matches(&native, &empty),
        "empty scissor preserves exact native offset and extent");

}

fn void dynamic_color_lowering_preserves_heterogeneous_targets() @test {
    gpu::ColorWriteMask red_alpha = { .red, .alpha };
    gpu::ColorTargetState[2] targets = {
        gpu::color_blend_disabled(red_alpha),
        gpu::alpha_blend(),
    };
    gpu::ColorWriteMask green_blue = { .green, .blue };
    targets[1].write_mask = green_blue;
    gpu::ColorState color = { .targets = targets[..] };
    vk::LoweredColorState lowered = vk::lower_color_state(&color)!!;
    assert(lowered.count == 2
        && lowered.blend_enables[0] == vk::FALSE
        && lowered.blend_enables[1] == vk::TRUE,
        "blend enables preserve heterogeneous target order");
    vk::ColorBlendEquationEXT* disabled = &lowered.blend_equations[0];
    assert(disabled.src_color_blend_factor == vk::BlendFactor.ONE
        && disabled.dst_color_blend_factor == vk::BlendFactor.ZERO
        && disabled.color_blend_op == vk::BlendOp.ADD
        && disabled.src_alpha_blend_factor == vk::BlendFactor.ONE
        && disabled.dst_alpha_blend_factor == vk::BlendFactor.ZERO
        && disabled.alpha_blend_op == vk::BlendOp.ADD,
        "disabled blend lowers to the canonical opaque equation");
    vk::ColorBlendEquationEXT* alpha = &lowered.blend_equations[1];
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
    assert(lowered.write_masks[0] == red_alpha_native
        && lowered.write_masks[1] == green_blue_native,
        "write masks preserve heterogeneous target order");
}

fn void zero_target_color_lowering_is_empty() @test {
    gpu::ColorState color;
    vk::LoweredColorState lowered = vk::lower_color_state(&color)!!;
    assert(lowered.count == 0,
        "a depth-only dynamic color domain lowers to no targets");
}
