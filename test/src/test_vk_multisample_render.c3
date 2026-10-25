module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;
import vk;
import std::io;

const char[*] MULTISAMPLE_FETCH_SPIRV =
    $embed("shaders/multisample_fetch.comp.spv");

const char[*] SAMPLE_COVERAGE_FRAG_SPIRV =
    $embed("shaders/sample_coverage.frag.spv");

const uint MS_FETCH_PACKED_COLOR = 0;
const uint MS_FETCH_DEPTH_BITS = 1;
const uint MS_FETCH_UNSIGNED = 2;
const uint MS_FETCH_SINGLE_SAMPLE_DEPTH_BITS = 3;
const uint MS_FETCH_GROUP_SIZE = 8; // local_size of multisample_fetch.comp.glsl
const uint MS_DRAW_PACKED = 0xFFC08040;
const uint MS_CLEAR_PACKED = 0xFF201008;
const uint MS_UNSIGNED_CLEAR = 0x13579BDF;

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
    gpu::QueueRoles access = { .graphics },
) {
    gpu::TextureDesc desc = {
        .access       = access,
        .width        = DP_RT,
        .height       = DP_RT,
        .format       = format,
        .usage        = usage,
        .sample_count = sample_count,
        .debug_name   = debug_name,
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void rendering_attachments_use_classic_layouts() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle color = make_multisample_target(
        device:       &device,
        format:       gpu::Format.RGBA8_UNORM,
        usage:        { .color_attach },
        sample_count: gpu::SampleCount.ONE,
        debug_name:   "layout_policy_color",
    );
    defer gpu::destroy_texture(&device, color)!!;
    gpu::AttachmentViewHandle color_view =
        create_test_attachment_view(&device, color)!!;
    defer destroy_test_attachment_view(&device, &color_view)!!;
    gpu::TextureHandle depth = make_multisample_target(
        device:       &device,
        format:       gpu::Format.D32_FLOAT,
        usage:        { .depth_attach },
        sample_count: gpu::SampleCount.ONE,
        debug_name:   "layout_policy_depth",
    );
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view =
        create_test_attachment_view(&device, depth)!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ColorTargetDesc[1] colors = {{ .view = color_view }};
    gpu::DepthTargetDesc depth_target = { .view = depth_view };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .depth  = &depth_target,
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::internal::vk::CommandRecord* record =
        gpu::internal::recording_command(&commands);

    gpu::internal::vk::PreparedRenderPass prepared =
        gpu::internal::vk::lower_render_pass(record, &pass)!!;
    if (gpu::get_device_caps(&device).unified_layouts) {
        assert(prepared.attachments[0].image_layout == vk::ImageLayout.GENERAL
            && prepared.depth_attachment.image_layout == vk::ImageLayout.GENERAL,
            "unified mode renders in GENERAL");
        return;
    }
    assert(prepared.attachments[0].image_layout
            == vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL
        && prepared.depth_attachment.image_layout
            == vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        "rendering uses distinct classic color and depth layouts");
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
    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .color_formats   = pipeline_color_formats[..],
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

    gpu::Queue command_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    );
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        &device,
        command_queue,
    )!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
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
    gpu::internal::vk::CommandRecord* record =
        gpu::internal::recording_command(&commands);
    gpu::internal::vk::PreparedRenderPass prepared =
        gpu::internal::vk::lower_render_pass(record, &pass)!!;
    bool unified = gpu::get_device_caps(&device).unified_layouts;
    assert(prepared.attachments[0].resolve_image_layout
            == (unified ? vk::ImageLayout.GENERAL : vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL),
        "resolve uses the classic color-attachment layout");

    gpu::DepthState depth_state = {
        .test_enable  = true,
        .write_enable = true,
        .compare      = gpu::CompareOp.LESS,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = depth_state;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
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

fn uint sample_count_value(gpu::SampleCount sample_count) {
    return (uint)1 << (uint)sample_count.ordinal;
}

fn gpu::SampleCount sampled_multisample_count(
    gpu::Device* device,
    gpu::Format format,
    gpu::TextureUsage usage,
) {
    gpu::SampleCount[3] candidates = {
        gpu::SampleCount.FOUR,
        gpu::SampleCount.TWO,
        gpu::SampleCount.EIGHT,
    };
    foreach (candidate : candidates) {
        gpu::TextureDesc desc = {
            .access       = { .graphics, .compute },
            .width        = DP_RT,
            .height       = DP_RT,
            .format       = format,
            .usage        = usage,
            .sample_count = candidate,
        };
        if (gpu::supports_texture_desc(device, &desc)!!) return candidate;
    }

    return gpu::SampleCount.ONE;
}

struct SampleReadback {
    OwnedSpan root;
    OwnedSpan values;
    uint      samples;
}

fn SampleReadback make_sample_readback(
    gpu::Device* device,
    gpu::TextureIndex texture_index,
    uint samples,
    uint mode,
) {
    SampleReadback readback = {
        .root    = allocate_owned_span(
            device:       device,
            size:         MultisampleFetchRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "multisample_fetch_root",
        )!!,
        .values  = make_readback(device, (usz)DP_RT * DP_RT * samples * 4),
        .samples = samples,
    };
    MultisampleFetchRoot* root =
        (MultisampleFetchRoot*)test_span_mapping(device, readback.root.span).ptr;
    *root = {
        .out_gpu       = test_span_address(device, readback.values.span),
        .texture_index = texture_index,
        .width         = DP_RT,
        .height        = DP_RT,
        .samples       = samples,
        .mode          = mode,
    };
    gpu::flush_mapped_span(device, readback.root.span)!!;
    return readback;
}

fn void free_sample_readback(gpu::Device* device, SampleReadback* readback) {
    free_owned_span(device, &readback.values)!!;
    free_owned_span(device, &readback.root)!!;
}

fn void record_sample_fetch(
    gpu::Device* device,
    gpu::CommandList* commands,
    SampleReadback* readback,
) {
    gpu::cmd_dispatch(
        commands: commands,
        root:     test_span_address(device, readback.root.span),
        groups:   {
            DP_RT / MS_FETCH_GROUP_SIZE,
            DP_RT / MS_FETCH_GROUP_SIZE,
            1,
        },
    )!!;
}

fn uint* mapped_sample_values(gpu::Device* device, SampleReadback* readback) {
    gpu::invalidate_mapped_span(device, readback.values.span)!!;
    return (uint*)test_span_mapping(device, readback.values.span).ptr;
}

fn uint texel_sample_count(
    uint* values,
    uint samples,
    uint x,
    uint y,
    uint expected,
) {
    uint matches;
    for (uint sample_index = 0; sample_index < samples; sample_index++) {
        if (values[(y * DP_RT + x) * samples + sample_index] == expected) {
            matches++;
        }
    }

    return matches;
}

fn void record_to_sampled(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
) {
    gpu::TextureBarrier to_sampled = gpu::texture_transition(
        texture: texture,
        before:  {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        after:   gpu::sampled_at({ .compute }),
    )!!;
    gpu::cmd_texture_barrier(commands, &to_sampled)!!;
}

fn void multisample_texture_reads_each_sample() @test {
    VulkanValidationCapture validation;
    gpu::Device device = make_offscreen_test_device(&validation);
    defer destroy_test_device(&device)!!;

    validation.errors = 0;
    gpu::SampleCount sample_count = sampled_multisample_count(
        &device,
        gpu::Format.RGBA8_UNORM,
        { .color_attach, .sampled },
    );
    if (sample_count == gpu::SampleCount.ONE) {
        io::eprintfn("sampled multisample color targets are unsupported on this adapter; skipping");
        return;
    }

    uint samples = sample_count_value(sample_count);
    gpu::SampleCount unsigned_count = sampled_multisample_count(
        &device,
        gpu::Format.R32_UINT,
        { .color_attach, .sampled },
    );
    bool with_unsigned = unsigned_count == sample_count;
    if (!with_unsigned) {
        io::eprintfn("R32_UINT sampled multisample targets differ in count on this adapter; skipping them");
    }

    gpu::TextureHandle color = make_multisample_target(
        device:       &device,
        format:       gpu::Format.RGBA8_UNORM,
        usage:        { .color_attach, .sampled },
        sample_count: sample_count,
        debug_name:   "multisample_sampled_color",
        access:       { .graphics, .compute },
    );
    defer gpu::destroy_texture(&device, color)!!;
    gpu::TextureView color_sampled = gpu::create_texture_view(&device, color, null)!!;
    defer gpu::destroy_texture_view(&device, color_sampled);
    gpu::AttachmentViewHandle color_view = create_test_attachment_view(&device, color)!!;
    defer destroy_test_attachment_view(&device, &color_view)!!;

    gpu::TextureHandle unsigned_texture;
    gpu::TextureView unsigned_sampled;
    gpu::AttachmentViewHandle unsigned_view;
    if (with_unsigned) {
        unsigned_texture = make_multisample_target(
            device:       &device,
            format:       gpu::Format.R32_UINT,
            usage:        { .color_attach, .sampled },
            sample_count: sample_count,
            debug_name:   "multisample_sampled_unsigned",
            access:       { .graphics, .compute },
        );
        unsigned_sampled = gpu::create_texture_view(&device, unsigned_texture, null)!!;
        unsigned_view = create_test_attachment_view(&device, unsigned_texture)!!;
    }

    defer {
        if (with_unsigned) {
            destroy_test_attachment_view(&device, &unsigned_view)!!;
            gpu::destroy_texture_view(&device, unsigned_sampled);
            gpu::destroy_texture(&device, unsigned_texture)!!;
        }
    }

    DepthCtx shaders = make_depth_ctx();
    gpu::Format[1] pipeline_color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .color_formats   = pipeline_color_formats[..],
        .sample_count    = sample_count,
        .debug_name      = "multisample_sampled_color",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::ShaderDesc fetch_shader = {
        .spirv       = MULTISAMPLE_FETCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc fetch_desc = { .shader = fetch_shader };
    gpu::PipelineHandle fetch = gpu::create_compute_pipeline(&device, &fetch_desc)!!;
    defer gpu::destroy_pipeline(&device, fetch)!!;

    OwnedSpan vertices = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &vertices)!!;
    OwnedSpan draw_color = write_color(
        &device,
        { 64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f },
    );
    defer free_owned_span(&device, &draw_color)!!;
    SampleReadback color_values = make_sample_readback(
        &device,
        color_sampled.index,
        samples,
        MS_FETCH_PACKED_COLOR,
    );
    defer free_sample_readback(&device, &color_values);
    SampleReadback unsigned_values;
    if (with_unsigned) {
        unsigned_values = make_sample_readback(
            &device,
            unsigned_sampled.index,
            samples,
            MS_FETCH_UNSIGNED,
        );
    }

    defer {
        if (with_unsigned) free_sample_readback(&device, &unsigned_values);
    }

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    record_color_to_attach(&commands, color);
    if (with_unsigned) record_color_to_attach(&commands, unsigned_texture);

    gpu::ColorTargetDesc[1] colors = {{
        .view     = color_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::GraphicsState graphics_state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_targets = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(&device, vertices.span),
        fragment_root:  test_span_address(&device, draw_color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    if (with_unsigned) {
        gpu::ColorTargetDesc[1] unsigned_colors = {{
            .view     = unsigned_view,
            .load_op  = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear    = { .uint_rgba = { MS_UNSIGNED_CLEAR, 0, 0, 0 } },
        }};
        gpu::RenderPassDesc unsigned_pass = {
            .colors = unsigned_colors[..],
            .width  = DP_RT,
            .height = DP_RT,
        };
        gpu::cmd_begin_render_pass(&commands, &unsigned_pass)!!;
        gpu::cmd_end_render_pass(&commands)!!;
    }

    record_to_sampled(&commands, color);
    if (with_unsigned) record_to_sampled(&commands, unsigned_texture);

    bind_test_compute_pipeline(&commands, fetch);
    record_sample_fetch(&device, &commands, &color_values);
    if (with_unsigned) record_sample_fetch(&device, &commands, &unsigned_values);

    gpu::Barrier host_read = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&commands, &host_read)!!;
    submit_and_finish(&device, &commands);

    uint* values = mapped_sample_values(&device, &color_values);
    uint center = DP_RT / 2;
    assert(texel_sample_count(values, samples, center, center, MS_DRAW_PACKED) == samples,
        "an interior texel holds the draw color at every sample");
    assert(texel_sample_count(values, samples, 0, 0, MS_CLEAR_PACKED) == samples,
        "an exterior texel holds the clear color at every sample");

    uint edge_texels;
    for (uint y = 0; y < DP_RT; y++) {
        for (uint x = 0; x < DP_RT; x++) {
            uint draw = texel_sample_count(values, samples, x, y, MS_DRAW_PACKED);
            uint clear = texel_sample_count(values, samples, x, y, MS_CLEAR_PACKED);
            assert(draw + clear == samples,
                "every sample holds the draw color or the clear color, never a blend");
            if (draw > 0 && clear > 0) edge_texels++;
        }
    }

    assert(edge_texels > 0, "a triangle edge splits some texels between the two colors");

    if (with_unsigned) {
        uint* unsigned_samples = mapped_sample_values(&device, &unsigned_values);
        for (uint y = 0; y < DP_RT; y++) {
            for (uint x = 0; x < DP_RT; x++) {
                assert(texel_sample_count(unsigned_samples, samples, x, y, MS_UNSIGNED_CLEAR) == samples,
                    "every sample of an unsigned target holds the clear value");
            }
        }
    }

    assert(validation.errors == 0, "sampling multisample targets is validation-clean");
}

struct DepthResolveTargets {
    gpu::TextureHandle        source;
    gpu::TextureView          source_sampled;
    gpu::AttachmentViewHandle source_view;
    gpu::TextureHandle        target;
    gpu::TextureView          target_sampled;
    gpu::AttachmentViewHandle target_view;
}

fn DepthResolveTargets make_depth_resolve_targets(
    gpu::Device* device,
    gpu::SampleCount sample_count,
) {
    DepthResolveTargets targets;
    targets.source = make_multisample_target(
        device:       device,
        format:       gpu::Format.D32_FLOAT,
        usage:        { .depth_attach, .sampled },
        sample_count: sample_count,
        debug_name:   "depth_resolve_source",
        access:       { .graphics, .compute },
    );
    targets.source_sampled = gpu::create_texture_view(device, targets.source, null)!!;
    targets.source_view = create_test_attachment_view(device, targets.source)!!;
    targets.target = make_multisample_target(
        device:       device,
        format:       gpu::Format.D32_FLOAT,
        usage:        { .depth_attach, .sampled },
        sample_count: gpu::SampleCount.ONE,
        debug_name:   "depth_resolve_target",
        access:       { .graphics, .compute },
    );
    targets.target_sampled = gpu::create_texture_view(device, targets.target, null)!!;
    targets.target_view = create_test_attachment_view(device, targets.target)!!;
    return targets;
}

fn void destroy_depth_resolve_targets(
    gpu::Device* device,
    DepthResolveTargets* targets,
) {
    destroy_test_attachment_view(device, &targets.target_view)!!;
    gpu::destroy_texture_view(device, targets.target_sampled);
    gpu::destroy_texture(device, targets.target)!!;
    destroy_test_attachment_view(device, &targets.source_view)!!;
    gpu::destroy_texture_view(device, targets.source_sampled);
    gpu::destroy_texture(device, targets.source)!!;
}

fn gpu::TextureState depth_resolve_state() {
    return {
        .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
        .stages = { .depth_output, .color_output },
        .access = { .read, .write },
    };
}

fn void record_depth_resolve_attach(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
) {
    gpu::TextureBarrier to_attach = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   depth_resolve_state(),
    )!!;
    gpu::cmd_texture_barrier(commands, &to_attach)!!;
}

fn void record_depth_resolve_order(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
) {
    gpu::TextureBarrier order = gpu::texture_transition(
        texture: texture,
        before:  depth_resolve_state(),
        after:   depth_resolve_state(),
    )!!;
    gpu::cmd_texture_barrier(commands, &order)!!;
}

fn void record_depth_resolve_to_sampled(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
) {
    gpu::TextureBarrier to_sampled = gpu::texture_transition(
        texture: texture,
        before:  depth_resolve_state(),
        after:   gpu::sampled_at({ .compute }),
    )!!;
    gpu::cmd_texture_barrier(commands, &to_sampled)!!;
}

fn void record_depth_resolve_pass(
    gpu::CommandList* commands,
    DepthResolveTargets* targets,
    gpu::LoadOp load_op,
    gpu::DepthResolveMode mode,
) {
    gpu::DepthTargetDesc depth_target = {
        .view         = targets.source_view,
        .load_op      = load_op,
        .store_op     = gpu::StoreOp.STORE,
        .clear        = { .depth = 1.0f },
        .resolve_view = targets.target_view,
        .resolve_mode = mode,
    };
    gpu::RenderPassDesc pass = {
        .depth  = &depth_target,
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::cmd_begin_render_pass(commands, &pass)!!;
}

fn float selected_depth(
    gpu::DepthResolveMode mode,
    float first,
    float lowest,
    float highest,
) {
    switch (mode) {
        case gpu::DepthResolveMode.MIN: return lowest;
        case gpu::DepthResolveMode.MAX: return highest;
        default:                        return first;
    }
}

fn void check_depth_resolve(
    gpu::Device* device,
    gpu::SampleCount sample_count,
    gpu::DepthResolveMode mode,
) {
    uint samples = sample_count_value(sample_count);
    DepthResolveTargets targets = make_depth_resolve_targets(device, sample_count);
    defer destroy_depth_resolve_targets(device, &targets);

    DepthCtx shaders = make_depth_ctx();
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .depth_format    = gpu::Format.D32_FLOAT,
        .sample_count    = sample_count,
        .debug_name      = "depth_resolve",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(device, pipeline)!!;
    gpu::ShaderDesc fetch_shader = {
        .spirv       = MULTISAMPLE_FETCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc fetch_desc = { .shader = fetch_shader };
    gpu::PipelineHandle fetch = gpu::create_compute_pipeline(device, &fetch_desc)!!;
    defer gpu::destroy_pipeline(device, fetch)!!;

    OwnedSpan far_vertices = write_tri(device, FAR_Z);
    defer free_owned_span(device, &far_vertices)!!;
    OwnedSpan near_vertices = write_tri(device, NEAR_Z);
    defer free_owned_span(device, &near_vertices)!!;
    OwnedSpan unused_color = write_color(device, { 0.0f, 0.0f, 0.0f, 0.0f });
    defer free_owned_span(device, &unused_color)!!;
    SampleReadback source_values = make_sample_readback(
        device,
        targets.source_sampled.index,
        samples,
        MS_FETCH_DEPTH_BITS,
    );
    defer free_sample_readback(device, &source_values);
    SampleReadback target_values = make_sample_readback(
        device,
        targets.target_sampled.index,
        1,
        MS_FETCH_SINGLE_SAMPLE_DEPTH_BITS,
    );
    defer free_sample_readback(device, &target_values);

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    record_depth_resolve_attach(&commands, targets.source);
    record_depth_resolve_attach(&commands, targets.target);

    gpu::GraphicsState graphics_state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = {
        .test_enable  = true,
        .write_enable = true,
        .compare      = gpu::CompareOp.LESS,
    };
    record_depth_resolve_pass(&commands, &targets, gpu::LoadOp.CLEAR, mode);
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(device, far_vertices.span),
        fragment_root:  test_span_address(device, unused_color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    record_depth_resolve_order(&commands, targets.source);
    record_depth_resolve_order(&commands, targets.target);

    record_depth_resolve_pass(&commands, &targets, gpu::LoadOp.LOAD, mode);
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(device, near_vertices.span),
        fragment_root:  test_span_address(device, unused_color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    record_depth_resolve_to_sampled(&commands, targets.source);
    record_depth_resolve_to_sampled(&commands, targets.target);

    bind_test_compute_pipeline(&commands, fetch);
    record_sample_fetch(device, &commands, &source_values);
    record_sample_fetch(device, &commands, &target_values);

    gpu::Barrier host_read = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&commands, &host_read)!!;
    submit_and_finish(device, &commands);

    uint* source_bits = mapped_sample_values(device, &source_values);
    uint* resolved_bits = mapped_sample_values(device, &target_values);
    uint edge_texels;
    for (uint texel = 0; texel < DP_RT * DP_RT; texel++) {
        float first = bitcast(source_bits[texel * samples], float);
        float lowest = first;
        float highest = first;
        for (uint sample_index = 1; sample_index < samples; sample_index++) {
            float depth = bitcast(source_bits[texel * samples + sample_index], float);
            if (depth < lowest) lowest = depth;
            if (depth > highest) highest = depth;
        }

        assert(bitcast(resolved_bits[texel], float) == selected_depth(mode, first, lowest, highest),
            "the resolve writes the selected sample value of every texel");
        assert((lowest == NEAR_Z || lowest == 1.0f) && (highest == NEAR_Z || highest == 1.0f),
            "every sample holds the drawn depth or the clear depth");
        if (lowest != highest) edge_texels++;
    }

    float center = bitcast(resolved_bits[DP_RT / 2 * DP_RT + DP_RT / 2], float);
    float corner = bitcast(resolved_bits[0], float);
    assert(center == NEAR_Z && corner == 1.0f,
        "interior texels resolve to the drawn depth and exterior texels to the clear depth");
    assert(edge_texels > 0, "a triangle edge splits some texels between the two depths");
}

fn fault begin_depth_resolve_fault(
    gpu::Device* device,
    gpu::AttachmentViewHandle source_view,
    gpu::AttachmentViewHandle resolve_view,
    gpu::DepthResolveMode mode,
) {
    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::DepthTargetDesc depth_target = {
        .view         = source_view,
        .load_op      = gpu::LoadOp.CLEAR,
        .store_op     = gpu::StoreOp.STORE,
        .resolve_view = resolve_view,
        .resolve_mode = mode,
    };
    gpu::RenderPassDesc pass = {
        .depth  = &depth_target,
        .width  = DP_RT,
        .height = DP_RT,
    };
    return @catch(gpu::cmd_begin_render_pass(&commands, &pass));
}

fn void depth_resolve_writes_the_selected_sample() @test {
    VulkanValidationCapture validation;
    gpu::Device device = make_offscreen_test_device(&validation);
    defer destroy_test_device(&device)!!;

    validation.errors = 0;
    gpu::SampleCount sample_count = sampled_multisample_count(
        &device,
        gpu::Format.D32_FLOAT,
        { .depth_attach, .sampled },
    );
    if (sample_count == gpu::SampleCount.ONE) {
        io::eprintfn("sampled multisample depth targets are unsupported on this adapter; skipping");
        return;
    }

    gpu::DepthResolveSupport supported = gpu::get_device_caps(&device).depth_resolve_modes;
    assert(supported.sample_zero, "every device resolves depth with sample zero");
    check_depth_resolve(&device, sample_count, gpu::DepthResolveMode.SAMPLE_ZERO);
    if (supported.min) check_depth_resolve(&device, sample_count, gpu::DepthResolveMode.MIN);
    if (supported.max) check_depth_resolve(&device, sample_count, gpu::DepthResolveMode.MAX);

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    state.caps.depth_resolve_modes = { .sample_zero = true };
    DepthResolveTargets targets = make_depth_resolve_targets(&device, sample_count);
    defer destroy_depth_resolve_targets(&device, &targets);

    assert(begin_depth_resolve_fault(&device, targets.source_view, targets.target_view, gpu::DepthResolveMode.MIN)
            == gpu::UNSUPPORTED_FEATURE,
        "an unreported MIN resolve faults UNSUPPORTED_FEATURE");
    assert(begin_depth_resolve_fault(&device, targets.source_view, targets.target_view, gpu::DepthResolveMode.MAX)
            == gpu::UNSUPPORTED_FEATURE,
        "an unreported MAX resolve faults UNSUPPORTED_FEATURE");
    state.caps.depth_resolve_modes = supported;

    gpu::Format stencil_format = gpu::get_device_caps(&device).depth_stencil_format;
    if (stencil_format == gpu::Format.UNDEFINED) {
        io::eprintfn("no combined depth-stencil format on this adapter; skipping the stencil resolve fault");
    } else {
        gpu::TextureHandle stencil_source = make_multisample_target(
            device:       &device,
            format:       stencil_format,
            usage:        { .depth_attach },
            sample_count: gpu::SampleCount.ONE,
            debug_name:   "depth_resolve_stencil_source",
        );
        defer gpu::destroy_texture(&device, stencil_source)!!;
        gpu::AttachmentViewHandle stencil_source_view = create_test_attachment_view(&device, stencil_source)!!;
        defer destroy_test_attachment_view(&device, &stencil_source_view)!!;
        gpu::TextureHandle stencil_target = make_multisample_target(
            device:       &device,
            format:       stencil_format,
            usage:        { .depth_attach },
            sample_count: gpu::SampleCount.ONE,
            debug_name:   "depth_resolve_stencil_target",
        );
        defer gpu::destroy_texture(&device, stencil_target)!!;
        gpu::AttachmentViewHandle stencil_target_view = create_test_attachment_view(&device, stencil_target)!!;
        defer destroy_test_attachment_view(&device, &stencil_target_view)!!;

        assert(begin_depth_resolve_fault(
                &device,
                stencil_source_view,
                stencil_target_view,
                gpu::DepthResolveMode.SAMPLE_ZERO,
            ) == gpu::INVALID_ARGUMENT,
            "a depth resolve view on a format with a stencil aspect faults INVALID_ARGUMENT");
    }

    assert(validation.errors == 0, "resolving multisample depth is validation-clean");
}

fn void check_sample_coverage(
    gpu::Device* device,
    gpu::SampleCount sample_count,
    float min_sample_shading,
) {
    uint samples = sample_count_value(sample_count);
    gpu::TextureHandle color = make_multisample_target(
        device:       device,
        format:       gpu::Format.RGBA8_UNORM,
        usage:        { .color_attach, .sampled },
        sample_count: sample_count,
        debug_name:   "sample_coverage_color",
        access:       { .graphics, .compute },
    );
    defer gpu::destroy_texture(device, color)!!;
    gpu::TextureView color_sampled = gpu::create_texture_view(device, color, null)!!;
    defer gpu::destroy_texture_view(device, color_sampled);
    gpu::AttachmentViewHandle color_view = create_test_attachment_view(device, color)!!;
    defer destroy_test_attachment_view(device, &color_view)!!;

    DepthCtx shaders = make_depth_ctx();
    gpu::ShaderDesc coverage_shader = {
        .spirv       = SAMPLE_COVERAGE_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::Format[1] pipeline_color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader      = shaders.vert,
        .fragment_shader    = coverage_shader,
        .color_formats      = pipeline_color_formats[..],
        .sample_count       = sample_count,
        .min_sample_shading = min_sample_shading,
        .debug_name         = "sample_coverage",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(device, pipeline)!!;
    gpu::ShaderDesc fetch_shader = {
        .spirv       = MULTISAMPLE_FETCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc fetch_desc = { .shader = fetch_shader };
    gpu::PipelineHandle fetch = gpu::create_compute_pipeline(device, &fetch_desc)!!;
    defer gpu::destroy_pipeline(device, fetch)!!;

    OwnedSpan vertices = write_tri(device, NEAR_Z);
    defer free_owned_span(device, &vertices)!!;
    OwnedSpan unused_color = write_color(device, { 0.0f, 0.0f, 0.0f, 0.0f });
    defer free_owned_span(device, &unused_color)!!;
    SampleReadback values = make_sample_readback(
        device,
        color_sampled.index,
        samples,
        MS_FETCH_PACKED_COLOR,
    );
    defer free_sample_readback(device, &values);

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    record_color_to_attach(&commands, color);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = color_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::GraphicsState graphics_state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_targets = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(device, vertices.span),
        fragment_root:  test_span_address(device, unused_color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    record_to_sampled(&commands, color);
    bind_test_compute_pipeline(&commands, fetch);
    record_sample_fetch(device, &commands, &values);

    gpu::Barrier host_read = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&commands, &host_read)!!;
    submit_and_finish(device, &commands);

    uint* invocation_coverage = mapped_sample_values(device, &values);
    uint edge_texels;
    for (uint y = 0; y < DP_RT; y++) {
        for (uint x = 0; x < DP_RT; x++) {
            uint covered;
            for (uint sample_index = 0; sample_index < samples; sample_index++) {
                if (invocation_coverage[(y * DP_RT + x) * samples + sample_index] != 0) covered++;
            }

            for (uint sample_index = 0; sample_index < samples; sample_index++) {
                uint value = invocation_coverage[(y * DP_RT + x) * samples + sample_index];
                if (value == 0) continue;

                uint expected = min_sample_shading > 0.0f ? 1 : covered;
                assert(value == (0xFF000000 | expected),
                    "a shaded sample records the samples its invocation covered");
            }

            if (covered > 0 && covered < samples) edge_texels++;
        }
    }

    uint center = DP_RT / 2;
    uint center_covered = texel_sample_count(
        invocation_coverage,
        samples,
        center,
        center,
        0xFF000000 | (min_sample_shading > 0.0f ? 1 : samples),
    );
    assert(center_covered == samples, "an interior texel is covered at every sample");
    assert(texel_sample_count(invocation_coverage, samples, 0, 0, 0) == samples,
        "an exterior texel is not shaded");
    assert(edge_texels > 0, "a triangle edge covers some texels partly");
}
