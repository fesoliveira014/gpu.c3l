module gpu_canonical_gpu_surface;

import gpu;

fn void pin_runtime_adapter_device() {
    gpu::RuntimeDesc runtime_desc = {
    };
    gpu::Runtime runtime = {};
    gpu::AdapterList adapters = {};
    gpu::Adapter adapter = {};
    gpu::QueueRequest queues = {
        .required = {
            .graphics,
            .compute,
            .transfer,
        },
    };
    gpu::DeviceDesc device_desc = { .queues = queues };
    gpu::Device device = {};

    (void)gpu::full_validation_runtime_desc();
    (void)gpu::create_runtime(&runtime_desc);
    (void)gpu::enumerate_adapters(&runtime);
    (void)adapters.get(0);
    (void)gpu::get_adapter_info(&adapter);
    (void)gpu::get_adapter_diagnostics(&adapter);
    (void)gpu::supports_device_desc(&adapter, &device_desc);
    (void)gpu::create_device(&adapter);
    (void)gpu::create_device(&adapter, &device_desc);
    (void)gpu::get_device_caps(&device);
    (void)gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    (void)gpu::get_queue_info(&device, {});
    (void)gpu::get_memory_stats(&device);
    (void)gpu::build_memory_report(&device, false);
    (void)gpu::destroy_device(&device);
    (void)gpu::destroy_runtime(&runtime);
}

fn void pin_memory_and_resources() {
    gpu::Device device = {};
    gpu::AllocationDesc allocation_desc = {
        .size         = 256,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = {
            .graphics = true,
            .compute  = true,
            .transfer = true,
        },
    };
    gpu::GpuAllocation allocation = {};
    gpu::GpuSpan span = {};
    gpu::TextureDesc texture_desc = {
        .width  = 1,
        .height = 1,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
        .access = {
            .graphics = true,
            .compute  = true,
        },
    };
    gpu::TextureHandle texture = {};
    gpu::AttachmentViewHandle attachment_view = {};
    gpu::AttachmentViewDesc attachment_desc = { .texture = texture };
    gpu::TextureViewDesc view_desc = {};
    gpu::TextureView view = {};
    gpu::SamplerDesc sampler_desc = {};
    gpu::TextureViewCreateDesc[1] view_descs = {
        { .texture = texture },
    };
    gpu::TextureView[1] views;

    (void)gpu::allocate_memory(&device, &allocation_desc);
    (void)gpu::get_allocation_info(&device, allocation);
    (void)gpu::get_allocation_span(&device, allocation);
    (void)gpu::get_span_mapping(&device, span);
    (void)gpu::get_span_address(&device, span);
    (void)gpu::flush_mapped_span(&device, span);
    (void)gpu::invalidate_mapped_span(&device, span);
    (void)gpu::get_texture_format_support(&device, texture_desc.format);
    (void)gpu::supports_texture_desc(&device, &texture_desc);
    (void)gpu::get_texture_requirements(&device, &texture_desc);
    (void)gpu::create_texture(&device, &texture_desc);
    (void)gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     0,
    );
    (void)gpu::create_dedicated_texture(
        device:          &device,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
    );
    (void)gpu::create_texture_view(&device, texture, &view_desc);
    (void)gpu::create_texture_views(
        &device,
        view_descs[..],
        views[..],
    );
    (void)gpu::create_attachment_view(&device, &attachment_desc);
    (void)gpu::intern_sampler(&device, &sampler_desc);
    (void)gpu::destroy_attachment_view(&device, attachment_view);
    (void)gpu::destroy_texture_view(&device, view);
    (void)gpu::destroy_texture(&device, texture);
    (void)gpu::free_allocation(&device, &allocation);
}

fn void pin_shaders_and_pipelines() {
    gpu::Device device = {};
    char[4] spirv;
    gpu::ShaderDesc shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE,
        .spirv = spirv[..],
    };
    gpu::ShaderCode shader = {};
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::ComputePipelineDesc compute_desc = {
        .shader     = shader,
        .debug_name = "compute",
    };
    gpu::ComputePipelineDesc[1] compute_descs = { compute_desc };
    gpu::GraphicsPipelineDesc graphics_desc = {
        .vertex_shader   = shader,
        .fragment_shader = shader,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::Format.D32_FLOAT,
        .sample_count    = gpu::SampleCount.ONE,
        .polygon_mode    = gpu::PolygonMode.FILL,
        .debug_name      = "graphics",
    };
    gpu::GraphicsPipelineDesc[1] graphics_descs = { graphics_desc };
    gpu::ColorTargetState[1] color_targets;
    gpu::ColorTargetState color_target =
        gpu::color_blend_disabled();
    gpu::ColorState color_state =
        gpu::uniform_color_state(color_targets[..], color_target);
    gpu::PipelineHandle pipeline = {};
    gpu::PipelineHandle[1] pipelines;
    char[1] pipeline_cache;

    (void)gpu::prepare_shader_code(&shader_desc);
    (void)gpu::create_compute_pipeline(&device, &compute_desc);
    (void)gpu::create_compute_pipelines(
        &device,
        compute_descs[..],
        pipelines[..],
    );
    (void)gpu::create_graphics_pipeline(&device, &graphics_desc);
    (void)gpu::create_graphics_pipelines(
        &device,
        graphics_descs[..],
        pipelines[..],
    );
    color_target = gpu::alpha_blend();
    color_target = gpu::premultiplied_alpha_blend();
    color_target = gpu::additive_blend();
    color_state = gpu::uniform_color_state(color_targets[..], color_target);
    (void)gpu::get_pipeline_cache_size(&device);
    (void)gpu::get_pipeline_cache_data(&device, pipeline_cache[..]);
    (void)gpu::destroy_pipeline(&device, pipeline);
}

fn void pin_commands_and_completion() {
    gpu::Device device = {};
    gpu::Queue queue = {};
    gpu::CommandAllocator allocator = {};
    gpu::CommandAllocatorDesc command_allocator_desc = {};
    gpu::CommandList commands = {};
    gpu::ExecutableCommandList executable = {};
    (void)commands.is_valid();
    (void)executable.is_valid();
    gpu::GpuSpan span = {};
    gpu::BufferCopyDesc copy = {
        .src = span,
        .dst = span,
    };
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
        .hazards = {
            .draw_arguments = true,
            .depth_stencil  = true,
        },
    };
    gpu::TextureHandle texture = {};
    gpu::TextureViewDesc view = {};
    gpu::TextureState transfer_destination = {
        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
        .stages = { .transfer = true },
        .access = { .write = true },
    };
    gpu::TextureState sampled = gpu::sampled_at({
        .vertex_shader   = true,
        .fragment_shader = true,
    });
    gpu::TextureState storage = gpu::storage_at(
        { .compute = true },
        { .read = true, .write = true },
    );
    gpu::TextureBarrier texture_barrier = {};
    gpu::BufferTextureCopyDesc buffer_texture_copy = {
        .src     = span,
        .texture = texture,
    };
    gpu::TextureBufferCopyDesc texture_buffer_copy = {
        .texture = texture,
        .dst     = span,
    };
    gpu::PipelineHandle pipeline = {};
    gpu::AttachmentViewHandle attachment_view = {};
    gpu::DepthState depth = {};
    gpu::DynamicRasterState raster = {};
    gpu::Viewport viewport = {};
    gpu::ScissorRect scissor = {};
    gpu::ColorTargetDesc[1] colors = {
        { .view = attachment_view },
    };
    gpu::RenderPassDesc render_pass = {
        .colors = colors[..],
        .width  = 1,
        .height = 1,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(1, 1) ?? {};
    gpu::ColorTargetState[1] color_targets;
    gpu::ColorState color_state = gpu::uniform_color_state(
        color_targets[..],
        gpu::color_blend_disabled(),
    );
    graphics_state.raster = raster;
    graphics_state.depth = depth;
    graphics_state.color = color_state;
    gpu::SubmitDesc submit = {};
    gpu::CompletionPoint completion = {};
    gpu::GeneratedScratchDesc generated_scratch = {
        .pipeline                = pipeline,
        .kind                    = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list   = 1,
        .preprocess_buffer_count = 1,
    };

    (void)gpu::create_command_allocator(
        &device,
        queue,
        &command_allocator_desc,
    );
    (void)gpu::destroy_command_allocator(&allocator);
    (void)gpu::reserve_generated_scratch(&allocator, &generated_scratch);
    (void)gpu::release_generated_scratch(
        &allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    );
    (void)gpu::begin_commands(&allocator);
    (void)gpu::cmd_copy_buffer(&commands, &copy);
    (void)gpu::cmd_fill_buffer(&commands, span, 0);
    (void)gpu::cmd_copy_buffer_to_texture(&commands, &buffer_texture_copy);
    (void)gpu::cmd_copy_texture_to_buffer(&commands, &texture_buffer_copy);
    (void)gpu::cmd_barrier(&commands, &barrier);
    (void)gpu::texture_transition(
        texture,
        {},
        transfer_destination,
    );
    (void)gpu::texture_view_transition(
        texture: texture,
        view:    view,
        before:  transfer_destination,
        after:   sampled,
    );
    (void)storage;
    (void)gpu::cmd_texture_barrier(&commands, &texture_barrier);
    (void)gpu::cmd_bind_pipeline(&commands, pipeline);
    (void)gpu::cmd_dispatch(
        commands: &commands,
        root:     {},
        groups:   { 1, 1, 1 },
    );
    (void)gpu::cmd_dispatch_indirect(&commands, {}, span);
    (void)gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            span,
        count_span:         span,
        max_dispatch_count: 1,
    );
    (void)gpu::cmd_begin_render_pass(&commands, &render_pass);
    (void)gpu::cmd_begin_render_pass_with_state(
        &commands,
        &render_pass,
        &graphics_state,
    );
    (void)gpu::cmd_set_graphics_state(&commands, &graphics_state);
    (void)gpu::cmd_set_color_state(&commands, &color_state);
    (void)gpu::cmd_set_depth_state(&commands, &depth);
    (void)gpu::cmd_set_raster_state(&commands, &raster);
    (void)gpu::cmd_set_viewport(&commands, &viewport);
    (void)gpu::cmd_set_scissor(&commands, &scissor);
    (void)gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        vertex_count:  3,
        instance_count: 1,
    );
    (void)gpu::cmd_draw_indexed(
        commands:       &commands,
        vertex_root:    {},
        fragment_root:  {},
        index_span:     span,
        index_count:    3,
        instance_count: 1,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
    );
    (void)gpu::cmd_draw_indexed_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
        index_span:    span,
        index_type:    gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indexed_indirect_count(
        commands:       &commands,
        vertex_root:    {},
        fragment_root:  {},
        args:           span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
    );
    (void)gpu::cmd_draw_indexed_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_begin_label(&commands, "canonical");
    (void)gpu::cmd_end_label(&commands);
    (void)gpu::cmd_end_render_pass(&commands);
    (void)gpu::end_commands(&commands);
    (void)gpu::discard_commands(&commands);
    (void)gpu::submit(queue, &submit);
    (void)gpu::poll_completion(completion);
    (void)gpu::wait_completion(completion);
    (void)gpu::discard_executable_commands(&executable);
}

fn void pin_presentation() {
    gpu::Surface surface = {};
    gpu::DeviceDesc device_desc = { .surface = surface };
    gpu::Adapter adapter = {};
    gpu::Device device = {};
    gpu::SwapchainDesc desc = {
        .width  = 1,
        .height = 1,
    };
    gpu::SwapchainHandle swapchain = {};
    gpu::AcquiredImage image = {};
    gpu::TextureState prior_state = image.prior_state;
    gpu::CompletionPoint completion = {};

    (void)gpu::supports_device_desc(&adapter, &device_desc);
    (void)gpu::supports_presentation(&adapter, &surface);
    (void)gpu::create_swapchain(&device, &surface, &desc);
    (void)gpu::get_swapchain_info(&device, swapchain);
    (void)gpu::get_present_mode_support(&device, swapchain);
    (void)gpu::acquire_next_image(&device, swapchain);
    (void)gpu::acquire_next_image(&device, swapchain, 2_000_000);
    (void)gpu::acquire_next_image(
        device:     &device,
        swapchain:  swapchain,
        timeout_ns: gpu::TIMEOUT_INFINITE,
    );
    (void)gpu::present(&device, &image, completion);
    (void)prior_state;
    (void)gpu::resize_swapchain(
        device:    &device,
        swapchain: swapchain,
        width:     1,
        height:    1,
    );
    (void)gpu::destroy_swapchain(&device, swapchain);
    (void)gpu::destroy_surface(&surface);
}
