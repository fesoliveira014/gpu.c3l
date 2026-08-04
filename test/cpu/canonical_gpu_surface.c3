module gpu_canonical_gpu_surface;

import gpu;

fn void pin_runtime_adapter_device() {
    gpu::RuntimeDesc runtime_desc = {
        .acceleration_structure_heap_capacity = 16,
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
    gpu::DeviceDesc device_desc = {
        .queues                 = queues,
        .enable_sparse_textures = true,
        .enable_ray_queries     = true,
        .enable_ray_tracing_pipelines = true,
    };
    gpu::Device device = {};
    gpu::DeviceCaps caps = {
        .presentation_enabled         = true,
        .draw_indirect_count          = true,
        .generated_work               = true,
        .async_compute                = true,
        .queues                       = { .graphics, .compute, .transfer },
        .line_polygon_mode            = true,
        .timestamps                   = {
            .queues              = { .graphics, .compute, .transfer },
            .graphics_valid_bits = 64,
            .compute_valid_bits  = 64,
            .transfer_valid_bits = 64,
            .period_ns           = 1.0f,
        },
        .acceleration_structures       = {
            .enabled             = true,
            .heap_capacity       = 16,
            .max_geometry_count  = 8,
            .max_primitive_count = 1024,
            .max_instance_count  = 128,
            .scratch_alignment   = 256,
        },
        .ray_queries                  = { .enabled = true },
        .ray_tracing_pipelines        = {
            .enabled                           = true,
            .max_recursion_depth               = 2,
            .max_ray_dispatch_invocation_count = 1024,
            .max_ray_dispatch_dimensions       = { 64, 32, 16 },
            .shader_group_handle_size          = 32,
            .shader_group_handle_alignment     = 32,
            .shader_group_base_alignment       = 64,
            .max_shader_group_stride           = 4096,
            .max_hit_attribute_size            = 32,
        },
        .sparse_textures              = {
            .image_2d           = true,
            .image_3d           = true,
            .nonresident_strict = true,
            .binding_queues     = { .graphics, .compute },
        },
        .texture_heap_capacity        = 1,
        .sampler_heap_capacity        = 1,
        .max_color_attachments        = 1,
        .max_push_constant_size       = 1,
        .max_compute_work_group_count = { .x = 1, .y = 1, .z = 1 },
        .max_draw_indirect_count      = 1,
        .max_generated_work_count     = 1,
        .min_uniform_alignment        = 1,
        .min_storage_alignment        = 1,
        .min_texel_buffer_alignment   = 1,
        .max_sampler_lod_bias         = 1.0f,
        .max_sampler_anisotropy       = 1.0f,
    };

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
    (void)caps.presentation_enabled;
    (void)caps.draw_indirect_count;
    (void)caps.generated_work;
    (void)caps.async_compute;
    (void)caps.queues;
    (void)caps.line_polygon_mode;
    (void)caps.timestamps;
    (void)caps.acceleration_structures;
    (void)caps.ray_queries;
    (void)caps.ray_tracing_pipelines;
    (void)caps.sparse_textures;
    (void)caps.sparse_textures.image_2d;
    (void)caps.sparse_textures.image_3d;
    (void)caps.sparse_textures.nonresident_strict;
    (void)caps.sparse_textures.binding_queues;
    (void)caps.texture_heap_capacity;
    (void)caps.sampler_heap_capacity;
    (void)caps.max_color_attachments;
    (void)caps.max_push_constant_size;
    (void)caps.max_compute_work_group_count;
    (void)caps.max_draw_indirect_count;
    (void)caps.max_generated_work_count;
    (void)caps.min_uniform_alignment;
    (void)caps.min_storage_alignment;
    (void)caps.min_texel_buffer_alignment;
    (void)caps.max_sampler_lod_bias;
    (void)caps.max_sampler_anisotropy;
}

fn void pin_ray_tracing_pipeline_values() {
    gpu::ShaderDesc raygen = {};
    gpu::ShaderDesc closest_hit = {};
    gpu::RayTracingHitGroupDesc[1] hit_groups = {
        {
            .kind               = gpu::RayTracingHitGroupKind.TRIANGLES,
            .closest_hit_shader = &closest_hit,
        },
    };
    gpu::RayTracingPipelineDesc pipeline_desc = {
        .ray_generation_shaders = (&raygen)[:1],
        .hit_groups             = hit_groups[..],
        .max_recursion_depth    = 2,
        .debug_name             = "canonical_ray_tracing_pipeline",
    };
    gpu::RayTracingPipelineInfo info = {
        .ray_generation = { .first_group = 0, .group_count = 1 },
        .hit            = { .first_group = 1, .group_count = 1 },
        .total_group_count = 2,
    };
    gpu::RayQueryCaps ray_query_caps = { .enabled = true };
    gpu::RayTracingShaderGroupRange ray_generation_range = {
        .first_group = 0,
        .group_count = 1,
    };
    gpu::RayTracingShaderBindingTable table = {
        .ray_generation = { .stride = 32 },
        .hit            = { .stride = 32 },
    };
    gpu::StageMask stage = { .ray_tracing = true };

    (void)pipeline_desc;
    (void)ray_query_caps;
    (void)ray_generation_range;
    (void)info;
    (void)table;
    (void)stage;
}

fn void pin_acceleration_structures() {
    gpu::Device device = {};
    gpu::GpuAllocation allocation = {};
    gpu::CommandList commands = {};
    gpu::AccelerationStructureGeometryDesc[1] geometry = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count = 3,
                .max_primitive_count = 1,
                .index_type = gpu::AccelerationStructureIndexType.NONE,
                .has_transform = true,
            },
        },
    };
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = geometry[..],
    };
    gpu::AccelerationStructureHandle acceleration_structure = {};
    gpu::AccelerationStructureInstanceDesc instance_desc = {
        .bottom_level = acceleration_structure,
    };
    gpu::AccelerationStructureBuildDesc build_desc = {
        .destination = acceleration_structure,
    };
    gpu::AccelerationStructureView view = {};
    gpu::AccelerationStructureTriangleGeometryDesc triangle_geometry = {};
    gpu::AccelerationStructureAabbGeometryDesc aabb_geometry = {};
    gpu::AccelerationStructureTriangleBuildDesc triangle_build = {};
    gpu::AccelerationStructureAabbBuildDesc aabb_build = {};
    gpu::AccelerationStructureBuildFlags build_flags = {
        .prefer_fast_build = true,
        .minimize_memory = true,
    };
    gpu::AccelerationStructureGeometryFlags geometry_flags = {
        .opaque = true,
    };

    (void)triangle_geometry.has_transform;
    (void)aabb_geometry.max_primitive_count;
    (void)triangle_build.transform;
    (void)aabb_build.stride;
    (void)build_flags.prefer_fast_build;
    (void)build_flags.minimize_memory;
    (void)geometry_flags.opaque;

    (void)gpu::get_acceleration_structure_requirements(&device, &desc);
    (void)gpu::create_acceleration_structure(&device, &desc);
    (void)gpu::create_placed_acceleration_structure(
        device:     &device,
        desc:       &desc,
        allocation: allocation,
        offset:     0,
    );
    (void)gpu::create_dedicated_acceleration_structure(&device, &desc);
    (void)gpu::destroy_acceleration_structure(
        &device,
        acceleration_structure,
    );
    (void)gpu::get_acceleration_structure_address(
        &device,
        acceleration_structure,
    );
    (void)gpu::make_acceleration_structure_instance(&device, &instance_desc);
    (void)gpu::create_acceleration_structure_view(
        &device,
        acceleration_structure,
    );
    (void)gpu::destroy_acceleration_structure_view(&device, view);
    (void)gpu::cmd_build_acceleration_structure(&commands, &build_desc);
    (void)gpu::cmd_update_acceleration_structure(&commands, &build_desc);
    (void)gpu::cmd_clone_acceleration_structure(
        &commands,
        acceleration_structure,
        acceleration_structure,
    );
}

fn void pin_timestamps() {
    gpu::Device device = {};
    gpu::CommandList commands = {};
    gpu::TimestampPoolDesc desc = {
        .capacity   = 2,
        .debug_name = "canonical_timestamp_pool",
    };
    gpu::TimestampPoolHandle pool = gpu::TIMESTAMP_POOL_HANDLE_INVALID;
    gpu::TimestampCaps caps = {
        .queues              = { .compute = true },
        .compute_valid_bits  = 64,
        .period_ns           = 1.0f,
    };
    gpu::GpuSpan dst = {};
    ulong[2] values;

    (void)pool.is_valid();
    (void)(pool == gpu::TIMESTAMP_POOL_HANDLE_INVALID);
    (void)gpu::DebugResourceKind.TIMESTAMP_POOL;
    (void)caps.queues;
    (void)caps.graphics_valid_bits;
    (void)caps.compute_valid_bits;
    (void)caps.transfer_valid_bits;
    (void)caps.period_ns;
    (void)gpu::create_timestamp_pool(&device, &desc);
    (void)gpu::destroy_timestamp_pool(&device, pool);
    (void)gpu::cmd_reset_timestamps(
        commands: &commands,
        pool:     pool,
        first:    0,
        count:    2,
    );
    (void)gpu::cmd_write_timestamp(
        commands: &commands,
        pool:     pool,
        index:    0,
        stage:    { .compute = true },
    );
    (void)gpu::cmd_resolve_timestamps(
        commands: &commands,
        pool:     pool,
        first:    0,
        count:    2,
        dst:      dst,
    );
    (void)gpu::read_timestamps(
        device: &device,
        pool:   pool,
        first:  0,
        count:  2,
        out:    values[..],
    );
    (void)gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: gpu::QueueKind.COMPUTE,
        begin: values[0],
        end:   values[1],
    );
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
    gpu::SparseTextureAspectRequirements sparse_aspect = {
        .aspect                 = gpu::SparseTextureAspect.METADATA,
        .tile_extent            = { 1, 1, 1 },
        .mip_tail_first_lod     = 1,
        .mip_tail_size          = 1,
        .mip_tail_offset        = 1,
        .mip_tail_stride        = 1,
        .single_mip_tail        = true,
        .aligned_mip_size       = true,
        .nonstandard_block_size = true,
    };
    gpu::SparseTextureRequirements sparse_requirements = {
        .virtual_size    = 1,
        .page_allocation = {},
        .aspect_count    = 1,
        .aspects         = { sparse_aspect, {} },
    };
    sparse_requirements.aspects[0].aspect =
        gpu::SparseTextureAspect.COLOR;
    gpu::SparseTextureTileBind[1] sparse_tiles = {
        {
            .aspect            = gpu::SparseTextureAspect.COLOR,
            .mip_level         = 0,
            .offset            = { 1, 1, 1 },
            .extent            = { 1, 1, 1 },
            .allocation        = allocation,
            .allocation_offset = 1,
        },
    };
    gpu::SparseTextureOpaqueBind[1] sparse_opaque = {
        {
            .aspect            = gpu::SparseTextureAspect.COLOR,
            .resource_offset   = 1,
            .size              = 1,
            .allocation        = allocation,
            .allocation_offset = 1,
        },
    };
    gpu::CompletionPoint[1] sparse_waits;
    gpu::SparseTextureBindDesc sparse_bind_desc = {
        .texture = texture,
        .tiles   = sparse_tiles[..],
        .opaque  = sparse_opaque[..],
        .waits   = sparse_waits[..],
    };
    gpu::Queue sparse_queue = {};
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
    (void)gpu::create_sparse_texture(&device, &texture_desc);
    (void)gpu::get_sparse_texture_requirements(&device, texture);
    (void)gpu::MAX_SPARSE_TEXTURE_ASPECTS;
    (void)sparse_requirements;
    (void)gpu::bind_sparse_texture_memory(sparse_queue, &sparse_bind_desc);
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
        .spirv = spirv[..],
    };
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::ComputePipelineDesc compute_desc = {
        .shader     = shader_desc,
        .debug_name = "compute",
    };
    gpu::GraphicsPipelineDesc graphics_desc = {
        .vertex_shader   = shader_desc,
        .fragment_shader = shader_desc,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::Format.D32_FLOAT,
        .sample_count    = gpu::SampleCount.ONE,
        .polygon_mode    = gpu::PolygonMode.FILL,
        .debug_name      = "graphics",
    };
    gpu::ShaderDesc[1] ray_generation_shaders = { shader_desc };
    gpu::RayTracingPipelineDesc ray_tracing_desc = {
        .ray_generation_shaders = ray_generation_shaders[..],
        .debug_name             = "ray tracing",
    };
    gpu::ColorTargetState[1] color_targets;
    gpu::ColorTargetState color_target =
        gpu::color_blend_disabled();
    gpu::ColorState color_state =
        gpu::uniform_color_state(color_targets[..], color_target);
    gpu::PipelineHandle pipeline = {};
    char[1] pipeline_cache;

    (void)gpu::create_compute_pipeline(&device, &compute_desc);
    (void)gpu::create_graphics_pipeline(&device, &graphics_desc);
    (void)gpu::create_ray_tracing_pipeline(&device, &ray_tracing_desc);
    (void)gpu::get_ray_tracing_pipeline_info(&device, pipeline);
    (void)gpu::get_ray_tracing_shader_group_handles(
        &device,
        pipeline,
        0,
        0,
        {},
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
    gpu::CommandAllocatorDesc command_allocator_desc = {
        .max_acceleration_structure_geometries_per_build = 8,
    };
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
        .after  = {
            .compute      = true,
            .depth_output = true,
            .indirect     = true,
            .acceleration_structure_build = true,
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
        .z       = 1,
        .depth   = 2,
    };
    gpu::TextureBufferCopyDesc texture_buffer_copy = {
        .texture = texture,
        .dst     = span,
        .z       = 1,
        .depth   = 2,
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
    gpu::RayTracingShaderBindingTable shader_binding_table;
    (void)gpu::cmd_trace_rays(
        commands:             &commands,
        root:                 {},
        shader_binding_table: &shader_binding_table,
        dimensions:           { 1, 1, 1 },
    );
    (void)gpu::cmd_dispatch_indirect(&commands, {}, span);
    (void)gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            span,
        count_span:         span,
        max_dispatch_count: 1,
    );
    (void)gpu::cmd_begin_render_pass(&commands, &render_pass);
    (void)gpu::cmd_set_graphics_state(&commands, &graphics_state);
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
