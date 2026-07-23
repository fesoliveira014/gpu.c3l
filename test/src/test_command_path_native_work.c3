module gpu_test;

import gpu @public;

const char[*] COMMAND_PATH_WORK_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] COMMAND_PATH_WORK_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] COMMAND_PATH_WORK_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

const ulong COMMAND_PATH_BEGIN_PACKET_NATIVE_COMMANDS = 11;
const uint COMMAND_PATH_DRAW_COUNT = 4;

fn gpu::PipelineHandle command_path_work_compute_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = COMMAND_PATH_WORK_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = gpu::prepare_shader_code(&shader_desc)!!,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::PipelineHandle command_path_work_graphics_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = COMMAND_PATH_WORK_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = COMMAND_PATH_WORK_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ColorTargetState[1] colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = gpu::prepare_shader_code(&vertex_desc)!!,
        .fragment_shader = gpu::prepare_shader_code(&fragment_desc)!!,
        .colors          = colors[..],
    };
    return gpu::create_graphics_pipeline(device, &pipeline_desc)!!;
}

fn void require_native_work(ulong expected) {
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.recording_commands == 1);
    assert(stats.native_commands == expected);
}

fn void require_render_work(ulong expected_commands, ulong expected_native) {
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert(stats.recording_commands == expected_commands);
    assert(stats.native_commands == expected_native);
    assert(stats.pipeline_table_lookups == 0);
    assert(stats.pipeline_cache_lookups == 0);
}

fn void public_command_native_work_is_exact() @test {
    gpu::RuntimeDesc desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.TRUSTED,
        .track_resource_lifetimes = false,
        .enable_vulkan_validation = false,
        .application_name         = "gpu_command_path_native_work",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::PipelineHandle compute = command_path_work_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, compute)!!;
    gpu::PipelineHandle graphics =
        command_path_work_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, graphics)!!;

    OwnedSpan source = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "command_path_work_source",
    )!!;
    defer free_owned_span(&device, &source)!!;
    OwnedSpan destination = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "command_path_work_destination",
    )!!;
    defer free_owned_span(&device, &destination)!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::cmd_bind_pipeline(&commands, compute)!!;
    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch(
        &commands,
        (gpu::GpuAddress)0,
        (gpu::Vec3u){ 1, 1, 1 },
    )!!;
    require_native_work(2);
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    commands = gpu::begin_commands(&allocator)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::reset_command_resolution_stats();
    gpu::cmd_barrier(&commands, &barrier)!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    commands = gpu::begin_commands(&allocator)!!;
    gpu::BufferCopyDesc copy = {
        .src = source.span,
        .dst = destination.span,
    };
    gpu::reset_command_resolution_stats();
    gpu::cmd_copy_buffer(&commands, &copy)!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    gpu::TextureDesc target_desc = {
        .access = { .graphics },
        .width  = 8,
        .height = 8,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &view)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view     = view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = 8,
        .height = 8,
    };
    gpu::GraphicsState graphics_state =
        gpu::full_render_graphics_state(pass.width, pass.height)!!;
    commands = gpu::begin_commands(&allocator)!!;
    gpu::cmd_bind_pipeline(&commands, graphics)!!;
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_render_pass(&commands, &pass, &graphics_state)!!;
    require_render_work(1, COMMAND_PATH_BEGIN_PACKET_NATIVE_COMMANDS);
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    require_render_work(
        2,
        COMMAND_PATH_BEGIN_PACKET_NATIVE_COMMANDS + 2,
    );
    for (uint i = 1; i < COMMAND_PATH_DRAW_COUNT; i++) {
        gpu::cmd_draw(
            commands:       &commands,
            vertex_root:    (gpu::GpuAddress)1,
            fragment_root:  (gpu::GpuAddress)1,
            vertex_count:   3,
            instance_count: 1,
        )!!;
    }
    require_render_work(
        1 + COMMAND_PATH_DRAW_COUNT,
        COMMAND_PATH_BEGIN_PACKET_NATIVE_COMMANDS
            + COMMAND_PATH_DRAW_COUNT * 2,
    );
    gpu::Viewport viewport = {
        .width     = 8,
        .height    = 8,
        .max_depth = 1.0f,
    };
    gpu::reset_command_resolution_stats();
    gpu::cmd_set_viewport(&commands, &viewport)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    require_native_work(2);
    gpu::cmd_end_render_pass(&commands)!!;
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
}
