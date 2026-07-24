module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const char[*] COMMAND_PATH_WORK_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] COMMAND_PATH_WORK_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] COMMAND_PATH_WORK_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

const ulong COMMAND_PATH_BEGIN_NATIVE_COMMANDS = 1;
const ulong COMMAND_PATH_GRAPHICS_STATE_NATIVE_COMMANDS = 10;
const ulong COMMAND_PATH_DEPTH_STATE_NATIVE_COMMANDS = 3;
const ulong COMMAND_PATH_RASTER_STATE_NATIVE_COMMANDS = 5;
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
    assert_warm_command_resolution(stats, 1);
    assert(stats.recording_commands == 1);
    assert(stats.native_commands == expected);
    assert(stats.pipeline_bind_commands == 0);
    assert(stats.strict_heap_set_bind_commands == 0);
    assert(stats.strict_heap_buffer_bind_commands == 0);
    assert(stats.strict_heap_offset_commands == 0);
    $if $feature(GPU_FAST_COMMANDS):
        assert(stats.contract_branches == 0
                && stats.tracking_branches == 0
                && stats.public_misuse_faults == 0,
            "FAST commands perform no runtime policy or misuse-fault work");
    $endif
}

fn void require_render_work(ulong expected_commands, ulong expected_native) {
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();
    assert_warm_command_resolution(stats, expected_commands);
    assert(stats.recording_commands == expected_commands);
    assert(stats.native_commands == expected_native);
    assert(stats.pipeline_bind_commands == 0);
    assert(stats.strict_heap_set_bind_commands == 0);
    assert(stats.strict_heap_buffer_bind_commands == 0);
    assert(stats.strict_heap_offset_commands == 0);
    assert(stats.pipeline_table_lookups == 0);
    assert(stats.pipeline_cache_lookups == 0);
    $if $feature(GPU_FAST_COMMANDS):
        assert(stats.contract_branches == 0
                && stats.tracking_branches == 0
                && stats.public_misuse_faults == 0,
            "FAST render commands perform no runtime policy or misuse-fault work");
    $endif
}

fn void public_command_native_work_is_exact() @test {
    gpu::RuntimeDesc desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .enable_vulkan_validation = false,
        .application_name         = "gpu_command_path_native_work",
    };
    $if $feature(GPU_FAST_COMMANDS):
        desc.contract_validation = gpu::ContractValidation.TRUSTED;
        desc.track_resource_lifetimes = false;
    $else
        $if $feature(DIRECT_COMMAND_TOKENS):
            desc.contract_validation =
                gpu::ContractValidation.OBJECT_BOUNDARIES;
            desc.track_resource_lifetimes = false;
        $else
            desc.contract_validation = gpu::ContractValidation.FULL;
            desc.track_resource_lifetimes = true;
        $endif
    $endif
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
    gpu::GpuAddress root = test_span_address(&device, source.span);
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    ulong label_native_commands =
        vk::device_dispatch_supports_labels(&state.device_dispatch) ? 1 : 0;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    bind_test_compute_pipeline(&commands, compute);
    gpu::reset_command_resolution_stats();
    record_test_dispatch(
        &commands,
        root,
        (gpu::Vec3u){ 1, 1, 1 },
    )!!;
    require_native_work(2);
    gpu::reset_command_resolution_stats();
    record_test_dispatch_indirect(&commands, root, source.span)!!;
    require_native_work(2);
    gpu::reset_command_resolution_stats();
    record_test_begin_label(
        &commands,
        "command profile work",
        { 0.25f, 0.5f, 0.75f, 1.0f },
    )!!;
    require_native_work(label_native_commands);
    gpu::reset_command_resolution_stats();
    record_test_end_label(&commands)!!;
    require_native_work(label_native_commands);
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    commands = gpu::begin_commands(&allocator)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::reset_command_resolution_stats();
    record_test_barrier(&commands, &barrier)!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    commands = gpu::begin_commands(&allocator)!!;
    gpu::BufferCopyDesc copy = {
        .src = source.span,
        .dst = destination.span,
    };
    gpu::reset_command_resolution_stats();
    record_test_copy_buffer(&commands, &copy)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_fill_buffer(
        &commands,
        destination.span,
        0xA5A5A5A5,
    )!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    gpu::TextureDesc transfer_desc = {
        .access = { .transfer },
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .transfer_src, .transfer_dst },
    };
    gpu::TextureHandle transfer_texture = gpu::create_texture(
        &device,
        &transfer_desc,
    )!!;
    defer gpu::destroy_texture(&device, transfer_texture)!!;
    gpu::TextureBarrier to_transfer_destination = {
        .texture = transfer_texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::TextureBarrier to_transfer_source = {
        .texture = transfer_texture,
        .before  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    };
    gpu::BufferTextureCopyDesc upload = {
        .src     = source.span,
        .texture = transfer_texture,
    };
    gpu::TextureBufferCopyDesc readback = {
        .texture = transfer_texture,
        .dst     = destination.span,
    };
    commands = gpu::begin_commands(&allocator)!!;
    gpu::reset_command_resolution_stats();
    record_test_texture_barrier(
        &commands,
        &to_transfer_destination,
    )!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_buffer_to_texture(&commands, &upload)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_texture_barrier(&commands, &to_transfer_source)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_texture_to_buffer(&commands, &readback)!!;
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
    gpu::TextureBarrier target_transition = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    commands = gpu::begin_commands(&allocator)!!;
    gpu::reset_command_resolution_stats();
    record_test_texture_barrier(&commands, &target_transition)!!;
    require_native_work(1);
    bind_test_graphics_pipeline(&commands, graphics);
    gpu::reset_command_resolution_stats();
    record_test_depth_state(&commands, &graphics_state.depth)!!;
    require_native_work(COMMAND_PATH_DEPTH_STATE_NATIVE_COMMANDS);
    gpu::reset_command_resolution_stats();
    record_test_raster_state(&commands, &graphics_state.raster)!!;
    require_native_work(COMMAND_PATH_RASTER_STATE_NATIVE_COMMANDS);
    gpu::reset_command_resolution_stats();
    record_test_viewport(&commands, &graphics_state.viewport)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_scissor(&commands, &graphics_state.scissor)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_graphics_state(&commands, &graphics_state)!!;
    require_render_work(
        1,
        COMMAND_PATH_GRAPHICS_STATE_NATIVE_COMMANDS,
    );
    gpu::reset_command_resolution_stats();
    record_test_begin_render_pass(&commands, &pass)!!;
    require_render_work(1, COMMAND_PATH_BEGIN_NATIVE_COMMANDS);
    record_test_draw(
        commands:       &commands,
        vertex_root:    root,
        fragment_root:  root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    require_render_work(
        2,
        COMMAND_PATH_BEGIN_NATIVE_COMMANDS + 2,
    );
    for (uint i = 1; i < COMMAND_PATH_DRAW_COUNT; i++) {
        record_test_draw(
            commands:       &commands,
            vertex_root:    root,
            fragment_root:  root,
            vertex_count:   3,
            instance_count: 1,
        )!!;
    }
    require_render_work(
        1 + COMMAND_PATH_DRAW_COUNT,
        COMMAND_PATH_BEGIN_NATIVE_COMMANDS
            + COMMAND_PATH_DRAW_COUNT * 2,
    );
    gpu::Viewport viewport = {
        .width     = 8,
        .height    = 8,
        .max_depth = 1.0f,
    };
    gpu::reset_command_resolution_stats();
    record_test_viewport(&commands, &viewport)!!;
    require_native_work(1);
    gpu::reset_command_resolution_stats();
    record_test_draw(
        commands:       &commands,
        vertex_root:    root,
        fragment_root:  root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    require_native_work(2);
    gpu::reset_command_resolution_stats();
    record_test_draw_indexed(
        commands:       &commands,
        vertex_root:    root,
        fragment_root:  root,
        index_span:     source.span,
        index_count:    3,
        instance_count: 1,
    )!!;
    require_native_work(3);
    gpu::reset_command_resolution_stats();
    record_test_draw_indirect(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        args:          source.span,
        draw_count:    1,
    )!!;
    require_native_work(2);
    gpu::reset_command_resolution_stats();
    record_test_draw_indexed_indirect(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        args:          source.span,
        draw_count:    1,
        index_span:    source.span,
    )!!;
    require_native_work(3);
    assert(gpu::get_device_caps(&device)!!.draw_indirect_count,
        "the pinned profile-equivalence adapter supports indirect count");
    gpu::reset_command_resolution_stats();
    record_test_draw_indexed_indirect_count(
        commands:       &commands,
        vertex_root:    root,
        fragment_root:  root,
        args:           source.span,
        count_span:     destination.span,
        max_draw_count: 1,
        index_span:     source.span,
    )!!;
    require_native_work(3);
    gpu::reset_command_resolution_stats();
    record_test_end_render_pass(&commands)!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;

    commands = gpu::begin_commands(&allocator)!!;
    record_test_texture_barrier(&commands, &target_transition)!!;
    bind_test_graphics_pipeline(&commands, graphics);
    gpu::reset_command_resolution_stats();
    record_test_begin_render_pass_with_state(
        &commands,
        &pass,
        &graphics_state,
    )!!;
    require_native_work(
        COMMAND_PATH_BEGIN_NATIVE_COMMANDS
            + COMMAND_PATH_GRAPHICS_STATE_NATIVE_COMMANDS,
    );
    gpu::reset_command_resolution_stats();
    record_test_end_render_pass(&commands)!!;
    require_native_work(1);
    executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
}
