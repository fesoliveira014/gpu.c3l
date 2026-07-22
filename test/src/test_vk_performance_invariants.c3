module gpu_test;

import gpu @public;
import gpu::vk @public;

const char[*] PERFORMANCE_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] PERFORMANCE_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] PERFORMANCE_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

fn gpu::Device make_performance_invariant_device() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_performance_invariants",
    };
    return create_test_device(&desc)!!;
}

fn void assert_zero_warm_creation_work(
    gpu::RecordingWorkCounters before,
    gpu::RecordingWorkCounters after,
    ulong shader_modules_before,
    ulong shader_modules_after,
    ulong pipelines_before,
    ulong pipelines_after,
) {
    assert(after.host_allocations == before.host_allocations,
        "warm recording performs no host allocation");
    assert(after.command_buffer_allocations
            == before.command_buffer_allocations
        && after.command_buffer_frees == before.command_buffer_frees,
        "warm recording allocates and frees no command buffer");
    assert(after.image_view_creations == before.image_view_creations,
        "warm recording creates no image view");
    assert(after.vma_allocations == before.vma_allocations,
        "warm recording performs no VMA allocation");
    assert(shader_modules_after == shader_modules_before,
        "warm recording creates no shader module");
    assert(pipelines_after == pipelines_before,
        "warm recording creates no pipeline");
}

fn void warm_bind_dispatch_complete_operation_has_zero_creation_work() @test {
    gpu::Device device = make_performance_invariant_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = PERFORMANCE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan root = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "warm_dispatch_root",
    )!!;
    defer free_owned_span(&device, &root)!!;
    gpu::TextureDesc texture_desc = {
        .access = { .compute },
        .width  = 1,
        .height = 1,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(
        &device,
        &texture_desc,
    )!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView view = gpu::create_texture_view(
        &device,
        texture,
        null,
    )!!;
    defer gpu::destroy_texture_view(&device, view)!!;
    gpu::SamplerDesc sampler_desc;
    gpu::SamplerIndex sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    (void)sampler;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::CommandList warmup = gpu::begin_commands(queue)!!;
    bind_test_compute_pipeline(&warmup, pipeline);
    gpu::cmd_dispatch(
        commands: &warmup,
        root:     test_span_address(&device, root.span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::ExecutableCommandList warmup_executable =
        gpu::end_commands(&warmup)!!;
    gpu::discard_executable_commands(&warmup_executable)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::device_backend_state(device);
    gpu::RecordingWorkCounters before =
        vk::command_recording_work_counters(state);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_before = vk::pipeline_native_create_count(state);
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(&device, root.span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::RecordingWorkCounters after =
        vk::command_recording_work_counters(state);
    ulong shader_modules_after = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_after = vk::pipeline_native_create_count(state);

    assert_zero_warm_creation_work(
        before:                before,
        after:                 after,
        shader_modules_before: shader_modules_before,
        shader_modules_after:  shader_modules_after,
        pipelines_before:      pipelines_before,
        pipelines_after:       pipelines_after,
    );
    assert(after.command_buffer_resets == before.command_buffer_resets + 1,
        "warm begin resets its reusable command buffer");
    assert_warm_command_resolution(resolution, 1);
    gpu::discard_executable_commands(&executable)!!;
}

fn void warm_render_pass_complete_operation_has_zero_creation_work() @test {
    gpu::Device device = make_performance_invariant_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = PERFORMANCE_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = PERFORMANCE_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader = gpu::prepare_shader_code(&vertex_desc)!!,
        .fragment_shader = gpu::prepare_shader_code(&fragment_desc)!!,
        .colors = pipeline_colors[..],
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
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
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList warmup = gpu::begin_commands(queue)!!;
    gpu::TextureBarrier transition = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&warmup, &transition)!!;
    gpu::cmd_begin_render_pass(&warmup, &pass)!!;
    bind_test_graphics_pipeline(&warmup, pipeline);
    gpu::cmd_draw(
        commands:       &warmup,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&warmup)!!;
    gpu::ExecutableCommandList warmup_executable =
        gpu::end_commands(&warmup)!!;
    gpu::discard_executable_commands(&warmup_executable)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::device_backend_state(device);
    gpu::RecordingWorkCounters before =
        vk::command_recording_work_counters(state);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_before = vk::pipeline_native_create_count(state);
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    bind_test_graphics_pipeline(&commands, pipeline);
    gpu::reset_command_resolution_stats();
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert_warm_command_resolution(resolution, 1);
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::RecordingWorkCounters after =
        vk::command_recording_work_counters(state);
    ulong shader_modules_after = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_after = vk::pipeline_native_create_count(state);
    assert_zero_warm_creation_work(
        before:                before,
        after:                 after,
        shader_modules_before: shader_modules_before,
        shader_modules_after:  shader_modules_after,
        pipelines_before:      pipelines_before,
        pipelines_after:       pipelines_after,
    );
    assert(after.command_buffer_resets == before.command_buffer_resets + 1,
        "warm render recording resets its reusable command buffer");
    gpu::discard_executable_commands(&executable)!!;
}
