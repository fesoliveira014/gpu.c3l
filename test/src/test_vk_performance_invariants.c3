module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const char[*] PERFORMANCE_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] PERFORMANCE_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] PERFORMANCE_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

const uint PERFORMANCE_PIPELINE_CHURN_COUNT = 16;
const usz BOUND_PIPELINE_SNAPSHOT_SIZE = 96;

$assert vk::BoundPipeline::size == BOUND_PIPELINE_SNAPSHOT_SIZE;
$assert vk::BoundPipeline::members.len == 7;
$assert vk::BoundPipeline::members[0].name == "handle";
$assert vk::BoundPipeline::members[0].type == gpu::PipelineHandle::typeid;
$assert vk::BoundPipeline::members[1].name == "pipeline";
$assert vk::BoundPipeline::members[1].type == vk::Pipeline::typeid;
$assert vk::BoundPipeline::members[2].name == "layout";
$assert vk::BoundPipeline::members[2].type == vk::PipelineLayout::typeid;
$assert vk::BoundPipeline::members[3].name == "generated_dispatch_layout";
$assert vk::BoundPipeline::members[3].type
    == vk::IndirectCommandsLayoutEXT::typeid;
$assert vk::BoundPipeline::members[4].name == "cache_entry";
$assert vk::BoundPipeline::members[4].type == uint::typeid;
$assert vk::BoundPipeline::members[5].name == "kind";
$assert vk::BoundPipeline::members[5].type == vk::PipelineKind::typeid;
$assert vk::BoundPipeline::members[6].name == "render";
$assert vk::BoundPipeline::members[6].type
    == vk::RenderCompatibility::typeid;

fn gpu::Device make_performance_invariant_device() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_performance_invariants",
    };
    return create_test_device(&desc)!!;
}

fn void assert_zero_warm_creation_work(
    gpu::internal::RecordingWorkCounters before,
    gpu::internal::RecordingWorkCounters after,
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
    gpu::ShaderDesc alternate_vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = PERFORMANCE_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc alternate_fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = PERFORMANCE_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ColorTargetState[1] alternate_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc alternate_desc = {
        .vertex_shader = gpu::prepare_shader_code(&alternate_vertex_desc)!!,
        .fragment_shader = gpu::prepare_shader_code(
            &alternate_fragment_desc,
        )!!,
        .colors = alternate_colors[..],
    };
    gpu::PipelineHandle alternate_pipeline = gpu::create_graphics_pipeline(
        &device,
        &alternate_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, alternate_pipeline)!!;
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
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    gpu::CommandList warmup = gpu::begin_commands(&allocator)!!;
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
        gpu::internal::device_backend_state(device);
    gpu::internal::RecordingWorkCounters before =
        vk::command_recording_work_counters(state);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_before = vk::pipeline_native_create_count(state);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::reset_command_resolution_stats();
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::CommandResolutionStats after_bind =
        gpu::command_resolution_stats();
    assert_warm_command_resolution(after_bind, 1, 1);
    gpu::PipelineHandle[PERFORMANCE_PIPELINE_CHURN_COUNT] churned_pipelines;
    for (uint i = 0; i < churned_pipelines.len; i++) {
        churned_pipelines[i] = gpu::create_compute_pipeline(
            &device,
            &pipeline_desc,
        )!!;
    }
    foreach (churned : churned_pipelines) {
        assert(churned != pipeline,
            "pipeline-table churn publishes distinct owning handles");
        gpu::destroy_pipeline(&device, churned)!!;
    }
    gpu::CommandResolutionStats after_churn =
        gpu::command_resolution_stats();
    assert(after_churn.pipeline_table_lookups
            == after_bind.pipeline_table_lookups
                + PERFORMANCE_PIPELINE_CHURN_COUNT * 2
        && after_churn.pipeline_cache_lookups
            == after_bind.pipeline_cache_lookups,
        "pipeline churn performs two table lookups per destroyed alias and no cache lookup");
    gpu::reset_command_resolution_stats();
    gpu::CommandResolutionStats post_churn =
        gpu::command_resolution_stats();
    assert_warm_command_resolution(post_churn, 0);
    vk::PipelineCell* bound_cell = &state.pipelines.slots[pipeline.index];
    vk::PipelineCell* alternate_cell =
        &state.pipelines.slots[alternate_pipeline.index];
    vk::Pipeline saved_pipeline = bound_cell.value.pipeline;
    vk::PipelineLayout saved_layout = bound_cell.value.layout;
    vk::IndirectCommandsLayoutEXT saved_generated_layout =
        bound_cell.value.generated_dispatch_layout;
    uint saved_entry = bound_cell.value.entry;
    uint saved_generation = bound_cell.generation;
    bool saved_used = bound_cell.used;
    assert(saved_layout != alternate_cell.value.layout,
        "the alternate pipeline provides a distinct backing layout");
    vk::reset_root_push_observation();
    {
        defer {
            bound_cell.value.pipeline = saved_pipeline;
            bound_cell.value.layout = saved_layout;
            bound_cell.value.generated_dispatch_layout =
                saved_generated_layout;
            bound_cell.value.entry = saved_entry;
            bound_cell.generation = saved_generation;
            bound_cell.used = saved_used;
        };
        bound_cell.value.pipeline = alternate_cell.value.pipeline;
        bound_cell.value.layout = alternate_cell.value.layout;
        bound_cell.value.generated_dispatch_layout =
            alternate_cell.value.generated_dispatch_layout;
        bound_cell.value.entry = alternate_cell.value.entry;
        bound_cell.generation = pipeline.generation + 1;
        bound_cell.used = false;
        gpu::cmd_dispatch(
            commands: &commands,
            root:     test_span_address(&device, root.span),
            groups:   { 1, 1, 1 },
        )!!;
        vk::RootPushObservation observation =
            vk::root_push_observation();
        assert(observation.compute_layout == (ulong)saved_layout,
            "dispatch consumes the bind-time pipeline layout snapshot");
    }
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::RecordingWorkCounters after =
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
    assert(resolution.pipeline_table_lookups
            == post_churn.pipeline_table_lookups
        && resolution.pipeline_cache_lookups
            == post_churn.pipeline_cache_lookups,
        "dispatch performs no pipeline resolution after cache churn");
    assert(resolution.native_commands == post_churn.native_commands + 2,
        "dispatch emits one root push and one native dispatch");
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
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList warmup = gpu::begin_commands(&allocator)!!;
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
        gpu::internal::device_backend_state(device);
    gpu::internal::RecordingWorkCounters before =
        vk::command_recording_work_counters(state);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(
        state,
    );
    ulong pipelines_before = vk::pipeline_native_create_count(state);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::reset_command_resolution_stats();
    bind_test_graphics_pipeline(&commands, pipeline);
    gpu::CommandResolutionStats after_bind =
        gpu::command_resolution_stats();
    assert_warm_command_resolution(after_bind, 2, 1);
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert_warm_command_resolution(resolution, 3, 1);
    assert(resolution.pipeline_table_lookups
            == after_bind.pipeline_table_lookups
        && resolution.pipeline_cache_lookups
            == after_bind.pipeline_cache_lookups,
        "draw performs no additional pipeline resolution");
    assert(resolution.native_commands == after_bind.native_commands + 2,
        "draw emits one root push and one native draw");
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::RecordingWorkCounters after =
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
