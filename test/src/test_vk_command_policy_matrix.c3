module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

const char[*] POLICY_MATRIX_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] POLICY_MATRIX_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] POLICY_MATRIX_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

struct CommandPolicyExpectation {
    ulong semantic_checks;
    ulong tracking_calls;
    ulong reference_allocations;
    ulong reference_increments;
    ulong reference_releases;
    ulong recording_commands;
    ulong command_table_dispatches;
    ulong native_commands;
    ulong generated_dispatch_emissions;
    ulong generated_draw_emissions;
    ulong generated_indexed_draw_emissions;
    bool  rejected;
}

struct CommandPolicyObservation {
    vk::CommandPolicyCounters            policy;
    gpu::internal::RecordingWorkCounters recording;
    vk::CommandRecordingStats            recording_before;
    vk::CommandRecordingStats            recording_after;
    gpu::CommandResolutionStats          resolution;
}

struct PolicyMatrixCommandContext {
    gpu::CommandAllocator* allocator;
    vk::VkDeviceState*     state;
    gpu::ContractValidation contract;
    bool                    tracking;
    OwnedSpan*              source;
    OwnedSpan*              destination;
    OwnedSpan*              indirect;
    OwnedSpan*              indices;
    gpu::TextureHandle      transfer_texture;
    gpu::PipelineHandle     compute;
    gpu::PipelineHandle     graphics;
    gpu::RenderPassDesc*    pass;
}

fn bool command_policy_observation_matches(
    CommandPolicyObservation* observation,
    CommandPolicyExpectation* expectation,
) {
    vk::CommandPolicyCounters policy = observation.policy;
    gpu::internal::RecordingWorkCounters recording = observation.recording;
    vk::CommandRecordingStats recording_before = observation.recording_before;
    vk::CommandRecordingStats recording_after = observation.recording_after;
    gpu::CommandResolutionStats resolution = observation.resolution;
    ulong command_table_lookups = $feature(GPU_FAST_COMMANDS)
        ? 0
        : expectation.command_table_dispatches;
    ulong generated_acquires = expectation.generated_dispatch_emissions
        + expectation.generated_draw_emissions
        + expectation.generated_indexed_draw_emissions;
    return policy.semantic_checks == expectation.semantic_checks
        && policy.tracking_calls == expectation.tracking_calls
        && policy.reference_allocations == expectation.reference_allocations
        && policy.reference_increments == expectation.reference_increments
        && policy.reference_releases == expectation.reference_releases
        && recording.host_allocations == 0
        && recording.command_pool_creations == 0
        && recording.command_buffer_allocations == 0
        && recording.command_buffer_frees == 0
        && recording.command_buffer_resets == 0
        && recording.image_view_creations == 0
        && recording.vma_allocations == 0
        && recording.registry_lock_acquisitions == 0
        && recording.generated_preprocess_acquires == generated_acquires
        && recording.generated_allocator_lock_acquisitions
            == generated_acquires
        && recording.generated_scratch_misses == 0
        && recording.generated_dispatch_emissions
            == expectation.generated_dispatch_emissions
        && recording.generated_draw_emissions
            == expectation.generated_draw_emissions
        && recording.generated_indexed_draw_emissions
            == expectation.generated_indexed_draw_emissions
        && recording_after.preprocess_allocations
            == recording_before.preprocess_allocations
        && recording_after.preprocess_reuses
            == recording_before.preprocess_reuses
        && recording_after.recording_allocations
            == recording_before.recording_allocations
        && resolution.device_registry_resolutions == 0
        && resolution.retained_pin_operations == 0
        && resolution.lifecycle_vtable_dispatches == 0
        && resolution.command_table_lookups == command_table_lookups
        && resolution.command_table_indirect_loads == command_table_lookups
        && resolution.command_table_indirect_calls == command_table_lookups
        && resolution.pipeline_table_lookups == 0
        && resolution.pipeline_cache_lookups == 0
        && resolution.command_policy_selections == 0
        && resolution.encoder_cell_computations == 0
        && resolution.encoder_lease_comparisons == 0
        && resolution.recording_commands == expectation.recording_commands
        && resolution.native_commands == expectation.native_commands
        && (!expectation.rejected || resolution.native_commands == 0);
}

fn void assert_command_policy_observation(
    CommandPolicyObservation* observation,
    CommandPolicyExpectation* expectation,
    ZString group,
) {
    assert(command_policy_observation_matches(observation, expectation),
        "%s command-policy observation matches its exact expectation", group);
}

fn void assert_command_policy_mutation_rejected(
    CommandPolicyObservation* observation,
    CommandPolicyExpectation* expectation,
    ZString mutation,
) {
    assert(!command_policy_observation_matches(observation, expectation),
        "%s mutation is rejected by command-policy evidence", mutation);
}

fn ulong policy_matrix_semantic_checks(
    gpu::ContractValidation contract,
    ulong full_checks,
) {
    return contract == gpu::ContractValidation.FULL ? full_checks : 0;
}

fn CommandPolicyExpectation policy_matrix_expectation(
    gpu::ContractValidation contract,
    bool tracking,
    ulong full_semantic_checks,
    ulong tracking_calls,
    ulong unique_references,
    ulong recording_commands,
    ulong native_commands,
) {
    return {
        .semantic_checks       = policy_matrix_semantic_checks(
            contract,
            full_semantic_checks,
        ),
        .tracking_calls        = tracking ? tracking_calls : 0,
        .reference_allocations = 0,
        .reference_increments  = tracking ? unique_references : 0,
        .reference_releases    = tracking ? unique_references : 0,
        .recording_commands    = recording_commands,
        .command_table_dispatches = recording_commands,
        .native_commands       = native_commands,
    };
}

fn vk::CommandRecordingStats begin_policy_matrix_observation(
    vk::VkDeviceState* state,
) {
    vk::reset_command_policy_counters(state);
    vk::reset_command_recording_work_counters(state);
    return vk::command_recording_stats(state);
}

fn CommandPolicyObservation finish_policy_matrix_observation(
    vk::VkDeviceState* state,
    gpu::CommandList* commands,
    vk::CommandRecordingStats recording_before,
    gpu::CommandResolutionStats resolution,
) {
    gpu::discard_commands(commands)!!;
    return {
        .policy           = vk::command_policy_counters(state),
        .recording        = vk::command_recording_work_counters(state),
        .recording_before = recording_before,
        .recording_after  = vk::command_recording_stats(state),
        .resolution       = resolution,
    };
}

fn gpu::PipelineHandle create_policy_matrix_compute_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = POLICY_MATRIX_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = gpu::prepare_shader_code(&shader_desc)!!,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::PipelineHandle create_policy_matrix_graphics_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = POLICY_MATRIX_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = POLICY_MATRIX_FRAG_SPIRV[..],
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

fn gpu::TextureHandle create_policy_matrix_transfer_texture(
    gpu::Device* device,
) {
    gpu::TextureDesc desc = {
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .transfer_src, .transfer_dst },
        .access = { .graphics },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle create_policy_matrix_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .width  = 8,
        .height = 8,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .color_attach },
        .access = { .graphics },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void observe_policy_matrix_transfer(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    OwnedSpan* source,
    OwnedSpan* destination,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);

    gpu::reset_command_resolution_stats();
    gpu::BufferCopyDesc copy = {
        .src = source.span,
        .dst = destination.span,
    };
    gpu::cmd_copy_buffer(&commands, &copy)!!;
    gpu::cmd_fill_buffer(
        commands: &commands,
        dst:      destination.span,
        value:    0xA5A5_A5A5,
    )!!;
    gpu::BufferTextureCopyDesc upload = {
        .src     = destination.span,
        .texture = texture,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &upload)!!;
    gpu::TextureBufferCopyDesc readback = {
        .texture = texture,
        .dst     = source.span,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &readback)!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 4,
        tracking_calls:       7,
        unique_references:    3,
        recording_commands:   4,
        native_commands:      4,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "transfer",
    );
}

fn void observe_policy_matrix_synchronization(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);

    gpu::reset_command_resolution_stats();
    gpu::Barrier barrier = {
        .before = { .transfer },
        .after  = { .compute },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::TextureBarrier texture_barrier = {
        .texture = texture,
        .before  = {
            .layout = gpu::TextureLayout.UNDEFINED,
        },
        .after = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &texture_barrier)!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 2,
        tracking_calls:       1,
        unique_references:    1,
        recording_commands:   2,
        native_commands:      2,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "synchronization",
    );
}

fn void observe_policy_matrix_compute(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    gpu::PipelineHandle pipeline,
    OwnedSpan* indirect,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)1,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::cmd_dispatch_indirect(
        commands: &commands,
        root:     (gpu::GpuAddress)1,
        args:     indirect.span,
    )!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 3,
        tracking_calls:       2,
        unique_references:    2,
        recording_commands:   2,
        native_commands:      4,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "compute",
    );
}

fn void observe_policy_matrix_render(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
    OwnedSpan* indices,
    OwnedSpan* indirect,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    gpu::GraphicsState graphics =
        gpu::full_render_graphics_state(pass.width, pass.height)!!;
    gpu::cmd_begin_render_pass_with_state(&commands, pass, &graphics)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_set_graphics_state(&commands, &graphics)!!;
    gpu::Viewport viewport = {
        .width     = 8.0f,
        .height    = 8.0f,
        .max_depth = 1.0f,
    };
    gpu::cmd_set_viewport(&commands, &viewport)!!;
    gpu::ScissorRect scissor = {
        .width  = 8,
        .height = 8,
    };
    gpu::cmd_set_scissor(&commands, &scissor)!!;
    gpu::DepthState depth;
    gpu::cmd_set_depth_state(&commands, &depth)!!;
    gpu::DynamicRasterState raster;
    gpu::cmd_set_raster_state(&commands, &raster)!!;
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_draw_indexed(
        commands:       &commands,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        index_span:     indices.span,
        index_count:    3,
        instance_count: 1,
    )!!;
    gpu::cmd_draw_indirect(
        commands:      &commands,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
        args:          indirect.span,
        draw_count:    1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 10,
        tracking_calls:       4,
        unique_references:    4,
        recording_commands:   9,
        native_commands:      28,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "render",
    );
}

fn void observe_policy_matrix_labels(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);

    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_label(&commands, "policy_matrix")!!;
    gpu::cmd_end_label(&commands)!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    bool labels_supported = state.device_dispatch.debug.cmd_begin_label != null
        && state.device_dispatch.debug.cmd_end_label != null;
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 0,
        tracking_calls:       0,
        unique_references:    0,
        recording_commands:   2,
        native_commands:      labels_supported ? 2 : 0,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "labels",
    );
}

fn void observe_policy_matrix_generated_dispatch(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    gpu::PipelineHandle pipeline,
) {
    bool generated_work_enabled = state.generated_work.enabled;
    state.generated_work.enabled = true;
    defer state.generated_work.enabled = generated_work_enabled;

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            {},
        count_span:         {},
        max_dispatch_count: 0,
    )!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 2,
        tracking_calls:       1,
        unique_references:    1,
        recording_commands:   1,
        native_commands:      0,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "generated dispatch",
    );
}

fn void observe_policy_matrix_generated_render(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::ContractValidation contract,
    bool tracking,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
) {
    bool generated_work_enabled = state.generated_work.enabled;
    state.generated_work.enabled = true;
    defer state.generated_work.enabled = generated_work_enabled;

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    gpu::GraphicsState graphics =
        gpu::full_render_graphics_state(pass.width, pass.height)!!;
    gpu::cmd_begin_render_pass_with_state(&commands, pass, &graphics)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;

    gpu::reset_command_resolution_stats();
    gpu::cmd_draw_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
    )!!;
    gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
        index_span:    {},
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             contract,
        tracking:             tracking,
        full_semantic_checks: 4,
        tracking_calls:       2,
        unique_references:    2,
        recording_commands:   3,
        native_commands:      1,
    );
    assert_command_policy_observation(
        &observation,
        &expectation,
        "generated render",
    );
}

fn void? observe_policy_matrix_commands(void* user_data) {
    PolicyMatrixCommandContext* context =
        (PolicyMatrixCommandContext*)user_data;
    observe_policy_matrix_transfer(
        allocator:   context.allocator,
        state:       context.state,
        contract:    context.contract,
        tracking:    context.tracking,
        source:      context.source,
        destination: context.destination,
        texture:     context.transfer_texture,
    );
    observe_policy_matrix_synchronization(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
        texture:   context.transfer_texture,
    );
    observe_policy_matrix_compute(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
        pipeline:  context.compute,
        indirect:  context.indirect,
    );
    observe_policy_matrix_render(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
        pipeline:  context.graphics,
        pass:      context.pass,
        indices:   context.indices,
        indirect:  context.indirect,
    );
    observe_policy_matrix_generated_dispatch(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
        pipeline:  context.compute,
    );
    observe_policy_matrix_generated_render(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
        pipeline:  context.graphics,
        pass:      context.pass,
    );
    observe_policy_matrix_labels(
        allocator: context.allocator,
        state:     context.state,
        contract:  context.contract,
        tracking:  context.tracking,
    );
}

fn void exercise_policy_matrix_row(
    gpu::ContractValidation contract,
    bool tracking,
) {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       contract,
        track_lifetimes: tracking,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     true,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);

    gpu::PipelineHandle compute =
        create_policy_matrix_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, compute)!!;
    gpu::PipelineHandle graphics =
        create_policy_matrix_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, graphics)!!;
    OwnedSpan source = policy_span(
        &device,
        { .graphics },
        "policy_matrix_source",
    );
    defer free_owned_span(&device, &source)!!;
    OwnedSpan destination = policy_span(
        &device,
        { .graphics },
        "policy_matrix_destination",
    );
    defer free_owned_span(&device, &destination)!!;
    OwnedSpan indirect = policy_span(
        &device,
        { .graphics },
        "policy_matrix_indirect",
    );
    defer free_owned_span(&device, &indirect)!!;
    OwnedSpan indices = policy_span(
        &device,
        { .graphics },
        "policy_matrix_indices",
    );
    defer free_owned_span(&device, &indices)!!;
    gpu::TextureHandle transfer_texture =
        create_policy_matrix_transfer_texture(&device);
    defer gpu::destroy_texture(&device, transfer_texture)!!;
    gpu::TextureHandle target = create_policy_matrix_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = 8,
        .height = 8,
    };

    PolicyMatrixCommandContext context = {
        .allocator        = &allocator,
        .state            = state,
        .contract         = contract,
        .tracking         = tracking,
        .source           = &source,
        .destination      = &destination,
        .indirect         = &indirect,
        .indices          = &indices,
        .transfer_texture = transfer_texture,
        .compute          = compute,
        .graphics         = graphics,
        .pass             = &pass,
    };
    observe_confined_thread_allocator(
        &state.recording_host_allocator,
        &observe_policy_matrix_commands,
        &context,
    )!!;
}

fn void full_tracking_off_invalid_barrier_emits_no_native_work() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:        gpu::ContractValidation.FULL,
        track_lifetimes: false,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    gpu::reset_command_resolution_stats();

    fault rejected = @catch(gpu::cmd_barrier(&commands, null));
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert(rejected == gpu::INVALID_ARGUMENT,
        "full tracking-off rejects a null barrier");

    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             gpu::ContractValidation.FULL,
        tracking:             false,
        full_semantic_checks: 0,
        tracking_calls:       0,
        unique_references:    0,
        recording_commands:   1,
        native_commands:      0,
    );
    expectation.command_table_dispatches = 0;
    expectation.rejected = true;
    assert_command_policy_observation(
        &observation,
        &expectation,
        "full tracking-off rejected barrier",
    );
}

fn void full_tracking_copy_capacity_rejection_is_transactional() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        track_lifetimes: true,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    OwnedSpan source = policy_span(
        &device,
        { .graphics },
        "policy_rejected_source",
    );
    defer free_owned_span(&device, &source)!!;
    OwnedSpan destination = policy_span(
        &device,
        { .graphics },
        "policy_rejected_destination",
    );
    defer free_owned_span(&device, &destination)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity          = 1,
        .max_resource_references_per_list = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    vk::VkCommandRecord* record = vk::resolve_command(&commands)!!;
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    uint reference_count_before = record.scratch.reference_count;
    uint reference_epoch_before = record.scratch.reference_index.epoch;
    vk::CommandRecordingStats recording_before =
        begin_policy_matrix_observation(state);
    vk::reset_command_reference_work_counters(state);
    gpu::reset_command_resolution_stats();
    gpu::BufferCopyDesc copy = {
        .src = source.span,
        .dst = destination.span,
    };

    fault rejected = @catch(gpu::cmd_copy_buffer(&commands, &copy));
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert(rejected == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "two unique copy references exceed one fixed reference slot");
    uint expected_reference_epoch = reference_epoch_before == 0
        ? 1
        : reference_epoch_before;
    assert(record.scratch.reference_count == reference_count_before
        && record.scratch.reference_index.epoch == expected_reference_epoch,
        "rejected copy preserves the reference checkpoint without publication");
    vk::CommandReferenceWorkCounters reference_work =
        vk::command_reference_work_counters(state);
    assert(reference_work.lookups == 2
        && reference_work.probes == 2
        && reference_work.equality_checks == 0
        && reference_work.index_publications == 0
        && reference_work.duplicate_hits == 0
        && reference_work.resource_mutex_acquisitions == 0
        && reference_work.retains == 0
        && reference_work.releases == 0
        && reference_work.rollback_rebuild_publications == 0
        && reference_work.rollover_cells_cleared == 0,
        "rejected copy performs exact preflight work without retain or rollback publication");
    assert(commands.handle.is_valid()
        && slot.live_recordings == 1
        && slot.available_count == 0
        && record.scratch.in_use,
        "rejected copy preserves the recording token and allocator unit");

    CommandPolicyObservation observation = finish_policy_matrix_observation(
        state,
        &commands,
        recording_before,
        resolution,
    );
    CommandPolicyExpectation expectation = policy_matrix_expectation(
        contract:             gpu::ContractValidation.FULL,
        tracking:             true,
        full_semantic_checks: 1,
        tracking_calls:       0,
        unique_references:    0,
        recording_commands:   1,
        native_commands:      0,
    );
    expectation.rejected = true;
    assert_command_policy_observation(
        &observation,
        &expectation,
        "rejected copy",
    );
}

fn void command_policy_matcher_rejects_each_forbidden_mutation() @test {
    CommandPolicyObservation passing;
    CommandPolicyExpectation expected;
    assert(command_policy_observation_matches(&passing, &expected),
        "zero observation is a known passing policy baseline");

    CommandPolicyObservation mutation = passing;
    CommandPolicyExpectation private_budget = expected;
    private_budget.recording_commands = 1;
    private_budget.command_table_dispatches = 1;
    CommandPolicyObservation lower_private_work = passing;
    lower_private_work.resolution.recording_commands = 1;
    $if !$feature(GPU_FAST_COMMANDS):
        lower_private_work.resolution.command_table_lookups = 1;
        lower_private_work.resolution.command_table_indirect_loads = 1;
        lower_private_work.resolution.command_table_indirect_calls = 1;
    $endif
    assert(command_policy_observation_matches(
        &lower_private_work,
        &private_budget,
    ), "removed encoder proof work is a known passing policy baseline");
    mutation = lower_private_work;
    mutation.resolution.encoder_cell_computations = 2;
    assert_command_policy_mutation_rejected(
        &mutation,
        &private_budget,
        "encoder-cell budget",
    );
    mutation = lower_private_work;
    mutation.resolution.encoder_lease_comparisons = 2;
    assert_command_policy_mutation_rejected(
        &mutation,
        &private_budget,
        "encoder-lease budget",
    );

    mutation = passing;
    mutation.policy.semantic_checks = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "trusted semantic work",
    );

    mutation = passing;
    mutation.policy.reference_increments = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "tracking-off retain",
    );

    mutation = passing;
    mutation.policy.reference_releases = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "unmatched release",
    );

    CommandPolicyObservation tracked_passing = passing;
    tracked_passing.policy.reference_increments = 1;
    tracked_passing.policy.reference_releases = 1;
    CommandPolicyExpectation tracked_expected = expected;
    tracked_expected.reference_increments = 1;
    tracked_expected.reference_releases = 1;
    assert(command_policy_observation_matches(
        &tracked_passing,
        &tracked_expected,
    ), "balanced tracking observation is a known passing policy baseline");
    mutation = tracked_passing;
    mutation.policy.reference_releases = 0;
    assert_command_policy_mutation_rejected(
        &mutation,
        &tracked_expected,
        "missing tracking release",
    );

    mutation = passing;
    mutation.recording.host_allocations = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "host allocation",
    );

    mutation = passing;
    mutation.recording.registry_lock_acquisitions = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "registry lock",
    );

    mutation = passing;
    mutation.resolution.command_policy_selections = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "policy reselection",
    );

    mutation = passing;
    mutation.resolution.pipeline_table_lookups = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "pipeline table resolution",
    );

    mutation = passing;
    mutation.resolution.pipeline_cache_lookups = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "pipeline cache resolution",
    );

    mutation = passing;
    mutation.resolution.native_commands = 1;
    assert_command_policy_mutation_rejected(
        &mutation,
        &expected,
        "extra native work",
    );

    mutation = passing;
    mutation.resolution.native_commands = 1;
    CommandPolicyExpectation rejected_expected = expected;
    rejected_expected.native_commands = 1;
    rejected_expected.rejected = true;
    assert_command_policy_mutation_rejected(
        &mutation,
        &rejected_expected,
        "rejected-input native work",
    );
}

fn void grouped_command_families_cover_all_six_policy_rows() @test {
    gpu::ContractValidation[3] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.OBJECT_BOUNDARIES,
        gpu::ContractValidation.FULL,
    };
    bool[2] tracking_values = { false, true };
    foreach (contract : contracts) {
        foreach (tracking : tracking_values) {
            exercise_policy_matrix_row(contract, tracking);
        }
    }
}
