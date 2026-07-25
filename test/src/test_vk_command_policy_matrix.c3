module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::atomic::types;

const char[*] POLICY_MATRIX_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] POLICY_MATRIX_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] POLICY_MATRIX_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

fn void begin_command_family_observation(vk::VkDeviceState* state) {
    vk::reset_command_recording_work_counters(state);
}

fn void finish_and_assert_command_family_observation(
    vk::VkDeviceState* state,
    gpu::CommandList* commands,
    ulong recording_commands,
    ulong native_commands,
    ulong generated_dispatch_emissions,
    ulong generated_draw_emissions,
    ulong generated_indexed_draw_emissions,
    ZString family,
) {
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    gpu::discard_commands(commands)!!;
    gpu::internal::RecordingWorkCounters recording =
        vk::command_recording_work_counters(state);
    assert(recording.host_allocations == 0
        && recording.command_pool_creations == 0
        && recording.command_buffer_allocations == 0
        && recording.command_buffer_frees == 0
        && recording.command_buffer_resets == 0
        && recording.image_view_creations == 0
        && recording.vma_allocations == 0
        && recording.registry_lock_acquisitions == 0
        && recording.generated_scratch_misses == 0,
        "%s command family performs no structural recording work", family);
    assert(recording.generated_dispatch_emissions
            == generated_dispatch_emissions
        && recording.generated_draw_emissions
            == generated_draw_emissions
        && recording.generated_indexed_draw_emissions
            == generated_indexed_draw_emissions,
        "%s command family emits the exact generated work", family);
    assert(resolution.device_registry_resolutions == 0
        && resolution.retained_pin_operations == 0
        && resolution.command_table_lookups == 0
        && resolution.pipeline_table_lookups == 0
        && resolution.pipeline_cache_lookups == 0,
        "%s command family performs no frontend lookup work", family);
    assert(resolution.recording_commands == recording_commands
        && resolution.native_commands == native_commands,
        "%s command family emits exact work: recording %d/%d, native %d/%d",
        family,
        resolution.recording_commands,
        recording_commands,
        resolution.native_commands,
        native_commands);
}

fn void full_invalid_barrier_emits_no_native_work() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      gpu::ContractValidation.FULL,
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
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
    defer gpu::discard_commands(&commands)!!;
    gpu::reset_command_resolution_stats();

    assert(@catch(gpu::cmd_barrier(&commands, null))
        == gpu::INVALID_ARGUMENT,
        "full mode rejects a null barrier");
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert(resolution.recording_commands == 1
        && resolution.native_commands == 0,
        "full rejection crosses the command boundary without native emission");
}

fn void full_copy_capacity_rejection_is_transactional() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      gpu::ContractValidation.FULL,
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
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
    defer gpu::discard_commands(&commands)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    vk::AllocationSlot* source_slot =
        state.allocations.get(source.allocation)!!;
    vk::AllocationSlot* destination_slot =
        state.allocations.get(destination.allocation)!!;
    uint source_references_before =
        source_slot.tracked_command_references.load(AtomicOrdering.RELAXED);
    uint destination_references_before =
        destination_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        );
    uint reference_count_before = record.scratch.reference_count;
    gpu::reset_command_resolution_stats();
    gpu::BufferCopyDesc copy = {
        .src = source.span,
        .dst = destination.span,
    };

    assert(@catch(gpu::cmd_copy_buffer(&commands, &copy))
        == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "two unique copy references exceed one fixed reference slot");
    assert(record.scratch.reference_count == reference_count_before,
        "rejected copy preserves the reference checkpoint");
    assert(source_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == source_references_before
        && destination_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == destination_references_before,
        "rejected copy performs no transient retain or release");
    gpu::CommandResolutionStats resolution =
        gpu::command_resolution_stats();
    assert(resolution.recording_commands == 1
        && resolution.native_commands == 0,
        "rejected copy emits no native command");
    assert(commands.is_valid()
        && slot.live_recordings == 1
        && slot.available_count == 0
        && record.scratch.in_use,
        "rejected copy preserves the recording token and allocator unit");
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
    gpu::ShaderCode vertex = gpu::prepare_shader_code(&vertex_desc)!!;
    gpu::ShaderCode fragment = gpu::prepare_shader_code(&fragment_desc)!!;
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = vertex,
        .fragment_shader = fragment,
        .color_formats   = color_formats[..],
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
        .usage  = { .sampled, .transfer_src, .transfer_dst },
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

fn void exercise_policy_matrix_transfer(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    OwnedSpan* source,
    OwnedSpan* destination,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    begin_command_family_observation(state);
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
    gpu::TextureBarrier readback_barrier = {
        .texture = texture,
        .before  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    };
    gpu::cmd_texture_barrier(&commands, &readback_barrier)!!;
    gpu::TextureBufferCopyDesc readback = {
        .texture = texture,
        .dst     = source.span,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &readback)!!;
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &commands,
        recording_commands:                 5,
        native_commands:                    5,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "transfer",
    );
}

fn void exercise_policy_matrix_synchronization(
    gpu::CommandAllocator* allocator,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    vk::VkDeviceState* state = commands.record.device_state;
    begin_command_family_observation(state);
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
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &commands,
        recording_commands:                 2,
        native_commands:                    2,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "synchronization",
    );
}

fn void exercise_policy_matrix_compute(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    OwnedSpan* indirect,
) {
    bool generated_work_enabled = state.generated_work.enabled;
    state.generated_work.enabled = true;
    defer state.generated_work.enabled = generated_work_enabled;

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    begin_command_family_observation(state);
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
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            {},
        count_span:         {},
        max_dispatch_count: 0,
    )!!;
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &commands,
        recording_commands:                 3,
        native_commands:                    4,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "compute",
    );
}

fn void exercise_policy_matrix_render(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
    OwnedSpan* indices,
    OwnedSpan* indirect,
) {
    bool generated_work_enabled = state.generated_work.enabled;
    state.generated_work.enabled = true;
    defer state.generated_work.enabled = generated_work_enabled;

    gpu::CommandList minimal = gpu::begin_commands(allocator)!!;
    begin_command_family_observation(state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_render_pass(&minimal, pass)!!;
    gpu::cmd_end_render_pass(&minimal)!!;
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &minimal,
        recording_commands:                 2,
        native_commands:                    2,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "minimal render pass",
    );

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics.color = { .targets = color_targets[..] };
    begin_command_family_observation(state);
    gpu::cmd_begin_render_pass(&commands, pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics)!!;
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
    gpu::ColorState color = { .targets = color_targets[..] };
    gpu::cmd_set_color_state(&commands, &color)!!;
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
    gpu::cmd_draw_indexed_indirect(
        commands:      &commands,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
        args:          indirect.span,
        draw_count:    1,
        index_span:    indices.span,
    )!!;
    if (state.draw_indirect_count) {
        gpu::cmd_draw_indexed_indirect_count(
            commands:       &commands,
            vertex_root:    (gpu::GpuAddress)1,
            fragment_root:  (gpu::GpuAddress)1,
            args:           indirect.span,
            count_span:     indirect.span,
            max_draw_count: 1,
            index_span:     indices.span,
        )!!;
    } else {
        assert(@catch(gpu::cmd_draw_indexed_indirect_count(
            commands:       &commands,
            vertex_root:    (gpu::GpuAddress)1,
            fragment_root:  (gpu::GpuAddress)1,
            args:           indirect.span,
            count_span:     indirect.span,
            max_draw_count: 1,
            index_span:     indices.span,
        )) == gpu::UNSUPPORTED_FEATURE,
            "unsupported indirect-count draw still dispatches through the selected table");
    }
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
    ulong native_commands = 38;
    if (state.draw_indirect_count) native_commands += 2;
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &commands,
        recording_commands:                 14,
        native_commands:                    native_commands,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "render",
    );
}

fn void exercise_policy_matrix_labels(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    begin_command_family_observation(state);
    gpu::reset_command_resolution_stats();
    gpu::cmd_begin_label(&commands, "policy_matrix")!!;
    gpu::cmd_end_label(&commands)!!;
    bool labels_supported = state.device_dispatch.debug.cmd_begin_label != null
        && state.device_dispatch.debug.cmd_end_label != null;
    finish_and_assert_command_family_observation(
        state:                              state,
        commands:                           &commands,
        recording_commands:                 2,
        native_commands:                    labels_supported ? 2 : 0,
        generated_dispatch_emissions:       0,
        generated_draw_emissions:           0,
        generated_indexed_draw_emissions:   0,
        family:                             "labels",
    );
}

fn void exercise_command_families(
    gpu::ContractValidation contract,
    bool vulkan_layers,
) {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      contract,
        vulkan_layers: vulkan_layers,
        callback:      false,
        debug_names:   true,
        capture:       &capture,
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

    exercise_policy_matrix_transfer(
        allocator:   &allocator,
        state:       state,
        source:      &source,
        destination: &destination,
        texture:     transfer_texture,
    );
    exercise_policy_matrix_synchronization(&allocator, transfer_texture);
    exercise_policy_matrix_compute(
        allocator: &allocator,
        state:     state,
        pipeline:  compute,
        indirect:  &indirect,
    );
    exercise_policy_matrix_render(
        allocator: &allocator,
        state:     state,
        pipeline:  graphics,
        pass:      &pass,
        indices:   &indices,
        indirect:  &indirect,
    );
    exercise_policy_matrix_labels(&allocator, state);
}

fn void command_families_run_under_both_contract_modes() @test {
    gpu::ContractValidation[2] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.FULL,
    };
    bool[2] layer_values = { false, true };
    foreach (contract : contracts) {
        foreach (vulkan_layers : layer_values) {
            exercise_command_families(contract, vulkan_layers);
        }
    }
}
