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

fn void full_invalid_barrier_is_rejected() @test {
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
    assert(@catch(gpu::cmd_barrier(&commands, null))
        == gpu::INVALID_ARGUMENT,
        "full mode rejects a null barrier");
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
        "rejected copy leaves both retained counters unchanged");
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
        .spirv       = POLICY_MATRIX_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader_desc,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::PipelineHandle create_policy_matrix_graphics_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc vertex_desc = {
        .spirv       = POLICY_MATRIX_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .spirv       = POLICY_MATRIX_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc vertex = vertex_desc;
    gpu::ShaderDesc fragment = fragment_desc;
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

fn void trusted_generated_dispatch_with_graphics_pipeline_faults_defined() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      gpu::ContractValidation.TRUSTED,
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::PipelineHandle pipeline =
        create_policy_matrix_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan indirect = policy_span(
        &device,
        { .graphics },
        "trusted_generated_wrong_pipeline",
    );
    defer free_owned_span(&device, &indirect)!!;

    vk::VkDeviceState* state = policy_device_state(&device);
    bool generated_work_enabled = state.generated_work.enabled;
    uint generated_work_limit = state.caps.max_generated_work_count;
    state.generated_work.enabled = true;
    state.caps.max_generated_work_count = 1;
    defer {
        state.generated_work.enabled = generated_work_enabled;
        state.caps.max_generated_work_count = generated_work_limit;
    }

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
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;

    fault wrong_pipeline = @catch(gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            indirect.span,
        count_span:         indirect.span,
        max_dispatch_count: 1,
    ));
    assert(wrong_pipeline == gpu::BACKEND_ERROR,
        "TRUSTED generated dispatch keeps null native lowering defined");
    assert(commands.is_valid(),
        "rejected generated dispatch preserves the recording token");
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
    OwnedSpan* source,
    OwnedSpan* destination,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    gpu::discard_commands(&commands)!!;
}

fn void exercise_policy_matrix_synchronization(
    gpu::CommandAllocator* allocator,
    gpu::TextureHandle texture,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    gpu::discard_commands(&commands)!!;
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
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
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
    gpu::discard_commands(&commands)!!;
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
    gpu::cmd_begin_render_pass(&minimal, pass)!!;
    gpu::cmd_end_render_pass(&minimal)!!;
    gpu::discard_commands(&minimal)!!;

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&commands, pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics)!!;
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
    gpu::discard_commands(&commands)!!;
}

fn void exercise_policy_matrix_labels(
    gpu::CommandAllocator* allocator,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_begin_label(&commands, "policy_matrix")!!;
    gpu::cmd_end_label(&commands)!!;
    gpu::discard_commands(&commands)!!;
}

fn void exercise_command_families(
    gpu::ContractValidation contract,
    bool vulkan_layers,
) {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      contract,
        vulkan_layers: vulkan_layers,
        callback:      true,
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

    capture.calls = 0;
    exercise_policy_matrix_transfer(
        allocator:   &allocator,
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
    exercise_policy_matrix_labels(&allocator);
    assert(capture.calls == 0,
        "recorded command families raise no recording diagnostic");
}

fn void command_families_run_under_each_contract_and_layer_mode() @test {
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
