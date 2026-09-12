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

fn gpu::PipelineHandle create_command_family_compute_pipeline(
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

fn gpu::PipelineHandle create_command_family_graphics_pipeline(
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

fn void generated_dispatch_requires_native_layout() @test {
    RuntimeDiagnosticCapture capture;
    gpu::RuntimeDesc desc = diagnostic_runtime_desc(
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::PipelineHandle pipeline =
        create_command_family_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan indirect = command_family_span(
        &device,
        { .graphics },
        "missing_generated_layout",
    );
    defer free_owned_span(&device, &indirect)!!;

    vk::VkDeviceState* state = diagnostic_device_state(&device);
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
        "Generated dispatch rejects a missing native layout");
    assert(commands.is_valid(),
        "rejected generated dispatch preserves the recording token");
}

fn gpu::TextureHandle create_command_family_transfer_texture(
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

fn gpu::TextureHandle create_command_family_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .width  = 8,
        .height = 8,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .color_attach },
        .access = { .graphics },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void exercise_command_family_transfer(
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

fn void exercise_command_family_synchronization(
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

fn void exercise_command_family_compute(
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

fn void exercise_command_family_render(
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

fn void exercise_command_family_labels(
    gpu::CommandAllocator* allocator,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_begin_label(&commands, "command_family")!!;
    gpu::cmd_end_label(&commands)!!;
    gpu::discard_commands(&commands)!!;
}

fn void exercise_command_families(
    bool vulkan_layers,
) {
    RuntimeDiagnosticCapture capture;
    gpu::RuntimeDesc desc = diagnostic_runtime_desc(
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
    vk::VkDeviceState* state = diagnostic_device_state(&device);

    gpu::PipelineHandle compute =
        create_command_family_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, compute)!!;
    gpu::PipelineHandle graphics =
        create_command_family_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, graphics)!!;
    OwnedSpan source = command_family_span(
        &device,
        { .graphics },
        "command_family_source",
    );
    defer free_owned_span(&device, &source)!!;
    OwnedSpan destination = command_family_span(
        &device,
        { .graphics },
        "command_family_destination",
    );
    defer free_owned_span(&device, &destination)!!;
    OwnedSpan indirect = command_family_span(
        &device,
        { .graphics },
        "command_family_indirect",
    );
    defer free_owned_span(&device, &indirect)!!;
    OwnedSpan indices = command_family_span(
        &device,
        { .graphics },
        "command_family_indices",
    );
    defer free_owned_span(&device, &indices)!!;
    gpu::TextureHandle transfer_texture =
        create_command_family_transfer_texture(&device);
    defer gpu::destroy_texture(&device, transfer_texture)!!;
    gpu::TextureHandle target = create_command_family_target(&device);
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
    exercise_command_family_transfer(
        allocator:   &allocator,
        source:      &source,
        destination: &destination,
        texture:     transfer_texture,
    );
    exercise_command_family_synchronization(&allocator, transfer_texture);
    exercise_command_family_compute(
        allocator: &allocator,
        state:     state,
        pipeline:  compute,
        indirect:  &indirect,
    );
    exercise_command_family_render(
        allocator: &allocator,
        state:     state,
        pipeline:  graphics,
        pass:      &pass,
        indices:   &indices,
        indirect:  &indirect,
    );
    exercise_command_family_labels(&allocator);
    assert(capture.calls == 0,
        "recorded command families raise no recording diagnostic");
}

fn void command_families_run_with_and_without_layers() @test {
    bool[2] layer_values = { false, true };
    {
        foreach (vulkan_layers : layer_values) {
            exercise_command_families(vulkan_layers);
        }
    }
}
