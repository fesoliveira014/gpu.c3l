module gpu_test;

import gpu;

fn void null_descriptor_arguments_fault_without_backend_dereference() @test {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_null_contract_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    assert(
        @catch(gpu::allocate_memory(&device, (gpu::AllocationDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "allocate_memory rejects a null descriptor"
    );
    assert(
        @catch(gpu::supports_texture_desc(&device, (gpu::TextureDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "supports_texture_desc rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_texture(&device, (gpu::TextureDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_sampler(&device, (gpu::SamplerDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_sampler rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_shader(&device, (gpu::ShaderDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_shader rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_compute_pipeline(&device, (gpu::ComputePipelineDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_compute_pipeline rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_graphics_pipeline(&device, (gpu::GraphicsPipelineDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_graphics_pipeline rejects a null descriptor"
    );
    assert(
        @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, (gpu::SubmitDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "submit rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_swapchain(
            &device,
            (gpu::Surface*)null,
            (gpu::SwapchainDesc*)null,
        )) == gpu::INVALID_ARGUMENT,
        "create_swapchain rejects null descriptors"
    );
    assert(
        @catch(gpu::present(
            &device,
            (gpu::AcquiredImage*)null,
            gpu::COMPLETION_POINT_INVALID,
        ))
            == gpu::INVALID_ARGUMENT,
        "present rejects a null acquired image"
    );
}

fn void null_command_arguments_fault_without_recording_side_effects() @test {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_null_command_contract_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    assert(
        @catch(gpu::cmd_copy_buffer(&commands, (gpu::BufferCopyDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_copy_buffer rejects a null descriptor"
    );
    assert(
        @catch(gpu::cmd_copy_buffer_to_texture(
            &commands,
            (gpu::BufferTextureCopyDesc*)null,
        )) == gpu::INVALID_ARGUMENT,
        "cmd_copy_buffer_to_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::cmd_copy_texture_to_buffer(
            &commands,
            (gpu::TextureBufferCopyDesc*)null,
        )) == gpu::INVALID_ARGUMENT,
        "cmd_copy_texture_to_buffer rejects a null descriptor"
    );
    assert(
        @catch(gpu::cmd_buffer_barrier(&commands, (gpu::BufferBarrier*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_buffer_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::cmd_global_barrier(&commands, (gpu::GlobalBarrier*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_global_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::cmd_texture_barrier(&commands, (gpu::TextureBarrier*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_texture_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::cmd_begin_render_pass(&commands, (gpu::RenderPassDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_begin_render_pass rejects a null descriptor"
    );
    assert(
        @catch(gpu::cmd_set_viewport(&commands, (gpu::Viewport*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_set_viewport rejects a null viewport"
    );
    assert(
        @catch(gpu::cmd_set_scissor(&commands, (gpu::ScissorRect*)null))
            == gpu::INVALID_ARGUMENT,
        "cmd_set_scissor rejects a null scissor"
    );
    assert(
        @catch(gpu::cmd_begin_label(&commands, null))
            == gpu::INVALID_ARGUMENT,
        "cmd_begin_label rejects a null label"
    );

    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}
