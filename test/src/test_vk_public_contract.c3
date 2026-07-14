module gpu_test;

import gpu;

fn void null_descriptor_arguments_fault_without_backend_dereference() @test {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .frames_in_flight = 2,
        .application_name = "gpu_null_contract_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    assert(
        @catch(gpu::create_buffer(&device, (gpu::BufferDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_buffer rejects a null descriptor"
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
        @catch(gpu::alloc_persistent_span(&device, (gpu::PersistentAllocDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "alloc_persistent_span rejects a null descriptor"
    );
    assert(
        @catch(gpu::submit(&device, (gpu::SubmitDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "submit rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_swapchain(
            &device,
            (gpu::SurfaceDesc*)null,
            (gpu::SwapchainDesc*)null,
        )) == gpu::INVALID_ARGUMENT,
        "create_swapchain rejects null descriptors"
    );
    assert(
        @catch(gpu::present(&device, (gpu::PresentDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "present rejects a null descriptor"
    );
    assert(
        @catch(gpu::create_semaphore(&device, (gpu::SemaphoreDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "create_semaphore rejects a null descriptor"
    );
}

fn void null_command_arguments_fault_without_recording_side_effects() @test {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .frames_in_flight = 2,
        .application_name = "gpu_null_command_contract_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;

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
    char[] no_data = {};
    assert(
        @catch(gpu::cmd_upload_texture(
            &commands,
            (gpu::TextureUploadDesc*)null,
            no_data,
        )) == gpu::INVALID_ARGUMENT,
        "cmd_upload_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::cmd_begin_label(&commands, null))
            == gpu::INVALID_ARGUMENT,
        "cmd_begin_label rejects a null label"
    );

    gpu::end_commands(&commands)!!;
    gpu::end_frame(&frame)!!;
}
