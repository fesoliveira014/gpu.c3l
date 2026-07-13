module gpu_test;

import gpu::compat;

fn void null_descriptor_arguments_fault_without_backend_dereference() @test {
    gpu::compat::DeviceDesc device_desc = {
        .backend          = gpu::compat::BackendKind.VULKAN,
        .frames_in_flight = 2,
        .application_name = "gpu_null_contract_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    assert(
        @catch(gpu::compat::create_buffer(&device, (gpu::compat::BufferDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_buffer rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::supports_texture_desc(&device, (gpu::compat::TextureDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "supports_texture_desc rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_texture(&device, (gpu::compat::TextureDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_sampler(&device, (gpu::compat::SamplerDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_sampler rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_shader(&device, (gpu::compat::ShaderDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_shader rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_compute_pipeline(&device, (gpu::compat::ComputePipelineDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_compute_pipeline rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_graphics_pipeline(&device, (gpu::compat::GraphicsPipelineDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_graphics_pipeline rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::alloc_persistent_span(&device, (gpu::compat::PersistentAllocDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "alloc_persistent_span rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::submit(&device, (gpu::compat::SubmitDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "submit rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_swapchain(
            &device,
            (gpu::compat::SurfaceDesc*)null,
            (gpu::compat::SwapchainDesc*)null,
        )) == gpu::compat::INVALID_ARGUMENT,
        "create_swapchain rejects null descriptors"
    );
    assert(
        @catch(gpu::compat::present(&device, (gpu::compat::PresentDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "present rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::create_semaphore(&device, (gpu::compat::SemaphoreDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "create_semaphore rejects a null descriptor"
    );
}

fn void null_command_arguments_fault_without_recording_side_effects() @test {
    gpu::compat::DeviceDesc device_desc = {
        .backend          = gpu::compat::BackendKind.VULKAN,
        .frames_in_flight = 2,
        .application_name = "gpu_null_command_contract_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList commands = gpu::compat::begin_commands(
        &device,
        gpu::compat::QueueKind.GRAPHICS,
    )!!;

    assert(
        @catch(gpu::compat::cmd_copy_buffer(&commands, (gpu::compat::BufferCopyDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_copy_buffer rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::cmd_copy_buffer_to_texture(
            &commands,
            (gpu::compat::BufferTextureCopyDesc*)null,
        )) == gpu::compat::INVALID_ARGUMENT,
        "cmd_copy_buffer_to_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::cmd_copy_texture_to_buffer(
            &commands,
            (gpu::compat::TextureBufferCopyDesc*)null,
        )) == gpu::compat::INVALID_ARGUMENT,
        "cmd_copy_texture_to_buffer rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::cmd_buffer_barrier(&commands, (gpu::compat::BufferBarrier*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_buffer_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::compat::cmd_global_barrier(&commands, (gpu::compat::GlobalBarrier*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_global_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::compat::cmd_texture_barrier(&commands, (gpu::compat::TextureBarrier*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_texture_barrier rejects a null barrier"
    );
    assert(
        @catch(gpu::compat::cmd_begin_render_pass(&commands, (gpu::compat::RenderPassDesc*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_begin_render_pass rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::cmd_set_viewport(&commands, (gpu::compat::Viewport*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_set_viewport rejects a null viewport"
    );
    assert(
        @catch(gpu::compat::cmd_set_scissor(&commands, (gpu::compat::ScissorRect*)null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_set_scissor rejects a null scissor"
    );
    char[] no_data = {};
    assert(
        @catch(gpu::compat::cmd_upload_texture(
            &commands,
            (gpu::compat::TextureUploadDesc*)null,
            no_data,
        )) == gpu::compat::INVALID_ARGUMENT,
        "cmd_upload_texture rejects a null descriptor"
    );
    assert(
        @catch(gpu::compat::cmd_begin_label(&commands, null))
            == gpu::compat::INVALID_ARGUMENT,
        "cmd_begin_label rejects a null label"
    );

    gpu::compat::end_commands(&commands)!!;
    gpu::compat::end_frame(&frame)!!;
}
