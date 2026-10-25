module gpu_test;

import gpu;

fn void null_descriptor_arguments_fault_without_backend_dereference() @test {
    gpu::RuntimeDesc device_desc = {
        .application_name = "gpu_null_contract_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    assert(
        @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, (gpu::SubmitDesc*)null))
            == gpu::INVALID_ARGUMENT,
        "submit rejects a null descriptor"
    );
    gpu::Surface no_surface;
    assert(
        @catch(gpu::create_swapchain(
            &device,
            &no_surface,
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
