module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::surface::wayland;
import gpu::surface::win32;
import gpu::surface::x11;
import gpu::internal::vk @public;

fn Surface create_test_surface(Runtime* runtime) {
    return gpu::surface::win32::create_surface(
        runtime,
        (gpu::surface::win32::InstanceHandle)(void*)1,
        (gpu::surface::win32::WindowHandle)(void*)2,
    )!!;
}

fn gpu::internal::vk::VkDeviceState* present_wait_state(Device* device) {
    return gpu::internal::device_slots[device.index].data.state;
}

fn void assert_device_ownership(
    Device* device,
    Adapter expected_adapter,
    Surface expected_surface,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!!;
    defer gpu::internal::end_device_operation(&operation);
    assert(
        (ulong)operation.data.adapter == (ulong)expected_adapter,
        "the device retains its exact adapter"
    );
    assert(
        (ulong)operation.data.presentation_surface == (ulong)expected_surface,
        "the device retains its exact presentation surface"
    );
}

fn void surface_tokens_are_compact() @test {
    assert(Surface::size == ulong::size, "Surface is one opaque 64-bit token");
}

fn void forged_surface_tokens_are_rejected() @test {
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface first = create_test_surface(&runtime);
    gpu::destroy_surface(&first)!!;

    Surface surface = create_test_surface(&runtime);
    Surface reserved = surface;
    reserved.reserved = 1;
    Surface out_of_range = surface;
    out_of_range.index = gpu::internal::MAX_SURFACES;
    Surface wrong_generation = surface;
    wrong_generation.generation--;

    assert(@catch(gpu::destroy_surface(&reserved)) == gpu::INVALID_HANDLE);
    assert(@catch(gpu::destroy_surface(&out_of_range)) == gpu::INVALID_HANDLE);
    assert(@catch(gpu::destroy_surface(&wrong_generation)) == gpu::INVALID_HANDLE);

    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void surfaces_retain_their_runtime_and_expire_on_destroy() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Surface stale = surface;

    assert(@catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE);
    assert(gpu::internal::vk::stub_surface_create_count() == 1);
    gpu::destroy_surface(&surface)!!;
    assert(!surface.is_valid());
    assert(@catch(gpu::destroy_surface(&stale)) == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_surface_destroy_count() == 1);
    gpu::destroy_runtime(&runtime)!!;
}

fn void surface_publication_failure_is_transactional() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface[gpu::internal::MAX_SURFACES] surfaces;
    for (uint i = 0; i < gpu::internal::MAX_SURFACES; i++) {
        surfaces[i] = create_test_surface(&runtime);
    }

    assert(
        @catch(gpu::surface::win32::create_surface(
            &runtime,
            (gpu::surface::win32::InstanceHandle)(void*)1,
            (gpu::surface::win32::WindowHandle)(void*)2,
        )) == gpu::SLOT_TABLE_FULL,
        "a full surface table faults cleanly"
    );
    assert(
        gpu::internal::vk::stub_surface_create_count()
            == gpu::internal::MAX_SURFACES + 1,
        "the failed publication creates one native surface"
    );
    assert(
        gpu::internal::vk::stub_surface_destroy_count() == 1,
        "the unpublished native surface is destroyed immediately"
    );

    foreach (&surface : surfaces) gpu::destroy_surface(surface)!!;
    assert(
        gpu::internal::vk::stub_surface_create_count()
            == gpu::internal::vk::stub_surface_destroy_count(),
        "every created native surface is released"
    );
    gpu::destroy_runtime(&runtime)!!;
}

fn void invalid_platform_handles_fault_before_backend_work() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;

    assert(
        @catch(gpu::surface::win32::create_surface(
            &runtime,
            (gpu::surface::win32::InstanceHandle)null,
            (gpu::surface::win32::WindowHandle)(void*)1,
        )) == gpu::INVALID_ARGUMENT
    );
    assert(
        @catch(gpu::surface::wayland::create_surface(
            &runtime,
            (gpu::surface::wayland::DisplayHandle)null,
            (gpu::surface::wayland::SurfaceHandle)(void*)1,
        )) == gpu::INVALID_ARGUMENT
    );
    assert(
        @catch(gpu::surface::x11::create_surface(
            &runtime,
            (gpu::surface::x11::DisplayHandle)(void*)1,
            (gpu::surface::x11::WindowHandle)0,
        )) == gpu::INVALID_ARGUMENT
    );
    assert(gpu::internal::vk::stub_surface_create_count() == 0);
}

fn void presentation_support_validates_runtime_ownership() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime first = gpu::create_runtime(&desc)!!;
    Runtime second = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&first);
    Adapter first_adapter = gpu::enumerate_adapters(&first)!!.get(0)!!;
    Adapter second_adapter = gpu::enumerate_adapters(&second)!!.get(0)!!;

    assert(gpu::supports_presentation(&first_adapter, &surface)!!);
    assert(
        @catch(gpu::supports_presentation(&second_adapter, &surface)) == gpu::INVALID_HANDLE
    );

    gpu::internal::vk::stub_set_presentation_support(false);
    assert(!gpu::supports_presentation(&first_adapter, &surface)!!);

    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&second)!!;
    gpu::destroy_runtime(&first)!!;
}

fn void presentation_desc_is_immutable_and_surface_bound() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    DeviceDesc original = device_desc;

    assert(gpu::supports_device_desc(&adapter, &device_desc)!!.supported);
    assert(device_desc_equal(device_desc, original));

    Device device = gpu::create_device(&adapter, &device_desc)!!;
    QueueRoles queues = gpu::get_device_caps(&device)!!.queues;
    Queue graphics = gpu::get_queue(&device, QueueKind.GRAPHICS)!!;
    assert(queues.graphics, "a supported presentation descriptor admits graphics");
    assert(graphics.roles.graphics, "the admitted graphics queue supports graphics");
    gpu::destroy_device(&device)!!;

    gpu::internal::vk::stub_set_presentation_support(false);
    DeviceSupport unsupported = gpu::supports_device_desc(&adapter, &device_desc)!!;
    assert(!unsupported.supported);
    assert(unsupported.unmet_requirement == "presentation");

    gpu::destroy_surface(&surface)!!;
    assert(@catch(gpu::supports_device_desc(&adapter, &device_desc))
        == gpu::INVALID_HANDLE);
    assert(@catch(gpu::create_device(&adapter, &device_desc))
        == gpu::INVALID_HANDLE);
    gpu::destroy_runtime(&runtime)!!;
}

fn void descriptor_surface_identity_rejects_before_backend_work() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime first = gpu::create_runtime(&desc)!!;
    Runtime second = gpu::create_runtime(&desc)!!;
    Surface first_surface = create_test_surface(&first);
    Surface second_surface = create_test_surface(&second);
    Surface malformed = first_surface;
    malformed.reserved = 1;
    Adapter adapter = gpu::enumerate_adapters(&first)!!.get(0)!!;
    DeviceDesc foreign_desc = { .surface = second_surface };
    DeviceDesc malformed_desc = { .surface = malformed };

    assert(@catch(gpu::supports_device_desc(&adapter, &foreign_desc))
        == gpu::INVALID_HANDLE);
    assert(@catch(gpu::create_device(&adapter, &foreign_desc))
        == gpu::INVALID_HANDLE);
    assert(@catch(gpu::supports_device_desc(&adapter, &malformed_desc))
        == gpu::INVALID_HANDLE);
    assert(@catch(gpu::create_device(&adapter, &malformed_desc))
        == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0);
    assert(gpu::internal::vk::stub_device_temporary_create_count() == 0);

    gpu::destroy_surface(&second_surface)!!;
    gpu::destroy_surface(&first_surface)!!;
    gpu::destroy_runtime(&second)!!;
    gpu::destroy_runtime(&first)!!;
}

fn void presentation_desc_requires_graphics() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc invalid = {
        .surface = surface,
        .queues  = { .required = { .compute } },
    };

    assert(
        @catch(gpu::supports_device_desc(&adapter, &invalid))
            == gpu::INVALID_ARGUMENT
    );
    assert(
        @catch(gpu::create_device(&adapter, &invalid)) == gpu::INVALID_ARGUMENT
    );
    assert(
        gpu::internal::vk::stub_device_support_query_count() == 0,
        "invalid presentation descriptors bypass support queries"
    );
    assert(
        gpu::internal::vk::stub_device_temporary_create_count() == 0,
        "invalid presentation descriptors allocate no device state"
    );

    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void live_swapchains_block_surface_destruction() @test {
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);

    gpu::internal::surface_retain_swapchain(&surface)!!;
    assert(@catch(gpu::destroy_surface(&surface)) == gpu::RESOURCE_IN_USE);
    assert(surface.is_valid());
    gpu::internal::surface_release_swapchain(&surface)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchains_borrow_runtime_owned_surfaces() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = {
        .width  = 640,
        .height = 480,
    };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;

    assert(@catch(gpu::destroy_surface(&surface)) == gpu::RESOURCE_IN_USE);
    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_creation_rejects_a_foreign_runtime_surface() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime first = gpu::create_runtime(&desc)!!;
    Runtime second = gpu::create_runtime(&desc)!!;
    Surface first_surface = create_test_surface(&first);
    Surface second_surface = create_test_surface(&second);
    Adapter adapter = gpu::enumerate_adapters(&first)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = first_surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = {
        .width  = 640,
        .height = 480,
    };

    assert(
        @catch(gpu::create_swapchain(
            &device,
            &second_surface,
            &swapchain_desc,
        )) == gpu::INVALID_HANDLE
    );

    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&second_surface)!!;
    gpu::destroy_surface(&first_surface)!!;
    gpu::destroy_runtime(&second)!!;
    gpu::destroy_runtime(&first)!!;
}

fn void non_presenting_devices_reject_swapchains_before_backend_work() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    Device device = gpu::create_device(&adapter)!!;
    SwapchainDesc swapchain_desc = {
        .width  = 640,
        .height = 480,
    };

    assert(!gpu::get_device_caps(&device)!!.presentation_enabled);
    assert(
        @catch(gpu::create_swapchain(
            &device,
            &surface,
            &swapchain_desc,
        )) == gpu::UNSUPPORTED_FEATURE
    );
    assert(
        gpu::internal::vk::stub_swapchain_create_attempt_count() == 0,
        "non-presenting rejection precedes backend mutation"
    );

    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void presentation_devices_accept_only_the_requested_surface() @test {
    gpu::internal::vk::stub_set_adapter_count(2);
    defer gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_reset_surface_counts();
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface first_surface = create_test_surface(&runtime);
    Surface second_surface = create_test_surface(&runtime);
    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    Adapter first_adapter = adapters.get(0)!!;
    Adapter second_adapter = adapters.get(1)!!;
    DeviceDesc first_desc = { .surface = first_surface };
    DeviceDesc second_desc = { .surface = second_surface };
    Device first = gpu::create_device(&first_adapter, &first_desc)!!;
    Device second = gpu::create_device(&second_adapter, &second_desc)!!;
    SwapchainDesc swapchain_desc = {
        .width  = 640,
        .height = 480,
    };

    assert(gpu::get_device_caps(&first)!!.presentation_enabled);
    assert(gpu::get_device_caps(&second)!!.presentation_enabled);
    assert_device_ownership(&first, first_adapter, first_surface);
    assert_device_ownership(&second, second_adapter, second_surface);
    assert(
        @catch(gpu::create_swapchain(
            &first,
            &second_surface,
            &swapchain_desc,
        )) == gpu::INVALID_HANDLE
    );
    assert(
        @catch(gpu::create_swapchain(
            &second,
            &first_surface,
            &swapchain_desc,
        )) == gpu::INVALID_HANDLE
    );
    assert(
        gpu::internal::vk::stub_swapchain_create_attempt_count() == 0,
        "both surface mismatches reject before backend mutation"
    );

    SwapchainHandle first_swapchain = gpu::create_swapchain(
        &first,
        &first_surface,
        &swapchain_desc,
    )!!;
    SwapchainHandle second_swapchain = gpu::create_swapchain(
        &second,
        &second_surface,
        &swapchain_desc,
    )!!;
    assert(gpu::internal::vk::stub_swapchain_create_attempt_count() == 2);

    gpu::destroy_swapchain(&first, first_swapchain)!!;
    gpu::destroy_device(&first)!!;
    gpu::destroy_surface(&first_surface)!!;
    assert(gpu::get_device_caps(&second)!!.presentation_enabled);
    assert_device_ownership(&second, second_adapter, second_surface);
    assert(@catch(gpu::destroy_surface(&second_surface)) == gpu::RESOURCE_IN_USE);

    gpu::destroy_swapchain(&second, second_swapchain)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_surface(&second_surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_rejects_foreign_and_stale_handles_before_native_work() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 1);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    SwapchainHandle foreign = swapchain;
    foreign.owner++;
    assert(
        @catch(gpu::wait_swapchain_presentations(&device, foreign, 0))
            == gpu::INVALID_HANDLE
    );
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 0
            && gpu::internal::vk::stub_present_reset_call_count(state) == 0,
        "foreign handles are rejected before the native wait"
    );

    SwapchainHandle stale = swapchain;
    stale.generation--;
    assert(
        @catch(gpu::wait_swapchain_presentations(&device, stale, 0))
            == gpu::INVALID_HANDLE
    );
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 0
            && gpu::internal::vk::stub_present_reset_call_count(state) == 0,
        "stale handles are rejected before the native wait"
    );

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_returns_immediately_without_pending_presentations() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 0);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    gpu::wait_swapchain_presentations(&device, swapchain, 37)!!;
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 0
            && gpu::internal::vk::stub_present_reset_call_count(state) == 0,
        "no pending presentations bypass native wait and reset"
    );

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_timeout_preserves_pending_state_and_supports_retry() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 2);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    fault timeout = @catch(
        gpu::wait_swapchain_presentations(&device, swapchain, 0)
    );
    assert(timeout == gpu::WAIT_TIMEOUT, "zero timeout reports WAIT_TIMEOUT");
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 1
            && gpu::internal::vk::stub_present_reset_call_count(state) == 0
            && gpu::internal::vk::stub_present_wait_last_timeout(state) == 0
            && gpu::internal::vk::stub_present_wait_pending_count(state) == 2,
        "timeout preserves every pending presentation"
    );

    gpu::wait_swapchain_presentations(&device, swapchain, 1)!!;
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 2
            && gpu::internal::vk::stub_present_reset_call_count(state) == 1
            && gpu::internal::vk::stub_present_wait_pending_count(state) == 0,
        "a retry clears pending state only after success"
    );

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_default_uses_infinite_timeout() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 1);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    gpu::wait_swapchain_presentations(&device, swapchain)!!;
    assert(
        gpu::internal::vk::stub_present_wait_last_timeout(state)
            == gpu::TIMEOUT_INFINITE,
        "the omitted timeout forwards TIMEOUT_INFINITE"
    );
    assert(
        gpu::internal::vk::stub_present_wait_pending_count(state) == 0,
        "the default wait retires pending presentations"
    );

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_backend_failure_preserves_handle() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 1);
    gpu::internal::vk::stub_set_present_wait_fault(state, gpu::DEVICE_LOST);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    fault failure = @catch(
        gpu::wait_swapchain_presentations(&device, swapchain, 99)
    );
    assert(failure == gpu::DEVICE_LOST, "backend failure propagates unchanged");
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 1
            && gpu::internal::vk::stub_present_reset_call_count(state) == 0
            && gpu::internal::vk::stub_present_wait_pending_count(state) == 1,
        "backend failure leaves pending state and handle intact"
    );

    gpu::internal::vk::stub_set_present_wait_fault(state, {});
    gpu::internal::vk::stub_set_present_reset_fault(state, gpu::DEVICE_LOST);
    fault reset_failure = @catch(
        gpu::wait_swapchain_presentations(&device, swapchain, 99)
    );
    assert(reset_failure == gpu::DEVICE_LOST,
        "reset failure propagates without consuming the handle");
    assert(
        gpu::internal::vk::stub_present_reset_call_count(state) == 1
            && gpu::internal::vk::stub_present_wait_pending_count(state) == 1,
        "reset failure preserves pending fake-backend state"
    );

    gpu::internal::vk::stub_set_present_reset_fault(state, {});
    gpu::wait_swapchain_presentations(&device, swapchain, 99)!!;
    assert(gpu::internal::vk::stub_present_wait_pending_count(state) == 0,
        "a later successful call can retire the preserved state");
    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_success_is_idempotent() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;
    gpu::internal::vk::VkDeviceState* state = present_wait_state(&device);
    gpu::internal::vk::stub_set_present_wait_pending_count(state, 1);
    gpu::internal::vk::stub_reset_present_wait_counts(state);

    gpu::wait_swapchain_presentations(&device, swapchain, 1)!!;
    gpu::wait_swapchain_presentations(&device, swapchain, 0)!!;
    assert(
        gpu::internal::vk::stub_present_wait_call_count(state) == 1
            && gpu::internal::vk::stub_present_reset_call_count(state) == 1
            && gpu::internal::vk::stub_present_wait_pending_count(state) == 0,
        "a repeated successful wait performs no additional native work"
    );

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void swapchain_wait_rejects_destroyed_device() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc device_desc = { .surface = surface };
    Device device = gpu::create_device(&adapter, &device_desc)!!;
    SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    SwapchainHandle swapchain = gpu::create_swapchain(
        &device,
        &surface,
        &swapchain_desc,
    )!!;

    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    assert(
        @catch(gpu::wait_swapchain_presentations(&device, swapchain, 0))
            == gpu::INVALID_HANDLE,
        "a destroyed device rejects before backend swapchain access"
    );

    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}
