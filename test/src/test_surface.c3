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

fn void surface_creation_rejects_an_incomplete_backend_contract() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    gpu::internal::vk::stub_set_surface_destroy_available(false);
    defer gpu::internal::vk::stub_set_surface_destroy_available(true);

    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;

    assert(
        @catch(gpu::surface::win32::create_surface(
            &runtime,
            (gpu::surface::win32::InstanceHandle)(void*)1,
            (gpu::surface::win32::WindowHandle)(void*)2,
        )) == gpu::UNSUPPORTED_FEATURE
    );
    assert(
        gpu::internal::vk::stub_surface_create_count() == 0,
        "an unusable surface must not reach the backend"
    );
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

fn void presentation_request_is_immutable_and_surface_bound() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceRequest strict = gpu::strict_device_request();
    DeviceRequest original = strict;
    DeviceRequest presentation = gpu::request_presentation(strict, &surface)!!;

    assert(DeviceRequest::size == uint128::size, "request fits in two machine words");
    assert((uint128)strict == (uint128)original, "composition preserves its input");
    assert((uint128)presentation != (uint128)strict, "presentation returns a new value");
    assert(
        @catch(gpu::request_presentation(presentation, &surface)) == gpu::INVALID_ARGUMENT
    );
    assert(gpu::supports_device_request(&adapter, &presentation)!!.supported);

    Device device = gpu::create_device(&adapter, &presentation)!!;
    QueueCounts counts = gpu::get_queue_counts(&device)!!;
    Queue graphics = gpu::get_queue(&device, QueueKind.GRAPHICS)!!;
    assert(counts.graphics == 1, "a supported presentation request admits graphics");
    assert(graphics.roles.graphics, "the admitted graphics queue supports graphics");
    gpu::destroy_device(&device)!!;

    gpu::internal::vk::stub_set_presentation_support(false);
    DeviceRequestSupport unsupported = gpu::supports_device_request(&adapter, &presentation)!!;
    assert(!unsupported.supported);
    assert(unsupported.unmet_requirement == "presentation");

    gpu::destroy_surface(&surface)!!;
    assert(@catch(gpu::supports_device_request(&adapter, &presentation)) == gpu::INVALID_HANDLE);
    gpu::destroy_runtime(&runtime)!!;
}

fn void presentation_requests_require_graphics_in_every_composition_order() @test {
    gpu::internal::vk::stub_set_presentation_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    QueueRequirements compute_only = { .counts = { .compute = 1 } };
    DeviceRequest strict = gpu::strict_device_request();
    DeviceRequest queues_first = gpu::request_queues(strict, compute_only)!!;
    DeviceRequest presentation_first = gpu::request_presentation(strict, &surface)!!;

    assert(
        @catch(gpu::request_presentation(queues_first, &surface))
            == gpu::INVALID_ARGUMENT,
        "presentation cannot be added to a request without graphics"
    );
    assert(
        @catch(gpu::request_queues(presentation_first, compute_only))
            == gpu::INVALID_ARGUMENT,
        "graphics cannot be removed from a presentation request"
    );

    DeviceRequest forged = (DeviceRequest)((uint128)queues_first
        | (uint128)presentation_first);
    assert(
        @catch(gpu::supports_device_request(&adapter, &forged))
            == gpu::INVALID_ARGUMENT,
        "support rejects forged presentation requests without graphics"
    );
    assert(
        @catch(gpu::create_device(&adapter, &forged)) == gpu::INVALID_ARGUMENT,
        "creation rejects forged presentation requests without graphics"
    );
    assert(
        gpu::internal::vk::stub_device_support_query_count() == 0,
        "invalid presentation requests bypass support queries"
    );
    assert(
        gpu::internal::vk::stub_device_temporary_create_count() == 0,
        "invalid presentation requests allocate no device state"
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
    DeviceRequest request = gpu::request_presentation(
        gpu::strict_device_request(),
        &surface,
    )!!;
    Device device = gpu::create_device(&adapter, &request)!!;
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
    DeviceRequest request = gpu::request_presentation(
        gpu::strict_device_request(),
        &first_surface,
    )!!;
    Device device = gpu::create_device(&adapter, &request)!!;
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

fn void strict_only_devices_reject_swapchains_before_backend_work() @test {
    gpu::internal::vk::stub_reset_surface_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Surface surface = create_test_surface(&runtime);
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceRequest request = gpu::strict_device_request();
    Device device = gpu::create_device(&adapter, &request)!!;
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
        "strict-only rejection precedes backend mutation"
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
    DeviceRequest first_request = gpu::request_presentation(
        gpu::strict_device_request(),
        &first_surface,
    )!!;
    DeviceRequest second_request = gpu::request_presentation(
        gpu::strict_device_request(),
        &second_surface,
    )!!;
    Device first = gpu::create_device(&first_adapter, &first_request)!!;
    Device second = gpu::create_device(&second_adapter, &second_request)!!;
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
