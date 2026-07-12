module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

struct WsiDiagnosticCapture {
    uint              calls;
    gpu::DebugMessage last;
}

fn void capture_wsi_diagnostic(gpu::DebugMessage* message, void* user_data) {
    WsiDiagnosticCapture* capture = (WsiDiagnosticCapture*)user_data;
    capture.calls++;
    capture.last = *message;
}

fn void wsi_presentation_gate_diagnostic() @test {
    WsiDiagnosticCapture capture;
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_wsi_diagnostic_gate_test",
        .debug_callback    = &capture_wsi_diagnostic,
        .debug_user_data   = &capture,
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    capture = {};

    gpu::SurfaceDesc surface = { .platform = gpu::PlatformKind.WAYLAND };
    gpu::SwapchainDesc swapchain_desc = { .width = 640, .height = 480 };
    fault gate_fault = @catch(gpu::create_swapchain(&device, &surface, &swapchain_desc));

    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "presentation gate fault is unchanged");
    assert(capture.calls == 1, "presentation gate emits exactly once");
    assert(capture.last.public_fault == gate_fault, "diagnostic preserves the returned fault");
    assert(capture.last.operation.str_view() == "create_swapchain", "diagnostic names create_swapchain");
    assert(
        capture.last.rejected_field.str_view() == "DeviceDesc.enable_presentation",
        "diagnostic names the presentation gate"
    );
    assert(capture.last.resource.kind == gpu::DebugResourceKind.SWAPCHAIN, "diagnostic identifies a swapchain");
    assert(!capture.last.resource.has_public_identity, "unpublished swapchain has no public identity");
}

fn void wsi_backend_result_diagnostic_preserves_mapping() @test {
    WsiDiagnosticCapture capture;
    gpu::vk::VkDeviceState state = {
        .debug_callback  = &capture_wsi_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::vk::WsiDiagnosticContext context = {
        .state      = &state,
        .operation  = "acquire_next_image",
        .debug_name = "main_swapchain",
    };

    fault result_fault = @catch(gpu::vk::check_wsi_backend_result(
        context: &context,
        kind:    gpu::vk::WsiResultKind.ACQUIRE,
        result:  vk::Result.TIMEOUT,
    ));

    assert(result_fault == gpu::WAIT_TIMEOUT, "acquire timeout mapping is unchanged");
    assert(capture.calls == 1, "WSI backend failure emits exactly once");
    assert(capture.last.category.backend, "WSI native result is a backend diagnostic");
    assert(capture.last.operation.str_view() == "acquire_next_image", "WSI result keeps operation context");
    assert(capture.last.validation_id_number == (int)vk::Result.TIMEOUT, "WSI result keeps raw VkResult");
    assert(capture.last.backend_text.str_view() == "Vulkan image acquire timed out", "WSI result has stable text");
    assert(!capture.last.resource.has_public_identity, "unresolved WSI result has no public identity");
}

fn void wsi_stale_handle_diagnostic_has_no_identity() @test {
    WsiDiagnosticCapture capture;
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_wsi_stale_handle_test",
        .debug_callback    = &capture_wsi_diagnostic,
        .debug_user_data   = &capture,
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    capture = {};

    gpu::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault query_fault = @catch(gpu::get_swapchain_info(&device, bogus));

    assert(query_fault == gpu::INVALID_HANDLE, "stale handle fault is unchanged");
    assert(capture.calls == 1, "stale swapchain lookup emits exactly once");
    assert(capture.last.public_fault == query_fault, "stale diagnostic preserves the fault");
    assert(capture.last.operation.str_view() == "get_swapchain_info", "stale lookup names the operation");
    assert(capture.last.rejected_field.str_view() == "SwapchainHandle", "stale lookup names the handle");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.SWAPCHAIN, "stale lookup keeps resource kind");
    assert(!capture.last.resource.has_public_identity, "unverified handle bits are not identity");

    capture = {};
    assert(
        @catch(gpu::destroy_swapchain(&device, bogus)) == gpu::INVALID_HANDLE,
        "destroy preserves stale-handle fault"
    );
    assert(
        capture.calls == 1 && capture.last.operation.str_view() == "destroy_swapchain",
        "destroy reports exactly once"
    );

    capture = {};
    assert(
        @catch(gpu::resize_swapchain(&device, bogus, 640, 480)) == gpu::INVALID_HANDLE,
        "resize preserves stale-handle fault"
    );
    assert(
        capture.calls == 1 && capture.last.operation.str_view() == "resize_swapchain",
        "resize reports exactly once"
    );

    capture = {};
    assert(
        @catch(gpu::acquire_next_image(&device, bogus)) == gpu::INVALID_HANDLE,
        "acquire preserves stale-handle fault"
    );
    assert(
        capture.calls == 1 && capture.last.operation.str_view() == "acquire_next_image",
        "acquire reports exactly once"
    );

    capture = {};
    gpu::PresentDesc present_desc = { .swapchain = bogus };
    assert(
        @catch(gpu::present(&device, &present_desc)) == gpu::INVALID_HANDLE,
        "present preserves stale-handle fault"
    );
    assert(capture.calls == 1 && capture.last.operation.str_view() == "present", "present reports exactly once");

    capture = {};
    assert(
        @catch(gpu::get_present_mode_support(&device, bogus)) == gpu::INVALID_HANDLE,
        "support query preserves stale-handle fault"
    );
    assert(
        capture.calls == 1 && capture.last.operation.str_view() == "get_present_mode_support",
        "support query reports exactly once"
    );
}

fn void expect_wsi_result(
    gpu::vk::WsiResultKind kind,
    vk::Result result,
    fault expected,
    ZString operation,
) {
    WsiDiagnosticCapture capture;
    gpu::vk::VkDeviceState state = {
        .debug_callback  = &capture_wsi_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::vk::WsiDiagnosticContext context = {
        .state               = &state,
        .operation           = operation,
        .has_public_identity = true,
        .index               = 2,
        .generation          = 9,
        .debug_name          = "resolved_swapchain",
    };

    fault actual = @catch(gpu::vk::check_wsi_backend_result(
        context: &context,
        kind:    kind,
        result:  result,
    ));

    assert(actual == expected, "specialized WSI mapping preserves the public fault");
    assert(capture.calls == 1, "specialized WSI mapping emits exactly once");
    assert(capture.last.category.backend && !capture.last.category.public_contract,
        "native WSI failure is backend-only");
    assert(capture.last.operation.str_view() == operation.str_view(), "native WSI failure preserves operation");
    assert(capture.last.validation_id_number == (int)result, "native WSI failure preserves raw VkResult");
    assert(capture.last.resource.has_public_identity, "resolved swapchain result carries identity");
    assert(capture.last.resource.index == 2 && capture.last.resource.generation == 9,
        "resolved swapchain identity is exact");
    assert(capture.last.rejected_field == null && capture.last.invariant == null,
        "backend WSI result has no rejected caller field");
}

fn void all_specialized_wsi_results_use_one_mapping_seam() @test {
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.SURFACE,
        result:    vk::Result.ERROR_SURFACE_LOST_KHR,
        expected:  gpu::SURFACE_LOST,
        operation: "create_swapchain",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.SURFACE,
        result:    vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR,
        expected:  gpu::INVALID_RESOURCE_STATE,
        operation: "create_swapchain",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.SURFACE_ENUMERATION,
        result:    vk::Result.INCOMPLETE,
        expected:  gpu::BACKEND_ERROR,
        operation: "get_present_mode_support",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.SWAPCHAIN_CREATE,
        result:    vk::Result.ERROR_OUT_OF_DATE_KHR,
        expected:  gpu::SWAPCHAIN_OUT_OF_DATE,
        operation: "resize_swapchain",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.SWAPCHAIN_CREATE,
        result:    vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
        expected:  gpu::UNSUPPORTED_FEATURE,
        operation: "create_swapchain",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.ENUMERATION,
        result:    vk::Result.INCOMPLETE,
        expected:  gpu::BACKEND_ERROR,
        operation: "create_swapchain",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.ACQUIRE,
        result:    vk::Result.TIMEOUT,
        expected:  gpu::WAIT_TIMEOUT,
        operation: "acquire_next_image",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.ACQUIRE,
        result:    vk::Result.NOT_READY,
        expected:  gpu::WAIT_TIMEOUT,
        operation: "acquire_next_image",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.ACQUIRE,
        result:    vk::Result.ERROR_OUT_OF_DATE_KHR,
        expected:  gpu::SWAPCHAIN_OUT_OF_DATE,
        operation: "acquire_next_image",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.PRESENT,
        result:    vk::Result.ERROR_SURFACE_LOST_KHR,
        expected:  gpu::SURFACE_LOST,
        operation: "present",
    );
    expect_wsi_result(
        kind:      gpu::vk::WsiResultKind.PRESENT,
        result:    vk::Result.ERROR_OUT_OF_DATE_KHR,
        expected:  gpu::SWAPCHAIN_OUT_OF_DATE,
        operation: "present",
    );
}

fn void wrapped_image_failure_is_backend_with_swapchain_identity() @test {
    WsiDiagnosticCapture capture;
    gpu::vk::VkDeviceState state = {
        .debug_callback  = &capture_wsi_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::vk::WsiDiagnosticContext context = {
        .state               = &state,
        .operation           = "present",
        .has_public_identity = true,
        .index               = 4,
        .generation          = 6,
        .debug_name          = "main_swapchain",
    };
    gpu::TextureHandle missing = { .index = 7, .generation = 3 };

    fault missing_fault = @catch(gpu::vk::resolve_swapchain_texture_for_operation(&context, missing));

    assert(missing_fault == gpu::INVALID_HANDLE, "wrapped-image invariant preserves internal lookup fault");
    assert(capture.calls == 1, "wrapped-image invariant emits exactly once");
    assert(capture.last.category.backend && !capture.last.category.public_contract,
        "wrapped-image invariant is backend-only");
    assert(capture.last.resource.has_public_identity, "wrapped-image failure carries resolved swapchain identity");
    assert(capture.last.resource.index == 4 && capture.last.resource.generation == 6,
        "wrapped-image failure carries exact swapchain identity");
    assert(capture.last.rejected_field == null && capture.last.invariant == null,
        "wrapped-image failure does not blame a caller field");
}

fn void wsi_lifecycle_and_layout_diagnostics_are_contract_failures() @test {
    WsiDiagnosticCapture capture;
    gpu::vk::VkDeviceState state = {
        .debug_callback  = &capture_wsi_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::vk::WsiDiagnosticContext context = {
        .state               = &state,
        .operation           = "acquire_next_image",
        .has_public_identity = true,
        .index               = 1,
        .generation          = 5,
    };
    gpu::vk::SwapchainSlot slot = { .info = { .dormant = true } };

    assert(@catch(gpu::vk::check_swapchain_acquire_lifecycle(&context, &slot)) == gpu::SWAPCHAIN_OUT_OF_DATE,
        "dormant acquire preserves OUT_OF_DATE");
    assert(capture.calls == 1 && capture.last.category.public_contract,
        "dormant acquire reports one contract diagnostic");

    capture = {};
    slot.info.dormant = false;
    slot.swapchain = (vk::SwapchainKHR)1;
    slot.acquire_pending = true;
    assert(@catch(gpu::vk::check_swapchain_acquire_lifecycle(&context, &slot)) == gpu::INVALID_RESOURCE_STATE,
        "double acquire preserves INVALID_RESOURCE_STATE");
    assert(capture.calls == 1 && capture.last.rejected_field.str_view() == "swapchain acquire lifecycle",
        "double acquire reports its lifecycle field");

    capture = {};
    context.operation = "present";
    slot.acquire_consumed = false;
    assert(@catch(gpu::vk::check_swapchain_present_lifecycle(&context, &slot)) == gpu::INVALID_RESOURCE_STATE,
        "unconsumed present preserves INVALID_RESOURCE_STATE");
    assert(capture.calls == 1 && capture.last.category.public_contract,
        "unconsumed present reports one contract diagnostic");

    capture = {};
    gpu::vk::TextureSlot image_slot;
    assert(@catch(gpu::vk::check_swapchain_present_layout(&context, &image_slot)) == gpu::INVALID_RESOURCE_STATE,
        "wrong present layout preserves INVALID_RESOURCE_STATE");
    assert(capture.calls == 1 && capture.last.rejected_field.str_view() == "acquired image layout",
        "wrong layout reports its exact field");

    capture = {};
    gpu::vk::set_texture_layout(&image_slot, gpu::TextureLayout.PRESENT);
    gpu::vk::check_swapchain_present_layout(&context, &image_slot)!!;
    assert(capture.calls == 0, "valid present layout emits nothing");
}

fn void capability_failures_are_backend_only_and_success_is_silent() @test {
    WsiDiagnosticCapture capture;
    gpu::vk::VkDeviceState state = {
        .debug_callback  = &capture_wsi_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::vk::WsiDiagnosticContext context = {
        .state      = &state,
        .operation  = "create_swapchain",
        .debug_name = "pending_swapchain",
    };

    fault capability_fault = gpu::vk::report_wsi_backend_failure(
        context:      &context,
        public_fault: gpu::UNSUPPORTED_FEATURE,
        backend_text: "selected graphics queue family cannot present to the surface",
    );
    assert(capability_fault == gpu::UNSUPPORTED_FEATURE, "capability fault is unchanged");
    assert(capture.calls == 1, "capability failure emits exactly once");
    assert(capture.last.category.backend && !capture.last.category.public_contract,
        "capability failure is backend-only");
    assert(capture.last.rejected_field == null && capture.last.invariant == null,
        "capability failure does not blame a caller field");
    assert(!capture.last.resource.has_public_identity, "unpublished capability failure has no identity");

    capture = {};
    gpu::vk::check_wsi_backend_result(
        context: &context,
        kind:    gpu::vk::WsiResultKind.ACQUIRE,
        result:  vk::Result.SUCCESS,
    )!!;
    gpu::vk::check_wsi_backend_result(
        context: &context,
        kind:    gpu::vk::WsiResultKind.PRESENT,
        result:  vk::Result.SUBOPTIMAL_KHR,
    )!!;
    assert(capture.calls == 0, "successful and suboptimal WSI results emit nothing");
}
