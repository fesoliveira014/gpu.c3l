module gpu_test;

import gpu;
import gpu::vk @public;
import std::core::mem;
import vma;
import vk;

uint captured_present_calls;
uint captured_fence_status_calls;
uint captured_fence_reset_calls;
vk::Image borrowed_view_backing_image;
uint borrowed_view_swapchain_creates;
uint borrowed_view_swapchain_destroys;
uint borrowed_view_acquires;

fn vk::Result borrowed_view_surface_capabilities(
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    vk::SurfaceCapabilitiesKHR* capabilities,
) {
    (void)physical_device;
    (void)surface;
    *capabilities = {
        .min_image_count       = 1,
        .max_image_count       = 1,
        .current_extent        = { .width = uint::max, .height = uint::max },
        .min_image_extent      = { .width = 1, .height = 1 },
        .max_image_extent      = { .width = 64, .height = 64 },
        .current_transform     = vk::SurfaceTransformFlagBitsKHR.IDENTITY_BIT_KHR,
        .supported_usage_flags = vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT,
    };
    return vk::Result.SUCCESS;
}

fn vk::Result borrowed_view_surface_formats(
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    uint* count,
    vk::SurfaceFormatKHR* formats,
) {
    (void)physical_device;
    (void)surface;
    *count = 1;
    if (formats != null) {
        formats[0] = {
            .format      = vk::Format.B8G8R8A8_UNORM,
            .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR,
        };
    }
    return vk::Result.SUCCESS;
}

fn vk::Result borrowed_view_present_modes(
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    uint* count,
    vk::PresentModeKHR* modes,
) {
    (void)physical_device;
    (void)surface;
    *count = 1;
    if (modes != null) modes[0] = vk::PresentModeKHR.FIFO_KHR;
    return vk::Result.SUCCESS;
}

fn vk::Result borrowed_view_create_swapchain(
    vk::Device device,
    vk::SwapchainCreateInfoKHR* create_info,
    vk::AllocationCallbacks* allocator,
    vk::SwapchainKHR* swapchain,
) {
    (void)device;
    (void)create_info;
    (void)allocator;
    borrowed_view_swapchain_creates++;
    *swapchain = (vk::SwapchainKHR)(uptr)borrowed_view_swapchain_creates;
    return vk::Result.SUCCESS;
}

fn void borrowed_view_destroy_swapchain(
    vk::Device device,
    vk::SwapchainKHR swapchain,
    vk::AllocationCallbacks* allocator,
) {
    (void)device;
    (void)swapchain;
    (void)allocator;
    borrowed_view_swapchain_destroys++;
}

fn vk::Result borrowed_view_get_swapchain_images(
    vk::Device device,
    vk::SwapchainKHR swapchain,
    uint* count,
    vk::Image* images,
) {
    (void)device;
    (void)swapchain;
    *count = 1;
    if (images != null) images[0] = borrowed_view_backing_image;
    return vk::Result.SUCCESS;
}

fn vk::Result borrowed_view_acquire(
    vk::Device device,
    vk::SwapchainKHR swapchain,
    ulong timeout,
    vk::Semaphore semaphore,
    vk::Fence fence,
    uint* image_index,
) {
    (void)device;
    (void)swapchain;
    (void)timeout;
    (void)semaphore;
    (void)fence;
    borrowed_view_acquires++;
    *image_index = 0;
    return vk::Result.SUCCESS;
}

fn vk::Result capture_present(
    vk::Queue queue,
    vk::PresentInfoKHR* info,
) {
    (void)queue;
    assert(info.swapchain_count == 1, "present seam receives one swapchain");
    assert(info.wait_semaphore_count == 1,
        "present seam receives one render wait");
    assert(info.p_next != null,
        "present attaches private presentation completion");
    vk::SwapchainPresentFenceInfoEXT* fence_info =
        (vk::SwapchainPresentFenceInfoEXT*)info.p_next;
    assert(fence_info.swapchain_count == 1
        && fence_info.p_fences[0] != null,
        "present completion covers the submitted swapchain");
    captured_present_calls++;
    return vk::Result.SUCCESS;
}

fn vk::Result capture_fence_ready(vk::Device device, vk::Fence fence) {
    (void)device;
    (void)fence;
    captured_fence_status_calls++;
    return vk::Result.SUCCESS;
}

fn vk::Result capture_fence_pending(vk::Device device, vk::Fence fence) {
    (void)device;
    (void)fence;
    captured_fence_status_calls++;
    return vk::Result.NOT_READY;
}

fn vk::Result capture_fence_reset(
    vk::Device device,
    uint fence_count,
    vk::Fence* fences,
) {
    (void)device;
    assert(fence_count == 1 && fences[0] != null,
        "present resets exactly one valid fence");
    captured_fence_reset_calls++;
    return vk::Result.SUCCESS;
}

// Pure swapchain selection logic over synthetic capability lists, plus the
// headless presentation gate. Windowed behavior is covered by the
// hello_triangle_sdl smoke run.

fn void surface_format_selection() @test {
    vk::SurfaceFormatKHR[2] bgra_only = {
        { .format = vk::Format.B8G8R8A8_UNORM, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
        { .format = vk::Format.B8G8R8A8_SRGB, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
    };

    gpu::Format preferred = gpu::vk::select_surface_format(bgra_only[..], gpu::Format.BGRA8_SRGB, srgb: true)!!;
    assert(preferred == gpu::Format.BGRA8_SRGB, "preferred format wins when available");

    gpu::Format fallback = gpu::vk::select_surface_format(bgra_only[..], gpu::Format.RGBA8_UNORM, srgb: false)!!;
    assert(fallback == gpu::Format.BGRA8_UNORM, "unavailable preferred falls back to BGRA unorm");

    gpu::Format srgb_pick = gpu::vk::select_surface_format(bgra_only[..], gpu::Format.UNDEFINED, srgb: true)!!;
    assert(srgb_pick == gpu::Format.BGRA8_SRGB, "srgb flag selects the srgb flavor");

    vk::SurfaceFormatKHR[1] alien = {
        { .format = vk::Format.R16G16B16A16_SFLOAT, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
    };
    fault none = @catch(gpu::vk::select_surface_format(alien[..], gpu::Format.UNDEFINED, srgb: false));
    assert(none == gpu::UNSUPPORTED_FEATURE, "no swapchain-usable format faults UNSUPPORTED_FEATURE");
}

fn void present_mode_selection() @test {
    vk::PresentModeKHR[2] fifo_mailbox = { vk::PresentModeKHR.FIFO_KHR, vk::PresentModeKHR.MAILBOX_KHR };
    assert(gpu::vk::select_present_mode(fifo_mailbox[..], gpu::PresentMode.MAILBOX) == vk::PresentModeKHR.MAILBOX_KHR,
        "supported mailbox is honored");

    vk::PresentModeKHR[1] fifo_only = { vk::PresentModeKHR.FIFO_KHR };
    assert(gpu::vk::select_present_mode(fifo_only[..], gpu::PresentMode.MAILBOX) == vk::PresentModeKHR.FIFO_KHR,
        "unsupported mailbox falls back to FIFO");
    assert(gpu::vk::select_present_mode(fifo_only[..], gpu::PresentMode.IMMEDIATE) == vk::PresentModeKHR.FIFO_KHR,
        "unsupported immediate falls back to FIFO");
    assert(gpu::vk::select_present_mode(fifo_mailbox[..], gpu::PresentMode.FIFO) == vk::PresentModeKHR.FIFO_KHR,
        "FIFO requested is FIFO");
}

fn void common_surface_lost_result_mapping() @test {
    fault surface_lost = @catch(gpu::vk::check(vk::Result.ERROR_SURFACE_LOST_KHR));
    assert(surface_lost == gpu::SURFACE_LOST, "common mapper preserves surface loss");
}

fn void acquire_result_mapping() @test {
    gpu::vk::check_acquire_result(vk::Result.SUCCESS)!!;
    gpu::vk::check_acquire_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault timeout = @catch(gpu::vk::check_acquire_result(vk::Result.TIMEOUT));
    fault not_ready = @catch(gpu::vk::check_acquire_result(vk::Result.NOT_READY));
    fault out_of_date = @catch(gpu::vk::check_acquire_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::vk::check_acquire_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::vk::check_acquire_result(vk::Result.ERROR_DEVICE_LOST));

    assert(timeout == gpu::WAIT_TIMEOUT, "acquire timeout is retryable");
    assert(not_ready == gpu::WAIT_TIMEOUT, "acquire not-ready is retryable");
    assert(out_of_date == gpu::SWAPCHAIN_OUT_OF_DATE, "out-of-date requires swapchain resize");
    assert(surface_lost == gpu::SURFACE_LOST, "surface loss remains distinct");
    assert(device_lost == gpu::DEVICE_LOST, "unrelated results delegate to the common mapper");
}

fn void present_result_mapping() @test {
    gpu::vk::check_present_result(vk::Result.SUCCESS)!!;
    gpu::vk::check_present_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault out_of_date = @catch(gpu::vk::check_present_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::vk::check_present_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::vk::check_present_result(vk::Result.ERROR_DEVICE_LOST));

    assert(out_of_date == gpu::SWAPCHAIN_OUT_OF_DATE, "out-of-date requires swapchain resize");
    assert(surface_lost == gpu::SURFACE_LOST, "surface loss requires surface recreation");
    assert(device_lost == gpu::DEVICE_LOST, "unrelated results delegate to the common mapper");
}

fn void image_count_clamp() @test {
    assert(gpu::vk::clamp_image_count(0, 2, 8) == 3, "zero requests min + 1");
    assert(gpu::vk::clamp_image_count(1, 2, 8) == 2, "below min clamps up");
    assert(gpu::vk::clamp_image_count(16, 2, 8) == 8, "above max clamps down");
    assert(gpu::vk::clamp_image_count(16, 2, 0) == 16, "max 0 means unbounded");
}

fn void extent_clamp_rules() @test {
    vk::SurfaceCapabilitiesKHR fixed = {
        .current_extent   = { .width = 640, .height = 480 },
        .min_image_extent = { .width = 1, .height = 1 },
        .max_image_extent = { .width = 4096, .height = 4096 },
    };
    vk::Extent2D from_fixed = gpu::vk::clamp_extent(&fixed, 1024, 768);
    assert(from_fixed.width == 640 && from_fixed.height == 480, "fixed currentExtent wins over the request");

    vk::SurfaceCapabilitiesKHR app_decides = {
        .current_extent   = { .width = uint::max, .height = uint::max },
        .min_image_extent = { .width = 64, .height = 64 },
        .max_image_extent = { .width = 2048, .height = 2048 },
    };
    vk::Extent2D clamped = gpu::vk::clamp_extent(&app_decides, 4096, 32);
    assert(clamped.width == 2048 && clamped.height == 64, "request clamps to min/max when the app decides");
}

fn void swapchain_requires_a_presentation_request() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_swapchain_gate_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Surface surface;
    gpu::SwapchainDesc sc_desc = { .width = 640, .height = 480 };
    fault gate_fault = @catch(gpu::vk::vk_create_swapchain(&device, &surface, &sc_desc));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "swapchain without a presentation request faults");
}

fn void swapchain_submit_guard_lifecycle() @test {
    gpu::QueueRoles graphics = { .graphics };
    fault not_acquired = @catch(gpu::vk::swapchain_submit_guard(
        acquire_pending:      false,
        readiness_consumed:   false,
        expected_acquisition: 0,
        provided_acquisition: 1,
        roles:                graphics,
    ));
    assert(not_acquired == gpu::INVALID_RESOURCE_STATE, "submit with no pending acquire faults INVALID_RESOURCE_STATE");

    gpu::vk::swapchain_submit_guard(
        acquire_pending:      true,
        readiness_consumed:   false,
        expected_acquisition: 7,
        provided_acquisition: 7,
        roles:                graphics,
    )!!;

    fault double_submit = @catch(gpu::vk::swapchain_submit_guard(
        acquire_pending:      true,
        readiness_consumed:   true,
        expected_acquisition: 7,
        provided_acquisition: 7,
        roles:                graphics,
    ));
    assert(double_submit == gpu::INVALID_RESOURCE_STATE, "second readiness submit faults INVALID_RESOURCE_STATE");

    fault stale_readiness = @catch(gpu::vk::swapchain_submit_guard(
        acquire_pending:      true,
        readiness_consumed:   false,
        expected_acquisition: 7,
        provided_acquisition: 6,
        roles:                graphics,
    ));
    assert(stale_readiness == gpu::INVALID_RESOURCE_STATE, "stale readiness faults INVALID_RESOURCE_STATE");

    fault future_readiness = @catch(gpu::vk::swapchain_submit_guard(
        acquire_pending:      true,
        readiness_consumed:   false,
        expected_acquisition: 7,
        provided_acquisition: 8,
        roles:                graphics,
    ));
    assert(future_readiness == gpu::INVALID_RESOURCE_STATE,
        "future readiness faults INVALID_RESOURCE_STATE");

    gpu::QueueRoles compute = { .compute };
    fault compute_coupled = @catch(gpu::vk::swapchain_submit_guard(
        acquire_pending:      true,
        readiness_consumed:   false,
        expected_acquisition: 7,
        provided_acquisition: 7,
        roles:                compute,
    ));
    assert(compute_coupled == gpu::INVALID_ARGUMENT, "COMPUTE readiness submit faults INVALID_ARGUMENT");
}

fn void swapchain_present_guard_lifecycle() @test {
    gpu::Device device = { .generation = 1 };
    gpu::CompletionPoint expected = { .device = device, .payload = 17 };
    gpu::CompletionPoint other = { .device = device, .payload = 18 };
    fault not_acquired = @catch(gpu::vk::swapchain_present_guard(
        acquire_pending:      false,
        readiness_consumed:   false,
        expected_acquisition: 0,
        provided_acquisition: 1,
        expected_completion:  expected,
        provided_completion:  expected,
    ));
    assert(not_acquired == gpu::INVALID_RESOURCE_STATE, "present with no pending acquire faults INVALID_RESOURCE_STATE");

    fault not_consumed = @catch(gpu::vk::swapchain_present_guard(
        acquire_pending:      true,
        readiness_consumed:   false,
        expected_acquisition: 7,
        provided_acquisition: 7,
        expected_completion:  expected,
        provided_completion:  expected,
    ));
    assert(not_consumed == gpu::INVALID_RESOURCE_STATE, "present without a readiness submit faults INVALID_RESOURCE_STATE");

    gpu::vk::swapchain_present_guard(
        acquire_pending:      true,
        readiness_consumed:   true,
        expected_acquisition: 7,
        provided_acquisition: 7,
        expected_completion:  expected,
        provided_completion:  expected,
    )!!;

    fault stale_image = @catch(gpu::vk::swapchain_present_guard(
        acquire_pending:      true,
        readiness_consumed:   true,
        expected_acquisition: 7,
        provided_acquisition: 6,
        expected_completion:  expected,
        provided_completion:  expected,
    ));
    assert(stale_image == gpu::INVALID_RESOURCE_STATE, "stale acquired image faults INVALID_RESOURCE_STATE");

    fault future_image = @catch(gpu::vk::swapchain_present_guard(
        acquire_pending:      true,
        readiness_consumed:   true,
        expected_acquisition: 7,
        provided_acquisition: 8,
        expected_completion:  expected,
        provided_completion:  expected,
    ));
    assert(future_image == gpu::INVALID_RESOURCE_STATE,
        "future acquired image faults INVALID_RESOURCE_STATE");

    fault wrong_completion = @catch(gpu::vk::swapchain_present_guard(
        acquire_pending:      true,
        readiness_consumed:   true,
        expected_acquisition: 7,
        provided_acquisition: 7,
        expected_completion:  expected,
        provided_completion:  other,
    ));
    assert(wrong_completion == gpu::INVALID_ARGUMENT, "unrelated completion faults INVALID_ARGUMENT");
}

fn void present_path_guards_precede_native_mutation() @test {
    WsiDiagnosticCapture capture;
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_present_guard_test",
        .debug_callback    = &capture_wsi_diagnostic,
        .debug_user_data   = &capture,
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    state.presentation.queue_mutex.init()!!;

    gpu::vk::SwapchainCell[gpu::MAX_SWAPCHAINS] cells;
    state.presentation.swapchains.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.SWAPCHAIN,
    );
    state.presentation.swapchains.slots = cells[..];
    defer state.presentation.swapchains.slots = {};

    gpu::vk::TextureSlot texture;
    gpu::vk::mark_texture_presented(&texture);
    gpu::TextureHandle texture_handle = state.textures.alloc(texture)!!;
    defer state.textures.free(texture_handle)!!;

    gpu::CompletionPoint expected = {
        .device  = device,
        .payload = 1ull << gpu::COMPLETION_QUEUE_BITS,
    };
    gpu::vk::SwapchainSlot pending = {
        .swapchain          = (vk::SwapchainKHR)1,
        .image_count        = 1,
        .acquire_pending    = true,
        .readiness_consumed = true,
        .acquisition        = 7,
        .pending_image      = 0,
        .render_completion  = expected,
    };
    pending.textures[0] = texture_handle;
    pending.present_semaphores[0] = (vk::Semaphore)2;
    pending.present_fences[0] = (vk::Fence)3;
    gpu::SwapchainHandle handle =
        state.presentation.swapchains.alloc(pending)!!;
    gpu::SwapchainReadiness readiness = gpu::vk::make_swapchain_readiness(
        device,
        handle,
        7,
    )!!;
    gpu::AcquiredImage image = {
        .texture   = texture_handle,
        .readiness = readiness,
        .index     = 0,
    };
    gpu::AcquiredImage replay = image;
    gpu::CompletionPoint unrelated = expected;
    unrelated.payload++;
    captured_present_calls = 0;
    capture = {};

    fault completion_fault = @catch(gpu::vk::vk_present_with(
        device:            &device,
        image:             &image,
        render_completion: unrelated,
        present_fn:        &capture_present,
    ));
    assert(completion_fault == gpu::INVALID_ARGUMENT,
        "present path rejects an unrelated completion point");
    assert(captured_present_calls == 0,
        "completion rejection precedes native present");
    assert(capture.calls == 1 && capture.last.category.public_contract,
        "completion rejection emits one contract diagnostic");

    capture = {};
    gpu::AcquiredImage wrong_image = image;
    wrong_image.index = 1;
    fault image_fault = @catch(gpu::vk::vk_present_with(
        device:            &device,
        image:             &wrong_image,
        render_completion: expected,
        present_fn:        &capture_present,
    ));
    assert(image_fault == gpu::INVALID_ARGUMENT,
        "present path rejects the wrong acquired image");
    assert(captured_present_calls == 0,
        "image rejection precedes native present");
    assert(capture.calls == 1 && capture.last.category.public_contract,
        "image rejection emits one contract diagnostic");

    capture = {};
    gpu::vk::vk_present_with(
        device:            &device,
        image:             &image,
        render_completion: expected,
        present_fn:        &capture_present,
    )!!;
    assert(captured_present_calls == 1,
        "valid present reaches the native seam exactly once");
    assert(state.presentation.swapchains.get(handle)!!.present_fence_pending[0],
        "an enqueued present tracks its private completion fence");
    assert(capture.calls == 0, "valid present emits no diagnostic");

    fault replay_fault = @catch(gpu::vk::vk_present_with(
        device:            &device,
        image:             &replay,
        render_completion: expected,
        present_fn:        &capture_present,
    ));
    assert(replay_fault == gpu::INVALID_RESOURCE_STATE,
        "present replay faults INVALID_RESOURCE_STATE");
    assert(captured_present_calls == 1,
        "present replay reaches no second native mutation");
    assert(capture.calls == 1 && capture.last.category.public_contract,
        "present replay emits one contract diagnostic");
}

fn void present_oom_preserves_the_acquisition_for_retry() @test {
    gpu::vk::SwapchainSlot slot = {
        .acquire_pending    = true,
        .readiness_consumed = true,
        .acquisition        = 9,
        .pending_image      = 2,
    };
    gpu::AcquiredImage image = {
        .texture   = { .index = 2, .generation = 1 },
        .readiness = {
            .device  = { .generation = 1 },
            .payload = 9,
        },
        .index = 2,
    };

    gpu::vk::commit_present_result(
        &slot,
        &image,
        vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    );
    assert(slot.acquire_pending && slot.readiness_consumed,
        "host OOM preserves the pending acquisition");
    assert(image.readiness.is_valid(), "host OOM preserves the acquired image");

    gpu::vk::commit_present_result(
        &slot,
        &image,
        vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    );
    assert(slot.acquire_pending && image.readiness.is_valid(),
        "device OOM preserves the acquired image");

    gpu::vk::commit_present_result(
        &slot,
        &image,
        vk::Result.ERROR_OUT_OF_DATE_KHR,
    );
    assert(!slot.acquire_pending && !slot.readiness_consumed,
        "an enqueued out-of-date present consumes the acquisition");
    assert(!image.readiness.is_valid(), "an enqueued present consumes the image");
}

fn bool? poll_only_second_retirement(
    gpu::Device* device,
    gpu::CompletionPoint point,
) {
    (void)device;
    return gpu::completion_point_sequence(point) == 2;
}

fn bool? poll_no_retirement(
    gpu::Device* device,
    gpu::CompletionPoint point,
) {
    (void)device;
    (void)point;
    return false;
}

fn void acquire_semaphores_retire_before_reuse() @test {
    gpu::Device device = { .generation = 1 };
    assert(gpu::vk::SWAPCHAIN_ACQUIRE_SLOT_COUNT == 2,
        "the production acquire ring remains fixed at two slots");
    gpu::CompletionPoint[gpu::vk::SWAPCHAIN_ACQUIRE_SLOT_COUNT] retirements = {
        { .device = device, .payload = 1ull << gpu::COMPLETION_QUEUE_BITS },
        { .device = device, .payload = 2ull << gpu::COMPLETION_QUEUE_BITS },
    };
    gpu::vk::SwapchainSlot slot = {
        .acquire_retirements   = retirements[..],
        .next_acquire_semaphore = 0,
    };

    uint selected = gpu::vk::select_acquire_semaphore_with(
        &device,
        &slot,
        &poll_only_second_retirement,
    )!!;
    assert(selected == 1,
        "selection skips pending waits and finds a retired semaphore");

    retirements[0] = {};
    selected = gpu::vk::select_acquire_semaphore_with(
        &device,
        &slot,
        &poll_no_retirement,
    )!!;
    assert(selected == 0, "a never-used semaphore is immediately available");

    slot.pending_acquire_semaphore_index = 0;
    gpu::CompletionPoint submitted = { .device = device, .payload = 3ull << gpu::COMPLETION_QUEUE_BITS };
    gpu::vk::commit_swapchain_submit(&slot, submitted);
    assert(slot.readiness_consumed && slot.render_completion == submitted,
        "successful submission commits readiness and render completion");
    assert(retirements[0] == submitted,
        "successful submission retires the exact acquire semaphore");

    fault busy = @catch(gpu::vk::select_acquire_semaphore_with(
        &device,
        &slot,
        &poll_no_retirement,
    ));
    assert(busy == gpu::WAIT_TIMEOUT,
        "acquire stays retryable while every semaphore wait is pending");
}

fn void swapchain_release_is_immediate_and_retryable() @test {
    gpu::vk::swapchain_release_guard(
        acquire_pending:       false,
        images_referenced:     false,
        presentations_pending: false,
    )!!;

    fault acquired = @catch(gpu::vk::swapchain_release_guard(
        acquire_pending:       true,
        images_referenced:     false,
        presentations_pending: false,
    ));
    assert(acquired == gpu::INVALID_RESOURCE_STATE,
        "release rejects an acquired image instead of consuming it");

    fault referenced = @catch(gpu::vk::swapchain_release_guard(
        acquire_pending:       false,
        images_referenced:     true,
        presentations_pending: false,
    ));
    assert(referenced == gpu::RESOURCE_IN_USE,
        "release rejects live command or view references");

    fault presenting = @catch(gpu::vk::swapchain_release_guard(
        acquire_pending:       false,
        images_referenced:     false,
        presentations_pending: true,
    ));
    assert(presenting == gpu::RESOURCE_IN_USE,
        "release rejects unfinished presentation without waiting");
}

fn void swapchain_release_detects_shared_resource_references() @test {
    gpu::vk::VkDeviceState state;
    gpu::vk::TextureCell[1] textures;
    gpu::vk::AttachmentViewCell[1] attachment_views;
    gpu::vk::TextureDescriptorCell[1] descriptors;
    state.textures.owner = 41;
    state.textures.slots = textures[..];
    state.attachment_views.owner = 43;
    state.attachment_views.slots = attachment_views[..];
    state.strict_heap.descriptors.texture_cells = descriptors[..];

    gpu::TextureHandle texture = state.textures.alloc({})!!;
    gpu::AttachmentViewHandle attachment_view = state.attachment_views.alloc({
        .texture         = texture,
        .swapchain_owned = true,
    })!!;
    gpu::vk::SwapchainSlot slot = { .image_count = 1 };
    slot.textures[0] = texture;
    slot.attachment_views[0] = attachment_view;
    assert(!gpu::vk::swapchain_images_referenced_locked(
        &state,
        &slot,
    )!!, "an unused wrapped texture is releasable");

    state.textures.get(texture)!!.validation_references.add(
        1,
        AtomicOrdering.RELAXED,
    );
    assert(gpu::vk::swapchain_images_referenced_locked(
        &state,
        &slot,
    )!!, "recorded or submitted texture use blocks release");
    state.textures.get(texture)!!.validation_references.sub(
        1,
        AtomicOrdering.RELAXED,
    );

    state.attachment_views.get(attachment_view)!!.validation_references.add(
        1,
        AtomicOrdering.RELAXED,
    );
    assert(gpu::vk::swapchain_images_referenced_locked(
        &state,
        &slot,
    )!!, "recorded or submitted attachment use blocks release");
    state.attachment_views.get(attachment_view)!!.validation_references.sub(
        1,
        AtomicOrdering.RELAXED,
    );

    descriptors[0] = { .owner = texture, .used = true };
    state.strict_heap.descriptors.texture_next_new = 1;
    assert(gpu::vk::swapchain_images_referenced_locked(
        &state,
        &slot,
    )!!, "a shared texture descriptor blocks release");

    descriptors[0].used = false;
    state.strict_heap.descriptors.texture_next_new = 0;
    state.attachment_views.free(attachment_view)!!;
    assert(!gpu::vk::swapchain_images_referenced_locked(
        &state,
        &slot,
    )!!, "a missing internal attachment-view slot is ignored defensively");
}

fn void borrowed_swapchain_view_lifecycle_reaches_resize_invalidation() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "borrowed_swapchain_view_lifecycle_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::vk::VkInstanceSurfaceDispatch original_surface_dispatch =
        state.instance_dispatch.surface;
    gpu::vk::VkDevicePresentationDispatch original_presentation_dispatch =
        state.device_dispatch.presentation;
    gpu::vk::SwapchainTable original_swapchains =
        state.presentation.swapchains;
    defer {
        state.instance_dispatch.surface = original_surface_dispatch;
        state.device_dispatch.presentation = original_presentation_dispatch;
        state.presentation.swapchains = original_swapchains;
    }

    gpu::TextureDesc backing_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.BGRA8_UNORM,
        .usage     = { .color_attach },
    };
    uint[1] queue_families = { state.graphics_family };
    vk::ImageCreateInfo image_info = gpu::vk::make_texture_create_info(
        desc:           &backing_desc,
        queue_families: queue_families[..],
    );
    vma::AllocationCreateInfo allocation_info = {
        .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE,
    };
    vma::ImageAllocation backing = state.allocator.try_create_image(
        &image_info,
        &allocation_info,
    )!!;
    defer state.allocator.destroy_image(backing.image, backing.allocation);
    borrowed_view_backing_image = backing.image;
    borrowed_view_swapchain_creates = 0;
    borrowed_view_swapchain_destroys = 0;
    borrowed_view_acquires = 0;

    state.instance_dispatch.surface.get_surface_capabilities =
        &borrowed_view_surface_capabilities;
    state.instance_dispatch.surface.get_surface_formats =
        &borrowed_view_surface_formats;
    state.instance_dispatch.surface.get_surface_present_modes =
        &borrowed_view_present_modes;
    state.device_dispatch.presentation = {
        .create_swapchain     = &borrowed_view_create_swapchain,
        .destroy_swapchain    = &borrowed_view_destroy_swapchain,
        .get_swapchain_images = &borrowed_view_get_swapchain_images,
        .acquire_next_image   = &borrowed_view_acquire,
    };

    gpu::vk::SwapchainCell[1] swapchain_cells;
    state.presentation.swapchains = {
        .slots = swapchain_cells[..],
        .owner = gpu::resource_owner_tag(
            state.resource_owner,
            gpu::ResourceOwnerKind.SWAPCHAIN,
        ),
    };
    gpu::vk::SwapchainSlot slot = {
        .surface   = (vk::SurfaceKHR)1,
        .requested = {
            .width            = 4,
            .height           = 4,
            .preferred_format = gpu::Format.BGRA8_UNORM,
            .present_mode     = gpu::PresentMode.FIFO,
            .image_count      = 1,
        },
    };
    slot.acquire_semaphores = mem::new_array(
        vk::Semaphore,
        gpu::vk::SWAPCHAIN_ACQUIRE_SLOT_COUNT,
    );
    slot.acquire_retirements = mem::new_array(
        gpu::CompletionPoint,
        gpu::vk::SWAPCHAIN_ACQUIRE_SLOT_COUNT,
    );
    for (uint i = 0; i < gpu::vk::SWAPCHAIN_ACQUIRE_SLOT_COUNT; i++) {
        vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info();
        assert(vk::create_semaphore(
            device:        state.device,
            p_create_info: &semaphore_info,
            p_allocator:   null,
            p_semaphore:   &slot.acquire_semaphores[i],
        ) == vk::Result.SUCCESS, "the acquire semaphore fixture is valid");
    }
    gpu::vk::WsiDiagnosticContext context = {
        .state     = state,
        .operation = "borrowed_view_lifecycle_test",
    };
    gpu::vk::build_swapchain(
        state:   state,
        slot:    &slot,
        width:   4,
        height:  4,
        old:     (vk::SwapchainKHR)null,
        context: &context,
    )!!;
    gpu::SwapchainHandle swapchain =
        state.presentation.swapchains.alloc(slot)!!;
    gpu::vk::SwapchainSlot* live_slot =
        state.presentation.swapchains.get(swapchain)!!;

    gpu::AcquiredImage acquired = gpu::vk::vk_acquire_next_image(
        &device,
        swapchain,
    )!!;
    gpu::TextureHandle old_texture = acquired.texture;
    gpu::AttachmentViewHandle old_view = acquired.attachment_view;
    gpu::vk::TextureSlot* texture = state.textures.get(old_texture)!!;
    gpu::vk::AttachmentViewSlot* view =
        state.attachment_views.get(old_view)!!;
    assert(borrowed_view_swapchain_creates == 1
        && borrowed_view_acquires == 1
        && live_slot.image_count == 1,
        "production build and acquire publish one wrapped image");
    assert(texture.swapchain_owned && view.swapchain_owned
        && view.texture == old_texture
        && view.native == texture.default_view,
        "acquire returns the jointly owned texture and borrowed native view");
    assert(@catch(gpu::destroy_attachment_view(&device, old_view))
        == gpu::INVALID_RESOURCE_STATE,
        "the acquired borrowed view rejects caller destruction");

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    gpu::TextureBarrier to_color = {
        .texture = old_texture,
        .before  = acquired.prior_state,
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &to_color)!!;
    gpu::ColorTargetDesc[1] colors = {{ .view = old_view }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = 4,
        .height = 4,
    };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::TextureBarrier to_present = {
        .texture = old_texture,
        .before = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        .after = { .layout = gpu::TextureLayout.PRESENT },
    };
    gpu::cmd_texture_barrier(&commands, &to_present)!!;
    gpu::TextureBarrier invalid_present = to_present;
    invalid_present.after.stages = { .color_output };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &invalid_present))
        == gpu::INVALID_ARGUMENT,
        "public PRESENT state rejects nonempty stages");
    invalid_present.after = { .layout = gpu::TextureLayout.PRESENT };
    invalid_present.after.access = { .read };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &invalid_present))
        == gpu::INVALID_ARGUMENT,
        "public PRESENT state rejects nonempty access");
    assert(view.validation_references.load(AtomicOrdering.RELAXED) == 1,
        "recorded rendering retains the borrowed view");
    gpu::vk::commit_present_result(live_slot, &acquired, vk::Result.SUCCESS);

    assert(@catch(gpu::vk::vk_resize_swapchain(
        device: &device,
        handle: swapchain,
        width:  8,
        height: 8,
    )) == gpu::RESOURCE_IN_USE,
        "resize rejects a recorded borrowed-view reference");
    assert(@catch(gpu::vk::vk_destroy_swapchain(&device, swapchain))
        == gpu::RESOURCE_IN_USE,
        "destruction rejects a recorded borrowed-view reference");
    assert(view.validation_references.load(AtomicOrdering.RELAXED) == 1
        && state.textures.get(old_texture)!! == texture
        && state.attachment_views.get(old_view)!! == view,
        "failed release attempts preserve the command and borrowed handles");

    gpu::discard_commands(&commands)!!;
    assert(view.validation_references.load(AtomicOrdering.RELAXED) == 0,
        "discard releases the recorded borrowed-view reference");
    gpu::vk::vk_resize_swapchain(
        device: &device,
        handle: swapchain,
        width:  8,
        height: 8,
    )!!;
    assert(@catch(state.textures.get(old_texture)) == gpu::INVALID_HANDLE
        && @catch(state.attachment_views.get(old_view)) == gpu::INVALID_HANDLE,
        "successful resize invalidates both handles from the old acquisition");
    assert(borrowed_view_swapchain_creates == 2
        && borrowed_view_swapchain_destroys == 1,
        "resize rebuilds once and destroys the retired native swapchain once");

    gpu::vk::vk_destroy_swapchain(&device, swapchain)!!;
    assert(@catch(state.presentation.swapchains.get(swapchain))
        == gpu::INVALID_HANDLE,
        "destroy invalidates the rebuilt swapchain handle");
    assert(borrowed_view_swapchain_destroys == 2,
        "each fake native swapchain is destroyed exactly once");
}

fn void present_fences_retire_without_waiting() @test {
    gpu::vk::VkDeviceState state = { .device = (vk::Device)1 };
    gpu::vk::SwapchainSlot slot;
    slot.present_fences[0] = (vk::Fence)17;
    slot.present_fence_pending[0] = true;
    slot.image_count = 1;
    captured_fence_status_calls = 0;

    bool ready = gpu::vk::poll_present_fences_with(
        &state,
        &slot,
        &capture_fence_pending,
    )!!;
    assert(!ready && slot.present_fence_pending[0],
        "pending presentation stays retryable");
    assert(captured_fence_status_calls == 1,
        "release polls the presentation fence exactly once");

    ready = gpu::vk::poll_present_fences_with(
        &state,
        &slot,
        &capture_fence_ready,
    )!!;
    assert(ready && !slot.present_fence_pending[0],
        "signaled presentation fences retire without a wait");
}

fn void present_fence_reuse_is_nonblocking() @test {
    gpu::vk::VkDeviceState state = { .device = (vk::Device)1 };
    gpu::vk::SwapchainSlot slot;
    slot.image_count = 1;
    slot.present_fences[0] = (vk::Fence)23;
    slot.present_fence_pending[0] = true;
    captured_fence_status_calls = 0;
    captured_fence_reset_calls = 0;

    fault pending = @catch(gpu::vk::prepare_present_fence_with(
        state:       &state,
        slot:        &slot,
        image_index: 0,
        status_fn:   &capture_fence_pending,
        reset_fn:    &capture_fence_reset,
    ));
    assert(pending == gpu::WAIT_TIMEOUT,
        "present retries when its fence is still in flight");
    assert(captured_fence_reset_calls == 0,
        "an unsignaled fence is never reset");

    gpu::vk::prepare_present_fence_with(
        state:       &state,
        slot:        &slot,
        image_index: 0,
        status_fn:   &capture_fence_ready,
        reset_fn:    &capture_fence_reset,
    )!!;
    assert(!slot.present_fence_pending[0],
        "a signaled fence becomes available for the next present");
    assert(captured_fence_reset_calls == 1,
        "reuse resets the retired fence exactly once");
}

fn void swapchain_readiness_identity_round_trips() @test {
    gpu::Device device = { .index = 3, .generation = 11 };
    gpu::SwapchainHandle swapchain = {
        .owner      = 99,
        .index      = 7,
        .generation = 0x00FF_FFFE,
    };
    gpu::SwapchainReadiness readiness = gpu::vk::make_swapchain_readiness(
        device,
        swapchain,
        0xFFFF_FFFE,
    )!!;

    assert(readiness.device == device, "readiness keeps exact device identity");
    assert(gpu::vk::swapchain_readiness_index(readiness) == swapchain.index,
        "readiness keeps swapchain index");
    assert(gpu::vk::swapchain_readiness_generation(readiness) == swapchain.generation,
        "readiness keeps swapchain generation");
    assert(gpu::vk::swapchain_readiness_acquisition(readiness) == 0xFFFF_FFFE,
        "readiness keeps acquisition identity");
}

fn void swapchain_acquisition_sequence_never_wraps() @test {
    assert(gpu::vk::next_swapchain_acquisition(0)!! == 1,
        "first acquisition uses the first non-zero identity");
    fault exhausted = @catch(gpu::vk::next_swapchain_acquisition(uint::max));
    assert(exhausted == gpu::SLOT_TABLE_FULL, "acquisition identity faults before wrapping");
}


fn void swapchain_image_usage_mapping() @test {
    gpu::TextureUsage color_only = gpu::vk::swapchain_image_usage((uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT);
    assert(color_only.color_attach && !color_only.transfer_src, "ungranted transfer_src does not carry over");

    uint with_transfer_src = (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT | (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT;
    gpu::TextureUsage granted = gpu::vk::swapchain_image_usage(with_transfer_src);
    assert(granted.color_attach && granted.transfer_src, "granted transfer_src carries over");

    // Bits the swapchain never grants (e.g. STORAGE) must not leak into the
    // wrapped usage even if somehow present in the input mask.
    uint with_unrelated_bit = (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT | (uint)vk::ImageUsageFlagBits.STORAGE_BIT;
    gpu::TextureUsage narrow = gpu::vk::swapchain_image_usage(with_unrelated_bit);
    assert(narrow.color_attach && !narrow.transfer_src && !narrow.storage && !narrow.sampled,
        "mapping only ever sets color_attach and transfer_src");
}

fn void selected_present_mode_mapping() @test {
    assert(gpu::vk::present_mode_from_vk(vk::PresentModeKHR.FIFO_KHR) == gpu::PresentMode.FIFO,
        "selected FIFO maps to public FIFO");
    assert(gpu::vk::present_mode_from_vk(vk::PresentModeKHR.IMMEDIATE_KHR) == gpu::PresentMode.IMMEDIATE,
        "selected immediate maps to public immediate");
    assert(gpu::vk::present_mode_from_vk(vk::PresentModeKHR.MAILBOX_KHR) == gpu::PresentMode.MAILBOX,
        "selected mailbox maps to public mailbox");
}

fn void swapchain_info_snapshots_are_coherent() @test {
    vk::Extent2D first_extent = { .width = 640, .height = 480 };
    gpu::SwapchainInfo first = gpu::vk::make_swapchain_info(
        format:       gpu::Format.BGRA8_UNORM,
        extent:       first_extent,
        image_count:  3,
        present_mode: vk::PresentModeKHR.FIFO_KHR,
    );
    assert(first.format == gpu::Format.BGRA8_UNORM, "snapshot reports selected format");
    assert(first.width == 640 && first.height == 480, "snapshot reports clamped extent");
    assert(first.image_count == 3, "snapshot reports driver-returned image count");
    assert(first.present_mode == gpu::PresentMode.FIFO && !first.dormant,
        "active snapshot reports selected mode and active state");

    gpu::vk::SwapchainSlot slot = {
        .info               = first,
        .acquire_pending    = true,
        .readiness_consumed = true,
        .acquisition        = 9,
    };
    vk::Extent2D resized_extent = { .width = 1280, .height = 720 };
    gpu::SwapchainInfo resized = gpu::vk::make_swapchain_info(
        format:       gpu::Format.RGBA8_SRGB,
        extent:       resized_extent,
        image_count:  4,
        present_mode: vk::PresentModeKHR.MAILBOX_KHR,
    );
    gpu::vk::publish_active_swapchain_info(&slot, resized);
    assert(slot.info.format == gpu::Format.RGBA8_SRGB, "replacement snapshot has only the new format");
    assert(slot.info.width == 1280 && slot.info.height == 720 && slot.info.image_count == 4,
        "replacement snapshot has only the new extent and count");
    assert(slot.info.present_mode == gpu::PresentMode.MAILBOX && !slot.info.dormant,
        "replacement snapshot has only the new mode and state");
    assert(!slot.acquire_pending && !slot.readiness_consumed,
        "active publication resets the acquire lifecycle");
    assert(slot.acquisition == 9, "resize does not reuse acquisition identities");

    slot.acquire_pending = true;
    slot.readiness_consumed = true;
    gpu::vk::publish_dormant_swapchain_info(&slot);
    assert(slot.info.format == gpu::Format.UNDEFINED, "dormant publication clears the prior format");
    assert(slot.info.width == 0 && slot.info.height == 0 && slot.info.image_count == 0,
        "dormant publication clears the prior extent and image count");
    assert(slot.info.present_mode == gpu::PresentMode.FIFO && slot.info.dormant,
        "dormant publication uses the documented ignored mode sentinel");
    assert(!slot.acquire_pending && !slot.readiness_consumed,
        "dormant publication resets the acquire lifecycle");
    assert(slot.acquisition == 9, "dormancy does not reuse acquisition identities");
}

fn void acquired_image_reports_prior_state() @test {
    gpu::vk::TextureSlot texture;
    gpu::Device device = { .generation = 1 };
    gpu::SwapchainHandle swapchain = { .index = 1, .generation = 1 };
    gpu::SwapchainReadiness readiness = gpu::vk::make_swapchain_readiness(
        device,
        swapchain,
        1,
    )!!;
    gpu::TextureHandle handle = { .index = 2, .generation = 7 };
    gpu::AttachmentViewHandle attachment_view = {
        .index = 3, .generation = 11,
    };

    gpu::AcquiredImage first = gpu::vk::make_acquired_image(
        texture:         &texture,
        handle:          handle,
        attachment_view: attachment_view,
        readiness:       readiness,
        index:           1,
        suboptimal:      false,
    );
    assert(first.prior_state.layout == gpu::TextureLayout.UNDEFINED
        && (uint)first.prior_state.stages == 0
        && (uint)first.prior_state.access == 0,
        "newly wrapped image reports the exact undefined prior state");
    assert(first.texture == handle && first.attachment_view == attachment_view
        && first.index == 1 && !first.suboptimal,
        "acquired-image construction preserves existing fields");
    assert(first.readiness == readiness, "acquired image carries one-shot readiness");

    gpu::vk::mark_texture_presented(&texture);
    gpu::AcquiredImage presented = gpu::vk::make_acquired_image(
        texture:         &texture,
        handle:          handle,
        attachment_view: attachment_view,
        readiness:       readiness,
        index:           1,
        suboptimal:      true,
    );
    assert(presented.prior_state.layout == gpu::TextureLayout.PRESENT
        && (uint)presented.prior_state.stages == 0
        && (uint)presented.prior_state.access == 0,
        "presented image reports the exact presentation state");
    assert(presented.suboptimal, "acquired-image construction preserves suboptimal state");
}

fn void swapchain_info_query_rejects_invalid_handle() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_swapchain_info_invalid_handle_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault query_fault = @catch(gpu::get_swapchain_info(&device, bogus));
    assert(query_fault == gpu::INVALID_HANDLE, "query on a bogus swapchain faults INVALID_HANDLE");
}
