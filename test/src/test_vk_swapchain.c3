module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

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

fn void acquire_result_mapping() @test {
    gpu::vk::map_acquire_result(vk::Result.SUCCESS)!!;
    gpu::vk::map_acquire_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault timeout = @catch(gpu::vk::map_acquire_result(vk::Result.TIMEOUT));
    fault not_ready = @catch(gpu::vk::map_acquire_result(vk::Result.NOT_READY));
    fault out_of_date = @catch(gpu::vk::map_acquire_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::vk::map_acquire_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::vk::map_acquire_result(vk::Result.ERROR_DEVICE_LOST));

    assert(timeout == gpu::WAIT_TIMEOUT, "acquire timeout is retryable");
    assert(not_ready == gpu::WAIT_TIMEOUT, "acquire not-ready is retryable");
    assert(out_of_date == gpu::SWAPCHAIN_OUT_OF_DATE, "out-of-date requires swapchain resize");
    assert(surface_lost == gpu::SURFACE_LOST, "surface loss remains distinct");
    assert(device_lost == gpu::DEVICE_LOST, "unrelated results delegate to the common mapper");
}

fn void present_result_mapping() @test {
    gpu::vk::map_present_result(vk::Result.SUCCESS)!!;
    gpu::vk::map_present_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault out_of_date = @catch(gpu::vk::map_present_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::vk::map_present_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::vk::map_present_result(vk::Result.ERROR_DEVICE_LOST));

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

fn void swapchain_requires_presentation() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_swapchain_gate_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::SurfaceDesc surface = { .platform = gpu::PlatformKind.WAYLAND };
    gpu::SwapchainDesc sc_desc = { .width = 640, .height = 480 };
    fault gate_fault = @catch(gpu::create_swapchain(&device, &surface, &sc_desc));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "swapchain without enable_presentation faults UNSUPPORTED_FEATURE");
}

// Acquire-submit-present coupling lifecycle. One state matrix per
// guard, covering every (pending, consumed[, kind]) combination the guard
// distinguishes; the not-acquired/not-consumed rows also cover the states
// build_swapchain's dormant path and vk_present's clear leave the slot in.

fn void swapchain_submit_guard_lifecycle() @test {
    // Never acquired: nothing to couple to.
    fault not_acquired = @catch(gpu::vk::swapchain_submit_guard(false, false, gpu::QueueKind.GRAPHICS));
    assert(not_acquired == gpu::INVALID_RESOURCE_STATE, "submit with no pending acquire faults INVALID_RESOURCE_STATE");

    // Acquired, not yet consumed, GRAPHICS: the one legal coupled submit.
    gpu::vk::swapchain_submit_guard(true, false, gpu::QueueKind.GRAPHICS)!!;

    // Already consumed by a prior coupled submit: the double-submit case
    // this guard exists to reject before the queue is touched.
    fault double_submit = @catch(gpu::vk::swapchain_submit_guard(true, true, gpu::QueueKind.GRAPHICS));
    assert(double_submit == gpu::INVALID_RESOURCE_STATE, "second coupled submit faults INVALID_RESOURCE_STATE");

    // consumed=true with pending=false cannot arise from the real lifecycle
    // (present clears both together) but the guard must still reject it.
    fault stale_consumed = @catch(gpu::vk::swapchain_submit_guard(false, true, gpu::QueueKind.GRAPHICS));
    assert(stale_consumed == gpu::INVALID_RESOURCE_STATE, "consumed without pending faults INVALID_RESOURCE_STATE");

    // Non-GRAPHICS coupling: rejected regardless of an otherwise-legal state.
    fault compute_coupled = @catch(gpu::vk::swapchain_submit_guard(true, false, gpu::QueueKind.COMPUTE));
    assert(compute_coupled == gpu::INVALID_ARGUMENT, "COMPUTE-coupled submit faults INVALID_ARGUMENT");

    fault transfer_coupled = @catch(gpu::vk::swapchain_submit_guard(true, false, gpu::QueueKind.TRANSFER));
    assert(transfer_coupled == gpu::INVALID_ARGUMENT, "TRANSFER-coupled submit faults INVALID_ARGUMENT");
}

fn void swapchain_present_guard_lifecycle() @test {
    // Never acquired.
    fault not_acquired = @catch(gpu::vk::swapchain_present_guard(false, false));
    assert(not_acquired == gpu::INVALID_RESOURCE_STATE, "present with no pending acquire faults INVALID_RESOURCE_STATE");

    // Acquired but never submitted: the unsignaled-semaphore hang this guard
    // replaces with a deterministic fault.
    fault not_consumed = @catch(gpu::vk::swapchain_present_guard(true, false));
    assert(not_consumed == gpu::INVALID_RESOURCE_STATE, "present without a consumed submit faults INVALID_RESOURCE_STATE");

    // Acquired and consumed by a coupled submit: the one legal present.
    gpu::vk::swapchain_present_guard(true, true)!!;

    // consumed=true with pending=false: same unreachable-but-must-reject case
    // as the submit guard.
    fault stale_consumed = @catch(gpu::vk::swapchain_present_guard(false, true));
    assert(stale_consumed == gpu::INVALID_RESOURCE_STATE, "consumed without pending faults INVALID_RESOURCE_STATE");
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
