module gpu_test;

import gpu::compat;
import gpu::compat::vk @public;
import vk;

// Pure swapchain selection logic over synthetic capability lists, plus the
// headless presentation gate. Windowed behavior is covered by the
// hello_triangle_sdl smoke run.

fn void surface_format_selection() @test {
    vk::SurfaceFormatKHR[2] bgra_only = {
        { .format = vk::Format.B8G8R8A8_UNORM, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
        { .format = vk::Format.B8G8R8A8_SRGB, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
    };

    gpu::compat::Format preferred = gpu::compat::vk::select_surface_format(bgra_only[..], gpu::compat::Format.BGRA8_SRGB, srgb: true)!!;
    assert(preferred == gpu::compat::Format.BGRA8_SRGB, "preferred format wins when available");

    gpu::compat::Format fallback = gpu::compat::vk::select_surface_format(bgra_only[..], gpu::compat::Format.RGBA8_UNORM, srgb: false)!!;
    assert(fallback == gpu::compat::Format.BGRA8_UNORM, "unavailable preferred falls back to BGRA unorm");

    gpu::compat::Format srgb_pick = gpu::compat::vk::select_surface_format(bgra_only[..], gpu::compat::Format.UNDEFINED, srgb: true)!!;
    assert(srgb_pick == gpu::compat::Format.BGRA8_SRGB, "srgb flag selects the srgb flavor");

    vk::SurfaceFormatKHR[1] alien = {
        { .format = vk::Format.R16G16B16A16_SFLOAT, .color_space = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR },
    };
    fault none = @catch(gpu::compat::vk::select_surface_format(alien[..], gpu::compat::Format.UNDEFINED, srgb: false));
    assert(none == gpu::compat::UNSUPPORTED_FEATURE, "no swapchain-usable format faults UNSUPPORTED_FEATURE");
}

fn void present_mode_selection() @test {
    vk::PresentModeKHR[2] fifo_mailbox = { vk::PresentModeKHR.FIFO_KHR, vk::PresentModeKHR.MAILBOX_KHR };
    assert(gpu::compat::vk::select_present_mode(fifo_mailbox[..], gpu::compat::PresentMode.MAILBOX) == vk::PresentModeKHR.MAILBOX_KHR,
        "supported mailbox is honored");

    vk::PresentModeKHR[1] fifo_only = { vk::PresentModeKHR.FIFO_KHR };
    assert(gpu::compat::vk::select_present_mode(fifo_only[..], gpu::compat::PresentMode.MAILBOX) == vk::PresentModeKHR.FIFO_KHR,
        "unsupported mailbox falls back to FIFO");
    assert(gpu::compat::vk::select_present_mode(fifo_only[..], gpu::compat::PresentMode.IMMEDIATE) == vk::PresentModeKHR.FIFO_KHR,
        "unsupported immediate falls back to FIFO");
    assert(gpu::compat::vk::select_present_mode(fifo_mailbox[..], gpu::compat::PresentMode.FIFO) == vk::PresentModeKHR.FIFO_KHR,
        "FIFO requested is FIFO");
}

fn void common_surface_lost_result_mapping() @test {
    fault surface_lost = @catch(gpu::compat::vk::check(vk::Result.ERROR_SURFACE_LOST_KHR));
    assert(surface_lost == gpu::compat::SURFACE_LOST, "common mapper preserves surface loss");
}

fn void acquire_result_mapping() @test {
    gpu::compat::vk::check_acquire_result(vk::Result.SUCCESS)!!;
    gpu::compat::vk::check_acquire_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault timeout = @catch(gpu::compat::vk::check_acquire_result(vk::Result.TIMEOUT));
    fault not_ready = @catch(gpu::compat::vk::check_acquire_result(vk::Result.NOT_READY));
    fault out_of_date = @catch(gpu::compat::vk::check_acquire_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::compat::vk::check_acquire_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::compat::vk::check_acquire_result(vk::Result.ERROR_DEVICE_LOST));

    assert(timeout == gpu::compat::WAIT_TIMEOUT, "acquire timeout is retryable");
    assert(not_ready == gpu::compat::WAIT_TIMEOUT, "acquire not-ready is retryable");
    assert(out_of_date == gpu::compat::SWAPCHAIN_OUT_OF_DATE, "out-of-date requires swapchain resize");
    assert(surface_lost == gpu::compat::SURFACE_LOST, "surface loss remains distinct");
    assert(device_lost == gpu::compat::DEVICE_LOST, "unrelated results delegate to the common mapper");
}

fn void present_result_mapping() @test {
    gpu::compat::vk::check_present_result(vk::Result.SUCCESS)!!;
    gpu::compat::vk::check_present_result(vk::Result.SUBOPTIMAL_KHR)!!;

    fault out_of_date = @catch(gpu::compat::vk::check_present_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault surface_lost = @catch(gpu::compat::vk::check_present_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault device_lost = @catch(gpu::compat::vk::check_present_result(vk::Result.ERROR_DEVICE_LOST));

    assert(out_of_date == gpu::compat::SWAPCHAIN_OUT_OF_DATE, "out-of-date requires swapchain resize");
    assert(surface_lost == gpu::compat::SURFACE_LOST, "surface loss requires surface recreation");
    assert(device_lost == gpu::compat::DEVICE_LOST, "unrelated results delegate to the common mapper");
}

fn void image_count_clamp() @test {
    assert(gpu::compat::vk::clamp_image_count(0, 2, 8) == 3, "zero requests min + 1");
    assert(gpu::compat::vk::clamp_image_count(1, 2, 8) == 2, "below min clamps up");
    assert(gpu::compat::vk::clamp_image_count(16, 2, 8) == 8, "above max clamps down");
    assert(gpu::compat::vk::clamp_image_count(16, 2, 0) == 16, "max 0 means unbounded");
}

fn void extent_clamp_rules() @test {
    vk::SurfaceCapabilitiesKHR fixed = {
        .current_extent   = { .width = 640, .height = 480 },
        .min_image_extent = { .width = 1, .height = 1 },
        .max_image_extent = { .width = 4096, .height = 4096 },
    };
    vk::Extent2D from_fixed = gpu::compat::vk::clamp_extent(&fixed, 1024, 768);
    assert(from_fixed.width == 640 && from_fixed.height == 480, "fixed currentExtent wins over the request");

    vk::SurfaceCapabilitiesKHR app_decides = {
        .current_extent   = { .width = uint::max, .height = uint::max },
        .min_image_extent = { .width = 64, .height = 64 },
        .max_image_extent = { .width = 2048, .height = 2048 },
    };
    vk::Extent2D clamped = gpu::compat::vk::clamp_extent(&app_decides, 4096, 32);
    assert(clamped.width == 2048 && clamped.height == 64, "request clamps to min/max when the app decides");
}

fn void swapchain_requires_presentation() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_swapchain_gate_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SurfaceDesc surface = { .platform = gpu::compat::PlatformKind.WAYLAND };
    gpu::compat::SwapchainDesc sc_desc = { .width = 640, .height = 480 };
    fault gate_fault = @catch(gpu::compat::create_swapchain(&device, &surface, &sc_desc));
    assert(gate_fault == gpu::compat::UNSUPPORTED_FEATURE, "swapchain without enable_presentation faults UNSUPPORTED_FEATURE");
}

// Acquire-submit-present coupling lifecycle. One state matrix per
// guard, covering every (pending, consumed[, kind]) combination the guard
// distinguishes; the not-acquired/not-consumed rows also cover the states
// build_swapchain's dormant path and vk_present's clear leave the slot in.

fn void swapchain_submit_guard_lifecycle() @test {
    // Never acquired: nothing to couple to.
    fault not_acquired = @catch(gpu::compat::vk::swapchain_submit_guard(false, false, gpu::compat::QueueKind.GRAPHICS));
    assert(not_acquired == gpu::compat::INVALID_RESOURCE_STATE, "submit with no pending acquire faults INVALID_RESOURCE_STATE");

    // Acquired, not yet consumed, GRAPHICS: the one legal coupled submit.
    gpu::compat::vk::swapchain_submit_guard(true, false, gpu::compat::QueueKind.GRAPHICS)!!;

    // Already consumed by a prior coupled submit: the double-submit case
    // this guard exists to reject before the queue is touched.
    fault double_submit = @catch(gpu::compat::vk::swapchain_submit_guard(true, true, gpu::compat::QueueKind.GRAPHICS));
    assert(double_submit == gpu::compat::INVALID_RESOURCE_STATE, "second coupled submit faults INVALID_RESOURCE_STATE");

    // consumed=true with pending=false cannot arise from the real lifecycle
    // (present clears both together) but the guard must still reject it.
    fault stale_consumed = @catch(gpu::compat::vk::swapchain_submit_guard(false, true, gpu::compat::QueueKind.GRAPHICS));
    assert(stale_consumed == gpu::compat::INVALID_RESOURCE_STATE, "consumed without pending faults INVALID_RESOURCE_STATE");

    // Non-GRAPHICS coupling: rejected regardless of an otherwise-legal state.
    fault compute_coupled = @catch(gpu::compat::vk::swapchain_submit_guard(true, false, gpu::compat::QueueKind.COMPUTE));
    assert(compute_coupled == gpu::compat::INVALID_ARGUMENT, "COMPUTE-coupled submit faults INVALID_ARGUMENT");

    fault transfer_coupled = @catch(gpu::compat::vk::swapchain_submit_guard(true, false, gpu::compat::QueueKind.TRANSFER));
    assert(transfer_coupled == gpu::compat::INVALID_ARGUMENT, "TRANSFER-coupled submit faults INVALID_ARGUMENT");
}

fn void swapchain_present_guard_lifecycle() @test {
    // Never acquired.
    fault not_acquired = @catch(gpu::compat::vk::swapchain_present_guard(false, false));
    assert(not_acquired == gpu::compat::INVALID_RESOURCE_STATE, "present with no pending acquire faults INVALID_RESOURCE_STATE");

    // Acquired but never submitted: the unsignaled-semaphore hang this guard
    // replaces with a deterministic fault.
    fault not_consumed = @catch(gpu::compat::vk::swapchain_present_guard(true, false));
    assert(not_consumed == gpu::compat::INVALID_RESOURCE_STATE, "present without a consumed submit faults INVALID_RESOURCE_STATE");

    // Acquired and consumed by a coupled submit: the one legal present.
    gpu::compat::vk::swapchain_present_guard(true, true)!!;

    // consumed=true with pending=false: same unreachable-but-must-reject case
    // as the submit guard.
    fault stale_consumed = @catch(gpu::compat::vk::swapchain_present_guard(false, true));
    assert(stale_consumed == gpu::compat::INVALID_RESOURCE_STATE, "consumed without pending faults INVALID_RESOURCE_STATE");
}

fn void swapchain_image_usage_mapping() @test {
    gpu::compat::TextureUsage color_only = gpu::compat::vk::swapchain_image_usage((uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT);
    assert(color_only.color_attach && !color_only.transfer_src, "ungranted transfer_src does not carry over");

    uint with_transfer_src = (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT | (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT;
    gpu::compat::TextureUsage granted = gpu::compat::vk::swapchain_image_usage(with_transfer_src);
    assert(granted.color_attach && granted.transfer_src, "granted transfer_src carries over");

    // Bits the swapchain never grants (e.g. STORAGE) must not leak into the
    // wrapped usage even if somehow present in the input mask.
    uint with_unrelated_bit = (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT | (uint)vk::ImageUsageFlagBits.STORAGE_BIT;
    gpu::compat::TextureUsage narrow = gpu::compat::vk::swapchain_image_usage(with_unrelated_bit);
    assert(narrow.color_attach && !narrow.transfer_src && !narrow.storage && !narrow.sampled,
        "mapping only ever sets color_attach and transfer_src");
}

fn void selected_present_mode_mapping() @test {
    assert(gpu::compat::vk::present_mode_from_vk(vk::PresentModeKHR.FIFO_KHR) == gpu::compat::PresentMode.FIFO,
        "selected FIFO maps to public FIFO");
    assert(gpu::compat::vk::present_mode_from_vk(vk::PresentModeKHR.IMMEDIATE_KHR) == gpu::compat::PresentMode.IMMEDIATE,
        "selected immediate maps to public immediate");
    assert(gpu::compat::vk::present_mode_from_vk(vk::PresentModeKHR.MAILBOX_KHR) == gpu::compat::PresentMode.MAILBOX,
        "selected mailbox maps to public mailbox");
}

fn void swapchain_info_snapshots_are_coherent() @test {
    vk::Extent2D first_extent = { .width = 640, .height = 480 };
    gpu::compat::SwapchainInfo first = gpu::compat::vk::make_swapchain_info(
        format:       gpu::compat::Format.BGRA8_UNORM,
        extent:       first_extent,
        image_count:  3,
        present_mode: vk::PresentModeKHR.FIFO_KHR,
    );
    assert(first.format == gpu::compat::Format.BGRA8_UNORM, "snapshot reports selected format");
    assert(first.width == 640 && first.height == 480, "snapshot reports clamped extent");
    assert(first.image_count == 3, "snapshot reports driver-returned image count");
    assert(first.present_mode == gpu::compat::PresentMode.FIFO && !first.dormant,
        "active snapshot reports selected mode and active state");

    gpu::compat::vk::SwapchainSlot slot = {
        .info             = first,
        .acquire_pending  = true,
        .acquire_consumed = true,
    };
    vk::Extent2D resized_extent = { .width = 1280, .height = 720 };
    gpu::compat::SwapchainInfo resized = gpu::compat::vk::make_swapchain_info(
        format:       gpu::compat::Format.RGBA8_SRGB,
        extent:       resized_extent,
        image_count:  4,
        present_mode: vk::PresentModeKHR.MAILBOX_KHR,
    );
    gpu::compat::vk::publish_active_swapchain_info(&slot, resized);
    assert(slot.info.format == gpu::compat::Format.RGBA8_SRGB, "replacement snapshot has only the new format");
    assert(slot.info.width == 1280 && slot.info.height == 720 && slot.info.image_count == 4,
        "replacement snapshot has only the new extent and count");
    assert(slot.info.present_mode == gpu::compat::PresentMode.MAILBOX && !slot.info.dormant,
        "replacement snapshot has only the new mode and state");
    assert(!slot.acquire_pending && !slot.acquire_consumed,
        "active publication resets the acquire lifecycle");

    slot.acquire_pending = true;
    slot.acquire_consumed = true;
    gpu::compat::vk::publish_dormant_swapchain_info(&slot);
    assert(slot.info.format == gpu::compat::Format.UNDEFINED, "dormant publication clears the prior format");
    assert(slot.info.width == 0 && slot.info.height == 0 && slot.info.image_count == 0,
        "dormant publication clears the prior extent and image count");
    assert(slot.info.present_mode == gpu::compat::PresentMode.FIFO && slot.info.dormant,
        "dormant publication uses the documented ignored mode sentinel");
    assert(!slot.acquire_pending && !slot.acquire_consumed,
        "dormant publication resets the acquire lifecycle");
}

fn void acquired_image_reports_committed_prior_layout() @test {
    gpu::compat::vk::TextureSlot texture;
    gpu::compat::TextureHandle handle = { .index = 2, .generation = 7 };

    gpu::compat::AcquiredImage first = gpu::compat::vk::make_acquired_image(
        texture:    &texture,
        handle:     handle,
        index:      1,
        suboptimal: false,
    );
    assert(first.prior_layout == gpu::compat::TextureLayout.UNDEFINED,
        "newly wrapped image reports undefined prior layout");
    assert((ulong)first.texture == (ulong)handle && first.index == 1 && !first.suboptimal,
        "acquired-image construction preserves existing fields");

    gpu::compat::vk::set_texture_layout(&texture, gpu::compat::TextureLayout.PRESENT);
    gpu::compat::AcquiredImage presented = gpu::compat::vk::make_acquired_image(
        texture:    &texture,
        handle:     handle,
        index:      1,
        suboptimal: true,
    );
    assert(presented.prior_layout == gpu::compat::TextureLayout.PRESENT,
        "presented image reports committed present prior layout");
    assert(presented.suboptimal, "acquired-image construction preserves suboptimal state");
}

fn void swapchain_info_query_rejects_invalid_handle() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_swapchain_info_invalid_handle_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault query_fault = @catch(gpu::compat::get_swapchain_info(&device, bogus));
    assert(query_fault == gpu::compat::INVALID_HANDLE, "query on a bogus swapchain faults INVALID_HANDLE");
}
