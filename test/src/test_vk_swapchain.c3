module gpu_test;

import gpu;
import gpu::vk;
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
