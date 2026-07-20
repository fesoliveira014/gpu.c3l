module gpu_test;

import gpu::vk @public;
import vk;

struct DispatchLookupCounts {
    uint instance_debug;
    uint instance_surface;
    uint instance_win32_surface;
    uint instance_wayland_surface;
    uint instance_xlib_surface;
    uint device_debug;
    uint device_presentation;
    uint descriptor_buffer;
    uint unexpected;
}

fn bool DispatchLookupCounts.is_empty(self) {
    return self.instance_debug == 0
        && self.instance_surface == 0
        && self.instance_win32_surface == 0
        && self.instance_wayland_surface == 0
        && self.instance_xlib_surface == 0
        && self.device_debug == 0
        && self.device_presentation == 0
        && self.descriptor_buffer == 0
        && self.unexpected == 0;
}

fn void fake_void_proc() {}

fn bool supports_no_instance_extensions(ZString name) => false;

fn bool supports_all_instance_extensions(ZString name) => true;

fn bool supports_win32_instance_extensions(ZString name) {
    return name.str_view() == gpu::vk::SURFACE_EXTENSION.str_view()
        || name.str_view() == gpu::vk::WIN32_SURFACE_EXTENSION.str_view();
}

fn bool supports_surface_maintenance_without_capabilities2(ZString name) {
    return name.str_view() == gpu::vk::SURFACE_EXTENSION.str_view()
        || name.str_view() == gpu::vk::SURFACE_MAINTENANCE_EXTENSION.str_view();
}

fn void instance_capabilities_require_request_and_support() @test {
    gpu::vk::VkInstanceDesc desc = {
        .enable_debug_utils        = true,
        .enable_available_surfaces = true,
    };
    gpu::vk::VkInstanceCapabilities capabilities =
        gpu::vk::resolve_instance_capabilities(
            &desc,
            &supports_no_instance_extensions,
        );
    assert(!capabilities.debug_utils, "unsupported debug utils remain disabled");
    assert(!capabilities.surface, "unsupported base surface support remains disabled");
    assert(!capabilities.surface_maintenance,
        "unsupported surface maintenance remains disabled");
    assert(!capabilities.win32_surface, "platform surfaces require the base surface extension");
    assert(!capabilities.wayland_surface, "platform surfaces require the base surface extension");
    assert(!capabilities.xlib_surface, "platform surfaces require the base surface extension");

    capabilities = gpu::vk::resolve_instance_capabilities(
        &desc,
        &supports_all_instance_extensions,
    );
    assert(capabilities.debug_utils, "requested supported debug utils are enabled");
    assert(capabilities.surface, "requested supported surface discovery is enabled");
    assert(capabilities.surface_maintenance,
        "requested supported surface maintenance is enabled");
    assert(capabilities.win32_surface, "available Win32 surface support is enabled");
    assert(capabilities.wayland_surface, "available Wayland surface support is enabled");
    assert(capabilities.xlib_surface, "available Xlib surface support is enabled");

    capabilities = gpu::vk::resolve_instance_capabilities(
        &desc,
        &supports_win32_instance_extensions,
    );
    assert(capabilities.surface, "supported base surface discovery is enabled");
    assert(!capabilities.surface_maintenance,
        "missing optional surface maintenance stays disabled");
    assert(capabilities.win32_surface, "supported Win32 surface support is enabled");
    assert(!capabilities.wayland_surface, "unsupported Wayland surface support remains disabled");
    assert(!capabilities.xlib_surface, "unsupported Xlib surface support remains disabled");

    capabilities = gpu::vk::resolve_instance_capabilities(
        &desc,
        &supports_surface_maintenance_without_capabilities2,
    );
    assert(!capabilities.surface_maintenance,
        "surface maintenance requires get-surface-capabilities2");

    desc.enable_available_surfaces = false;
    capabilities = gpu::vk::resolve_instance_capabilities(
        &desc,
        &supports_all_instance_extensions,
    );
    assert(!capabilities.surface, "unrequested surface discovery remains disabled");
    assert(!capabilities.surface_maintenance,
        "unrequested surface maintenance remains disabled");
    assert(!capabilities.win32_surface, "unrequested Win32 surface support remains disabled");
    assert(!capabilities.wayland_surface, "unrequested Wayland surface support remains disabled");
    assert(!capabilities.xlib_surface, "unrequested Xlib surface support remains disabled");
}

fn void debug_messenger_requires_resolved_capability() @test {
    gpu::vk::VkInstanceCapabilities capabilities;

    assert(
        !gpu::vk::should_create_debug_messenger(capabilities, true),
        "unsupported debug utils skip messenger creation"
    );
    assert(
        gpu::vk::should_report_debug_messenger_unavailable(capabilities, true),
        "requested unavailable debug utils require a diagnostic"
    );
    assert(
        !gpu::vk::should_report_debug_messenger_unavailable(capabilities, false),
        "unrequested debug utils remain silent"
    );

    capabilities.debug_utils = true;
    assert(
        gpu::vk::should_create_debug_messenger(capabilities, true),
        "enabled debug utils permit requested messenger creation"
    );
    assert(
        !gpu::vk::should_create_debug_messenger(capabilities, false),
        "unrequested messenger creation remains disabled"
    );
    assert(
        !gpu::vk::should_report_debug_messenger_unavailable(capabilities, true),
        "available debug utils do not report unavailability"
    );
}

fn vk::PFN_vkVoidFunction recording_instance_proc_loader(
    vk::Instance instance,
    char* name,
) {
    DispatchLookupCounts* counts = (DispatchLookupCounts*)instance;
    String proc_name = ((ZString)name).str_view();
    if (proc_name == "vkCreateDebugUtilsMessengerEXT"
        || proc_name == "vkDestroyDebugUtilsMessengerEXT") {
        counts.instance_debug++;
    } else if (proc_name == "vkCreateWin32SurfaceKHR") {
        counts.instance_win32_surface++;
    } else if (proc_name == "vkCreateWaylandSurfaceKHR") {
        counts.instance_wayland_surface++;
    } else if (proc_name == "vkCreateXlibSurfaceKHR") {
        counts.instance_xlib_surface++;
    } else if (proc_name == "vkDestroySurfaceKHR"
        || proc_name == "vkGetPhysicalDeviceSurfaceSupportKHR"
        || proc_name == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        || proc_name == "vkGetPhysicalDeviceSurfaceFormatsKHR"
        || proc_name == "vkGetPhysicalDeviceSurfacePresentModesKHR") {
        counts.instance_surface++;
    } else {
        counts.unexpected++;
    }
    return (vk::PFN_vkVoidFunction)&fake_void_proc;
}

fn vk::PFN_vkVoidFunction recording_device_proc_loader(
    vk::Device device,
    char* name,
) {
    DispatchLookupCounts* counts = (DispatchLookupCounts*)device;
    String proc_name = ((ZString)name).str_view();
    if (proc_name == "vkSetDebugUtilsObjectNameEXT"
        || proc_name == "vkCmdBeginDebugUtilsLabelEXT"
        || proc_name == "vkCmdEndDebugUtilsLabelEXT") {
        counts.device_debug++;
    } else if (proc_name == "vkCreateSwapchainKHR"
        || proc_name == "vkDestroySwapchainKHR"
        || proc_name == "vkGetSwapchainImagesKHR"
        || proc_name == "vkAcquireNextImageKHR"
        || proc_name == "vkQueuePresentKHR") {
        counts.device_presentation++;
    } else if (proc_name == "vkGetDescriptorSetLayoutSizeEXT"
        || proc_name == "vkGetDescriptorSetLayoutBindingOffsetEXT"
        || proc_name == "vkGetDescriptorEXT"
        || proc_name == "vkCmdBindDescriptorBuffersEXT"
        || proc_name == "vkCmdSetDescriptorBufferOffsetsEXT") {
        counts.descriptor_buffer++;
    } else {
        counts.unexpected++;
    }
    return (vk::PFN_vkVoidFunction)&fake_void_proc;
}

fn void instance_dispatch_loads_only_selected_groups() @test {
    gpu::vk::VkInstanceDispatchPlan plan;
    DispatchLookupCounts counts;

    counts = {};
    (void)gpu::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.is_empty(), "disabled instance groups perform no lookups");

    counts = {};
    plan.debug = true;
    (void)gpu::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.instance_debug == 2, "debug selection loads its complete pair");
    assert(counts.instance_surface == 0, "debug selection does not load surface procedures");
    assert(counts.unexpected == 0, "debug selection loads no unrelated procedures");

    counts = {};
    plan = { .surface = true };
    (void)gpu::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.instance_debug == 0, "surface selection does not load debug procedures");
    assert(counts.instance_surface == 5, "surface selection loads the common surface group");
    assert(counts.instance_win32_surface == 0, "surface selection does not load Win32 construction");
    assert(counts.instance_wayland_surface == 0, "surface selection does not load Wayland construction");
    assert(counts.instance_xlib_surface == 0, "surface selection does not load Xlib construction");
    assert(counts.unexpected == 0, "surface selection loads no unrelated procedures");

    counts = {};
    plan = {
        .surface         = true,
        .win32_surface   = true,
        .wayland_surface = true,
        .xlib_surface    = true,
    };
    (void)gpu::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.instance_surface == 5, "platform selection preserves common surface procedures");
    assert(counts.instance_win32_surface == 1, "Win32 selection loads its constructor");
    assert(counts.instance_wayland_surface == 1, "Wayland selection loads its constructor");
    assert(counts.instance_xlib_surface == 1, "Xlib selection loads its constructor");
    assert(counts.unexpected == 0, "platform selection loads no unrelated procedures");
}

fn void device_dispatch_loads_only_selected_groups() @test {
    gpu::vk::VkDeviceDispatchPlan plan;
    DispatchLookupCounts counts;

    counts = {};
    (void)gpu::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.is_empty(), "disabled device groups perform no lookups");

    counts = {};
    plan.debug = true;
    (void)gpu::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 3, "debug selection loads only device diagnostics");
    assert(counts.device_presentation == 0, "debug selection does not load presentation procedures");
    assert(counts.descriptor_buffer == 0, "debug selection does not load descriptor procedures");
    assert(counts.unexpected == 0, "debug selection loads no unrelated procedures");

    counts = {};
    plan = { .presentation = true };
    (void)gpu::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 0, "presentation selection does not load debug procedures");
    assert(counts.device_presentation == 5, "presentation selection loads only device WSI");
    assert(counts.descriptor_buffer == 0, "presentation selection does not load descriptor procedures");
    assert(counts.unexpected == 0, "presentation selection loads no unrelated procedures");

    counts = {};
    plan = { .descriptor_buffer = true };
    (void)gpu::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 0, "descriptor selection does not load debug procedures");
    assert(counts.device_presentation == 0, "descriptor selection does not load presentation procedures");
    assert(counts.descriptor_buffer == 5, "descriptor selection loads only descriptor-buffer procedures");
    assert(counts.unexpected == 0, "descriptor selection loads no unrelated procedures");
}

fn void fake_get_descriptor_a(
    vk::Device device,
    vk::DescriptorGetInfoEXT* info,
    sz data_size,
    void* descriptor,
) {}

fn void fake_get_descriptor_b(
    vk::Device device,
    vk::DescriptorGetInfoEXT* info,
    sz data_size,
    void* descriptor,
) {}

fn vk::PFN_vkVoidFunction fake_device_proc_loader(
    vk::Device device,
    char* name,
) {
    if (((ZString)name).str_view() != "vkGetDescriptorEXT") return null;
    return (vk::PFN_vkVoidFunction)(device == (vk::Device)(uptr)1
        ? &fake_get_descriptor_a
        : &fake_get_descriptor_b);
}

fn void device_dispatch_loads_per_device() @test {
    gpu::vk::VkDeviceDispatchPlan plan = { .descriptor_buffer = true };
    gpu::vk::VkDeviceDispatch first = gpu::vk::load_vk_device_dispatch(
        (vk::Device)(uptr)1,
        &fake_device_proc_loader,
        plan,
    );
    gpu::vk::VkDeviceDispatch second = gpu::vk::load_vk_device_dispatch(
        (vk::Device)(uptr)2,
        &fake_device_proc_loader,
        plan,
    );

    assert(first.descriptor_buffer.get_descriptor == &fake_get_descriptor_a, "first device keeps its loaded entry point");
    assert(second.descriptor_buffer.get_descriptor == &fake_get_descriptor_b, "second device keeps its loaded entry point");
    assert(first.descriptor_buffer.get_descriptor != second.descriptor_buffer.get_descriptor, "device dispatch values do not alias mutable state");
}

fn vk::Result fake_create_debug_messenger(
    vk::Instance instance,
    vk::DebugUtilsMessengerCreateInfoEXT* info,
    vk::AllocationCallbacks* allocator,
    vk::DebugUtilsMessengerEXT* messenger,
) {
    return vk::Result.SUCCESS;
}

fn void fake_destroy_debug_messenger(
    vk::Instance instance,
    vk::DebugUtilsMessengerEXT messenger,
    vk::AllocationCallbacks* allocator,
) {}

fn void fake_begin_label(
    vk::CommandBuffer command_buffer,
    vk::DebugUtilsLabelEXT* info,
) {}

fn void fake_end_label(vk::CommandBuffer command_buffer) {}

fn void instance_debug_messenger_dispatch_requires_complete_pair() @test {
    gpu::vk::VkInstanceDispatch empty;
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&empty), "empty pair is unavailable");

    gpu::vk::VkInstanceDispatch create_only = {
        .debug.create_debug_messenger = &fake_create_debug_messenger,
    };
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&create_only), "create-only pair is unavailable");

    gpu::vk::VkInstanceDispatch destroy_only = {
        .debug.destroy_debug_messenger = &fake_destroy_debug_messenger,
    };
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&destroy_only), "destroy-only pair is unavailable");

    gpu::vk::VkInstanceDispatch complete = {
        .debug = {
            .create_debug_messenger  = &fake_create_debug_messenger,
            .destroy_debug_messenger = &fake_destroy_debug_messenger,
        },
    };
    assert(gpu::vk::instance_dispatch_supports_debug_messenger(&complete), "complete pair is available");
}

fn void device_label_dispatch_requires_complete_pair() @test {
    gpu::vk::VkDeviceDispatch empty;
    assert(!gpu::vk::device_dispatch_supports_labels(&empty), "empty pair is unavailable");

    gpu::vk::VkDeviceDispatch begin_only = {
        .debug.cmd_begin_label = &fake_begin_label,
    };
    assert(!gpu::vk::device_dispatch_supports_labels(&begin_only), "begin-only pair is unavailable");

    gpu::vk::VkDeviceDispatch end_only = {
        .debug.cmd_end_label = &fake_end_label,
    };
    assert(!gpu::vk::device_dispatch_supports_labels(&end_only), "end-only pair is unavailable");

    gpu::vk::VkDeviceDispatch complete = {
        .debug = {
            .cmd_begin_label = &fake_begin_label,
            .cmd_end_label   = &fake_end_label,
        },
    };
    assert(gpu::vk::device_dispatch_supports_labels(&complete), "complete pair is available");
}
