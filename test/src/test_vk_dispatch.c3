module gpu_test;

import gpu;
import gpu::internal::vk @public;
import vk;

struct DispatchLookupCounts {
    uint instance_debug;
    uint instance_surface;
    uint instance_win32_surface;
    uint instance_wayland_surface;
    uint instance_xlib_surface;
    uint device_debug;
    uint device_presentation;
    uint dynamic_color;
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
        && self.dynamic_color == 0
        && self.unexpected == 0;
}

fn void fake_void_proc() {}

fn bool supports_no_instance_extensions(ZString name) => false;

fn bool supports_all_instance_extensions(ZString name) => true;

fn bool supports_win32_instance_extensions(ZString name) {
    return name.str_view() == gpu::internal::vk::SURFACE_EXTENSION.str_view()
        || name.str_view() == gpu::internal::vk::WIN32_SURFACE_EXTENSION.str_view();
}

fn bool supports_surface_maintenance_without_capabilities2(ZString name) {
    return name.str_view() == gpu::internal::vk::SURFACE_EXTENSION.str_view()
        || name.str_view() == gpu::internal::vk::SURFACE_MAINTENANCE_EXTENSION.str_view();
}

fn void capture_instance_message(
    gpu::DebugMessage* message,
    void* user_data,
) {}

fn void runtime_policies_plan_layers_and_debug_utils_independently() @test {
    vk::DebugUtilsMessengerCreateInfoEXT messenger_info;
    gpu::RuntimeDesc desc;
    gpu::internal::vk::VkInstanceDesc plan = gpu::internal::vk::runtime_instance_desc(
        &desc,
        &messenger_info,
    );
    assert(!plan.enable_validation, "trusted defaults do not request validation layers");
    assert(!plan.enable_debug_utils, "trusted defaults do not request debug utils");
    assert(plan.debug_messenger_info == null, "trusted defaults omit messenger chaining");

    desc.contract_validation = gpu::ContractValidation.FULL;
    plan = gpu::internal::vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation, "contract and lifetime policies do not request layers");
    assert(!plan.enable_debug_utils, "contract and lifetime policies do not request debug utils");

    desc.debug_callback = &capture_instance_message;
    plan = gpu::internal::vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation, "callback-only configuration does not request layers");
    assert(plan.enable_debug_utils, "callback-only configuration requests debug utils");
    assert(plan.debug_messenger_info == null, "callback-only configuration does not chain a layer messenger");

    desc.debug_callback = null;
    desc.enable_debug_names = true;
    plan = gpu::internal::vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation, "names-only configuration does not request layers");
    assert(plan.enable_debug_utils, "names-only configuration requests debug utils");

    desc.enable_debug_names = false;
    desc.enable_vulkan_validation = true;
    plan = gpu::internal::vk::runtime_instance_desc(&desc, &messenger_info);
    assert(plan.enable_validation, "the Vulkan validation field alone requests layers");
    assert(plan.enable_debug_utils, "Vulkan validation requests debug utils");
    assert(plan.debug_messenger_info == &messenger_info, "Vulkan validation chains the messenger");
}

fn void instance_capabilities_require_request_and_support() @test {
    gpu::internal::vk::VkInstanceDesc desc = {
        .enable_debug_utils        = true,
        .enable_available_surfaces = true,
    };
    gpu::internal::vk::VkInstanceCapabilities capabilities =
        gpu::internal::vk::resolve_instance_capabilities(
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

    capabilities = gpu::internal::vk::resolve_instance_capabilities(
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

    capabilities = gpu::internal::vk::resolve_instance_capabilities(
        &desc,
        &supports_win32_instance_extensions,
    );
    assert(capabilities.surface, "supported base surface discovery is enabled");
    assert(!capabilities.surface_maintenance,
        "missing optional surface maintenance stays disabled");
    assert(capabilities.win32_surface, "supported Win32 surface support is enabled");
    assert(!capabilities.wayland_surface, "unsupported Wayland surface support remains disabled");
    assert(!capabilities.xlib_surface, "unsupported Xlib surface support remains disabled");

    capabilities = gpu::internal::vk::resolve_instance_capabilities(
        &desc,
        &supports_surface_maintenance_without_capabilities2,
    );
    assert(!capabilities.surface_maintenance,
        "surface maintenance requires get-surface-capabilities2");

    desc.enable_available_surfaces = false;
    capabilities = gpu::internal::vk::resolve_instance_capabilities(
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
    gpu::internal::vk::VkInstanceCapabilities capabilities;

    assert(
        !gpu::internal::vk::should_create_debug_messenger(capabilities, true),
        "unsupported debug utils skip messenger creation"
    );
    assert(
        gpu::internal::vk::should_report_debug_messenger_unavailable(capabilities, true),
        "requested unavailable debug utils require a diagnostic"
    );
    assert(
        !gpu::internal::vk::should_report_debug_messenger_unavailable(capabilities, false),
        "unrequested debug utils remain silent"
    );

    capabilities.debug_utils = true;
    assert(
        gpu::internal::vk::should_create_debug_messenger(capabilities, true),
        "enabled debug utils permit requested messenger creation"
    );
    assert(
        !gpu::internal::vk::should_create_debug_messenger(capabilities, false),
        "unrequested messenger creation remains disabled"
    );
    assert(
        !gpu::internal::vk::should_report_debug_messenger_unavailable(capabilities, true),
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
    } else if (proc_name == "vkCmdSetColorBlendEnableEXT"
        || proc_name == "vkCmdSetColorBlendEquationEXT"
        || proc_name == "vkCmdSetColorWriteMaskEXT") {
        counts.dynamic_color++;
    } else {
        counts.unexpected++;
    }
    return (vk::PFN_vkVoidFunction)&fake_void_proc;
}

fn void instance_dispatch_loads_only_selected_groups() @test {
    gpu::internal::vk::VkInstanceDispatchPlan plan;
    DispatchLookupCounts counts;

    counts = {};
    (void)gpu::internal::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.is_empty(), "disabled instance groups perform no lookups");

    counts = {};
    plan.debug = true;
    (void)gpu::internal::vk::load_vk_instance_dispatch(
        (vk::Instance)&counts,
        &recording_instance_proc_loader,
        plan,
    );
    assert(counts.instance_debug == 2, "debug selection loads its complete pair");
    assert(counts.instance_surface == 0, "debug selection does not load surface procedures");
    assert(counts.unexpected == 0, "debug selection loads no unrelated procedures");

    counts = {};
    plan = { .surface = true };
    (void)gpu::internal::vk::load_vk_instance_dispatch(
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
    (void)gpu::internal::vk::load_vk_instance_dispatch(
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
    gpu::internal::vk::VkDeviceDispatchPlan plan;
    DispatchLookupCounts counts;

    counts = {};
    (void)gpu::internal::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 0
        && counts.device_presentation == 0
        && counts.dynamic_color == 3
        && counts.unexpected == 0,
        "the device baseline loads only the required color procedures");

    counts = {};
    plan.debug = true;
    (void)gpu::internal::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 3, "debug selection loads only device diagnostics");
    assert(counts.device_presentation == 0, "debug selection does not load presentation procedures");
    assert(counts.dynamic_color == 3, "debug selection retains baseline color procedures");
    assert(counts.unexpected == 0, "debug selection loads no unrelated procedures");

    counts = {};
    plan = { .presentation = true };
    (void)gpu::internal::vk::load_vk_device_dispatch(
        (vk::Device)&counts,
        &recording_device_proc_loader,
        plan,
    );
    assert(counts.device_debug == 0, "presentation selection does not load debug procedures");
    assert(counts.device_presentation == 5, "presentation selection loads only device WSI");
    assert(counts.dynamic_color == 3, "presentation selection retains baseline color procedures");
    assert(counts.unexpected == 0, "presentation selection loads no unrelated procedures");
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
    gpu::internal::vk::VkInstanceDispatch empty;
    assert(!gpu::internal::vk::instance_dispatch_supports_debug_messenger(&empty), "empty pair is unavailable");

    gpu::internal::vk::VkInstanceDispatch create_only = {
        .debug.create_debug_messenger = &fake_create_debug_messenger,
    };
    assert(!gpu::internal::vk::instance_dispatch_supports_debug_messenger(&create_only), "create-only pair is unavailable");

    gpu::internal::vk::VkInstanceDispatch destroy_only = {
        .debug.destroy_debug_messenger = &fake_destroy_debug_messenger,
    };
    assert(!gpu::internal::vk::instance_dispatch_supports_debug_messenger(&destroy_only), "destroy-only pair is unavailable");

    gpu::internal::vk::VkInstanceDispatch complete = {
        .debug = {
            .create_debug_messenger  = &fake_create_debug_messenger,
            .destroy_debug_messenger = &fake_destroy_debug_messenger,
        },
    };
    assert(gpu::internal::vk::instance_dispatch_supports_debug_messenger(&complete), "complete pair is available");
}

fn void device_label_dispatch_requires_complete_pair() @test {
    gpu::internal::vk::VkDeviceDispatch empty;
    assert(!gpu::internal::vk::device_dispatch_supports_labels(&empty), "empty pair is unavailable");

    gpu::internal::vk::VkDeviceDispatch begin_only = {
        .debug.cmd_begin_label = &fake_begin_label,
    };
    assert(!gpu::internal::vk::device_dispatch_supports_labels(&begin_only), "begin-only pair is unavailable");

    gpu::internal::vk::VkDeviceDispatch end_only = {
        .debug.cmd_end_label = &fake_end_label,
    };
    assert(!gpu::internal::vk::device_dispatch_supports_labels(&end_only), "end-only pair is unavailable");

    gpu::internal::vk::VkDeviceDispatch complete = {
        .debug = {
            .cmd_begin_label = &fake_begin_label,
            .cmd_end_label   = &fake_end_label,
        },
    };
    assert(gpu::internal::vk::device_dispatch_supports_labels(&complete), "complete pair is available");
}

fn void device_color_dispatch_requires_complete_set() @test {
    gpu::internal::vk::VkDeviceDispatchPlan plan;
    DispatchLookupCounts counts;
    gpu::internal::vk::VkDeviceDispatch complete =
        gpu::internal::vk::load_vk_device_dispatch(
            (vk::Device)&counts,
            &recording_device_proc_loader,
            plan,
        );
    assert(
        gpu::internal::vk::device_dispatch_supports_dynamic_color(&complete),
        "all three color commands make the dispatch available"
    );

    gpu::internal::vk::VkDeviceDispatch missing = complete;
    missing.dynamic_color.set_blend_enable = null;
    assert(!gpu::internal::vk::device_dispatch_supports_dynamic_color(&missing));
    missing = complete;
    missing.dynamic_color.set_blend_equation = null;
    assert(!gpu::internal::vk::device_dispatch_supports_dynamic_color(&missing));
    missing = complete;
    missing.dynamic_color.set_write_mask = null;
    assert(!gpu::internal::vk::device_dispatch_supports_dynamic_color(&missing));
}
