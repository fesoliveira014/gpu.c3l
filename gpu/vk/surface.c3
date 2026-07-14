module gpu::vk @private;

import gpu @public;
import vk;

fn void load_runtime_surface_dispatch(VkRuntimeState* state) {
    state.create_win32_surface = (vk::PFN_vkCreateWin32SurfaceKHR)vk::get_instance_proc_addr(
        state.instance,
        "vkCreateWin32SurfaceKHR",
    );
    state.create_wayland_surface = (vk::PFN_vkCreateWaylandSurfaceKHR)vk::get_instance_proc_addr(
        state.instance,
        "vkCreateWaylandSurfaceKHR",
    );
    state.create_xlib_surface = (vk::PFN_vkCreateXlibSurfaceKHR)vk::get_instance_proc_addr(
        state.instance,
        "vkCreateXlibSurfaceKHR",
    );
    state.destroy_surface = (vk::PFN_vkDestroySurfaceKHR)vk::get_instance_proc_addr(
        state.instance,
        "vkDestroySurfaceKHR",
    );
    state.get_surface_support = (vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vk::get_instance_proc_addr(
        state.instance,
        "vkGetPhysicalDeviceSurfaceSupportKHR",
    );
}

fn void*? vk_create_win32_surface(
    void* backend_state,
    void* instance,
    void* window,
) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || instance == null || window == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (state.create_win32_surface == null || state.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    vk::Win32SurfaceCreateInfoKHR info = {
        .s_type    = vk::StructureType.WIN32_SURFACE_CREATE_INFO_KHR,
        .hinstance = instance,
        .hwnd      = window,
    };
    vk::SurfaceKHR surface;
    check_surface_result(state.create_win32_surface(
        state.instance,
        &info,
        null,
        &surface,
    ))!;
    return (void*)surface;
}

fn void*? vk_create_wayland_surface(
    void* backend_state,
    void* display,
    void* window_surface,
) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || display == null || window_surface == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (state.create_wayland_surface == null || state.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    vk::WaylandSurfaceCreateInfoKHR info = {
        .s_type  = vk::StructureType.WAYLAND_SURFACE_CREATE_INFO_KHR,
        .display = display,
        .surface = window_surface,
    };
    vk::SurfaceKHR surface;
    check_surface_result(state.create_wayland_surface(
        state.instance,
        &info,
        null,
        &surface,
    ))!;
    return (void*)surface;
}

fn void*? vk_create_x11_surface(
    void* backend_state,
    void* display,
    ulong window,
) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || display == null || window == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (state.create_xlib_surface == null || state.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    vk::XlibSurfaceCreateInfoKHR info = {
        .s_type = vk::StructureType.XLIB_SURFACE_CREATE_INFO_KHR,
        .dpy    = display,
        .window = window,
    };
    vk::SurfaceKHR surface;
    check_surface_result(state.create_xlib_surface(
        state.instance,
        &info,
        null,
        &surface,
    ))!;
    return (void*)surface;
}

fn void vk_destroy_surface(void* backend_state, void* backend_surface) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || backend_surface == null || state.destroy_surface == null) {
        return;
    }
    state.destroy_surface(
        state.instance,
        (vk::SurfaceKHR)backend_surface,
        null,
    );
}

fn bool? vk_supports_presentation(
    void* backend_state,
    uint adapter_index,
    void* backend_surface,
) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || backend_surface == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.get_surface_support == null) return false;

    vk::PhysicalDevice physical_device = state.adapters[adapter_index].physical_device;
    if (!device_supports_extension(physical_device, SWAPCHAIN_EXTENSION)) return false;

    QueueFamilies? selection = select_queue_families_for_surface(
        state,
        physical_device,
        (vk::SurfaceKHR)backend_surface,
    );
    if (catch selection_fault = selection) {
        if (selection_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return selection_fault~;
    }
    return true;
}
