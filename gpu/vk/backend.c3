module gpu::vk @private;

import vk;
import vma;

alias InstanceProcLoader = fn vk::PFN_vkVoidFunction(vk::Instance, char*);
alias DeviceProcLoader = fn vk::PFN_vkVoidFunction(vk::Device, char*);

struct VkInstanceDispatch {
    vk::PFN_vkCreateDebugUtilsMessengerEXT             create_debug_messenger;
    vk::PFN_vkDestroyDebugUtilsMessengerEXT            destroy_debug_messenger;
    vk::PFN_vkCreateWin32SurfaceKHR                    create_win32_surface;
    vk::PFN_vkCreateWaylandSurfaceKHR                  create_wayland_surface;
    vk::PFN_vkCreateXlibSurfaceKHR                     create_xlib_surface;
    vk::PFN_vkDestroySurfaceKHR                        destroy_surface;
    vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR       get_surface_support;
    vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR  get_surface_capabilities;
    vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR       get_surface_formats;
    vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR  get_surface_present_modes;
}

struct VkDeviceDispatch {
    vk::PFN_vkCreateSwapchainKHR                       create_swapchain;
    vk::PFN_vkDestroySwapchainKHR                      destroy_swapchain;
    vk::PFN_vkGetSwapchainImagesKHR                    get_swapchain_images;
    vk::PFN_vkAcquireNextImageKHR                      acquire_next_image;
    vk::PFN_vkQueuePresentKHR                          queue_present;
    vk::PFN_vkSetDebugUtilsObjectNameEXT               set_object_name;
    vk::PFN_vkCmdBeginDebugUtilsLabelEXT               cmd_begin_label;
    vk::PFN_vkCmdEndDebugUtilsLabelEXT                 cmd_end_label;
    vk::PFN_vkGetDescriptorSetLayoutSizeEXT            get_descriptor_set_layout_size;
    vk::PFN_vkGetDescriptorSetLayoutBindingOffsetEXT   get_descriptor_set_layout_binding_offset;
    vk::PFN_vkGetDescriptorEXT                         get_descriptor;
    vk::PFN_vkCmdBindDescriptorBuffersEXT              cmd_bind_descriptor_buffers;
    vk::PFN_vkCmdSetDescriptorBufferOffsetsEXT         cmd_set_descriptor_buffer_offsets;
}

fn VkInstanceDispatch load_vk_instance_dispatch(
    vk::Instance instance,
    InstanceProcLoader loader,
) {
    return {
        .create_debug_messenger = (vk::PFN_vkCreateDebugUtilsMessengerEXT)loader(
            instance,
            "vkCreateDebugUtilsMessengerEXT",
        ),
        .destroy_debug_messenger = (vk::PFN_vkDestroyDebugUtilsMessengerEXT)loader(
            instance,
            "vkDestroyDebugUtilsMessengerEXT",
        ),
        .create_win32_surface = (vk::PFN_vkCreateWin32SurfaceKHR)loader(
            instance,
            "vkCreateWin32SurfaceKHR",
        ),
        .create_wayland_surface = (vk::PFN_vkCreateWaylandSurfaceKHR)loader(
            instance,
            "vkCreateWaylandSurfaceKHR",
        ),
        .create_xlib_surface = (vk::PFN_vkCreateXlibSurfaceKHR)loader(
            instance,
            "vkCreateXlibSurfaceKHR",
        ),
        .destroy_surface = (vk::PFN_vkDestroySurfaceKHR)loader(
            instance,
            "vkDestroySurfaceKHR",
        ),
        .get_surface_support = (vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceSupportKHR",
        ),
        .get_surface_capabilities = (vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        ),
        .get_surface_formats = (vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceFormatsKHR",
        ),
        .get_surface_present_modes = (vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfacePresentModesKHR",
        ),
    };
}

fn VkDeviceDispatch load_vk_device_dispatch(
    vk::Device device,
    DeviceProcLoader loader,
) {
    return {
        .create_swapchain = (vk::PFN_vkCreateSwapchainKHR)loader(
            device,
            "vkCreateSwapchainKHR",
        ),
        .destroy_swapchain = (vk::PFN_vkDestroySwapchainKHR)loader(
            device,
            "vkDestroySwapchainKHR",
        ),
        .get_swapchain_images = (vk::PFN_vkGetSwapchainImagesKHR)loader(
            device,
            "vkGetSwapchainImagesKHR",
        ),
        .acquire_next_image = (vk::PFN_vkAcquireNextImageKHR)loader(
            device,
            "vkAcquireNextImageKHR",
        ),
        .queue_present = (vk::PFN_vkQueuePresentKHR)loader(
            device,
            "vkQueuePresentKHR",
        ),
        .set_object_name = (vk::PFN_vkSetDebugUtilsObjectNameEXT)loader(
            device,
            "vkSetDebugUtilsObjectNameEXT",
        ),
        .cmd_begin_label = (vk::PFN_vkCmdBeginDebugUtilsLabelEXT)loader(
            device,
            "vkCmdBeginDebugUtilsLabelEXT",
        ),
        .cmd_end_label = (vk::PFN_vkCmdEndDebugUtilsLabelEXT)loader(
            device,
            "vkCmdEndDebugUtilsLabelEXT",
        ),
        .get_descriptor_set_layout_size = (vk::PFN_vkGetDescriptorSetLayoutSizeEXT)loader(
            device,
            "vkGetDescriptorSetLayoutSizeEXT",
        ),
        .get_descriptor_set_layout_binding_offset = (vk::PFN_vkGetDescriptorSetLayoutBindingOffsetEXT)loader(
            device,
            "vkGetDescriptorSetLayoutBindingOffsetEXT",
        ),
        .get_descriptor = (vk::PFN_vkGetDescriptorEXT)loader(
            device,
            "vkGetDescriptorEXT",
        ),
        .cmd_bind_descriptor_buffers = (vk::PFN_vkCmdBindDescriptorBuffersEXT)loader(
            device,
            "vkCmdBindDescriptorBuffersEXT",
        ),
        .cmd_set_descriptor_buffer_offsets = (vk::PFN_vkCmdSetDescriptorBufferOffsetsEXT)loader(
            device,
            "vkCmdSetDescriptorBufferOffsetsEXT",
        ),
    };
}

fn bool instance_dispatch_supports_debug_messenger(VkInstanceDispatch* dispatch) {
    return dispatch.create_debug_messenger != null
        && dispatch.destroy_debug_messenger != null;
}

fn bool device_dispatch_supports_labels(VkDeviceDispatch* dispatch) {
    return dispatch.cmd_begin_label != null
        && dispatch.cmd_end_label != null;
}

fn bool instance_dispatch_supports_presentation(VkInstanceDispatch* dispatch) {
    return dispatch.get_surface_support != null
        && dispatch.get_surface_capabilities != null
        && dispatch.get_surface_formats != null
        && dispatch.get_surface_present_modes != null;
}

fn bool device_dispatch_supports_presentation(VkDeviceDispatch* dispatch) {
    return dispatch.create_swapchain != null
        && dispatch.destroy_swapchain != null
        && dispatch.get_swapchain_images != null
        && dispatch.acquire_next_image != null
        && dispatch.queue_present != null;
}

fn bool device_dispatch_supports_descriptor_buffer(VkDeviceDispatch* dispatch) {
    return dispatch.get_descriptor_set_layout_size != null
        && dispatch.get_descriptor_set_layout_binding_offset != null
        && dispatch.get_descriptor != null
        && dispatch.cmd_bind_descriptor_buffers != null
        && dispatch.cmd_set_descriptor_buffer_offsets != null;
}

<* Loader instance version, or 0 if unavailable. A real `vk` call, so it pulls
   libvulkan.so.1 into the link. *>
fn uint probe_vulkan_version() @private {
    uint version = 0;
    if (catch vk::try_enumerate_instance_version(&version)) return 0;
    return version;
}

<* CPU-only VMA virtual-block round-trip. A real `vma` call, so it pulls
   libVulkanMemoryAllocator.a and stdc++ into the link. *>
fn bool probe_vma_allocator() @private {
    vma::VirtualBlockCreateInfo info = { .size = 4096 };
    if (try block = vma::try_create_virtual_block(&info)) {
        defer block.destroy();
        return true;
    }
    return false;
}
