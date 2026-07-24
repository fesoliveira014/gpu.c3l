module gpu::internal::vk @private;

import vk;
import vma;

alias InstanceProcLoader = fn vk::PFN_vkVoidFunction(vk::Instance, char*);
alias DeviceProcLoader = fn vk::PFN_vkVoidFunction(vk::Device, char*);

struct VkInstanceDispatchPlan {
    bool debug;
    bool surface;
    bool win32_surface;
    bool wayland_surface;
    bool xlib_surface;
}

struct VkDeviceDispatchPlan {
    bool debug;
    bool presentation;
    bool descriptor_buffer;
    bool generated_work;
}

struct VkInstanceDebugDispatch {
    vk::PFN_vkCreateDebugUtilsMessengerEXT  create_debug_messenger;
    vk::PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger;
}

struct VkInstanceSurfaceDispatch {
    vk::PFN_vkCreateWin32SurfaceKHR                   create_win32_surface;
    vk::PFN_vkCreateWaylandSurfaceKHR                 create_wayland_surface;
    vk::PFN_vkCreateXlibSurfaceKHR                    create_xlib_surface;
    vk::PFN_vkDestroySurfaceKHR                       destroy_surface;
    vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR      get_surface_support;
    vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR get_surface_capabilities;
    vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR      get_surface_formats;
    vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR get_surface_present_modes;
}

struct VkInstanceDispatch {
    VkInstanceDebugDispatch   debug;
    VkInstanceSurfaceDispatch surface;
}

struct VkDeviceDebugDispatch {
    vk::PFN_vkSetDebugUtilsObjectNameEXT set_object_name;
    vk::PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin_label;
    vk::PFN_vkCmdEndDebugUtilsLabelEXT   cmd_end_label;
}

struct VkDevicePresentationDispatch {
    vk::PFN_vkCreateSwapchainKHR    create_swapchain;
    vk::PFN_vkDestroySwapchainKHR   destroy_swapchain;
    vk::PFN_vkGetSwapchainImagesKHR get_swapchain_images;
    vk::PFN_vkAcquireNextImageKHR   acquire_next_image;
    vk::PFN_vkQueuePresentKHR       queue_present;
}

struct VkDescriptorBufferDispatch {
    vk::PFN_vkGetDescriptorSetLayoutSizeEXT          get_descriptor_set_layout_size;
    vk::PFN_vkGetDescriptorSetLayoutBindingOffsetEXT get_descriptor_set_layout_binding_offset;
    vk::PFN_vkGetDescriptorEXT                       get_descriptor;
    vk::PFN_vkCmdBindDescriptorBuffersEXT            cmd_bind_descriptor_buffers;
    vk::PFN_vkCmdSetDescriptorBufferOffsetsEXT       cmd_set_descriptor_buffer_offsets;
}

struct VkGeneratedWorkDispatch {
    vk::PFN_vkCmdExecuteGeneratedCommandsEXT         cmd_execute_generated_commands;
    vk::PFN_vkGetGeneratedCommandsMemoryRequirementsEXT get_memory_requirements;
    vk::PFN_vkCreateIndirectCommandsLayoutEXT        create_layout;
    vk::PFN_vkDestroyIndirectCommandsLayoutEXT       destroy_layout;
}

struct VkDynamicColorDispatch {
    vk::PFN_vkCmdSetColorBlendEnableEXT   set_blend_enable;
    vk::PFN_vkCmdSetColorBlendEquationEXT set_blend_equation;
    vk::PFN_vkCmdSetColorWriteMaskEXT     set_write_mask;
}

struct VkDeviceDispatch {
    VkDeviceDebugDispatch        debug;
    VkDevicePresentationDispatch presentation;
    VkDescriptorBufferDispatch   descriptor_buffer;
    VkGeneratedWorkDispatch      generated_work;
    VkDynamicColorDispatch       dynamic_color;
}

fn VkInstanceDispatch load_vk_instance_dispatch(
    vk::Instance instance,
    InstanceProcLoader loader,
    VkInstanceDispatchPlan plan,
) {
    VkInstanceDispatch dispatch;
    if (plan.debug) {
        dispatch.debug = {
            .create_debug_messenger = (vk::PFN_vkCreateDebugUtilsMessengerEXT)loader(
                instance,
                "vkCreateDebugUtilsMessengerEXT",
            ),
            .destroy_debug_messenger = (vk::PFN_vkDestroyDebugUtilsMessengerEXT)loader(
                instance,
                "vkDestroyDebugUtilsMessengerEXT",
            ),
        };
    }
    if (plan.surface) {
        dispatch.surface.destroy_surface = (vk::PFN_vkDestroySurfaceKHR)loader(
            instance,
            "vkDestroySurfaceKHR",
        );
        dispatch.surface.get_surface_support = (vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceSupportKHR",
        );
        dispatch.surface.get_surface_capabilities = (vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        );
        dispatch.surface.get_surface_formats = (vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfaceFormatsKHR",
        );
        dispatch.surface.get_surface_present_modes = (vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)loader(
            instance,
            "vkGetPhysicalDeviceSurfacePresentModesKHR",
        );
    }
    if (plan.surface && plan.win32_surface) {
        dispatch.surface.create_win32_surface = (vk::PFN_vkCreateWin32SurfaceKHR)loader(
            instance,
            "vkCreateWin32SurfaceKHR",
        );
    }
    if (plan.surface && plan.wayland_surface) {
        dispatch.surface.create_wayland_surface = (vk::PFN_vkCreateWaylandSurfaceKHR)loader(
            instance,
            "vkCreateWaylandSurfaceKHR",
        );
    }
    if (plan.surface && plan.xlib_surface) {
        dispatch.surface.create_xlib_surface = (vk::PFN_vkCreateXlibSurfaceKHR)loader(
            instance,
            "vkCreateXlibSurfaceKHR",
        );
    }
    return dispatch;
}

fn VkDeviceDispatch load_vk_device_dispatch(
    vk::Device device,
    DeviceProcLoader loader,
    VkDeviceDispatchPlan plan,
) {
    VkDeviceDispatch dispatch;
    if (plan.debug) {
        dispatch.debug = {
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
        };
    }
    if (plan.presentation) {
        dispatch.presentation = {
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
        };
    }
    if (plan.descriptor_buffer) {
        dispatch.descriptor_buffer = {
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
    if (plan.generated_work) {
        dispatch.generated_work = {
            .cmd_execute_generated_commands = (vk::PFN_vkCmdExecuteGeneratedCommandsEXT)loader(
                device,
                "vkCmdExecuteGeneratedCommandsEXT",
            ),
            .get_memory_requirements = (vk::PFN_vkGetGeneratedCommandsMemoryRequirementsEXT)loader(
                device,
                "vkGetGeneratedCommandsMemoryRequirementsEXT",
            ),
            .create_layout = (vk::PFN_vkCreateIndirectCommandsLayoutEXT)loader(
                device,
                "vkCreateIndirectCommandsLayoutEXT",
            ),
            .destroy_layout = (vk::PFN_vkDestroyIndirectCommandsLayoutEXT)loader(
                device,
                "vkDestroyIndirectCommandsLayoutEXT",
            ),
        };
    }
    return dispatch;
}

fn bool instance_dispatch_supports_debug_messenger(VkInstanceDispatch* dispatch) {
    return dispatch.debug.create_debug_messenger != null
        && dispatch.debug.destroy_debug_messenger != null;
}

fn bool device_dispatch_supports_labels(VkDeviceDispatch* dispatch) {
    return dispatch.debug.cmd_begin_label != null
        && dispatch.debug.cmd_end_label != null;
}

fn bool instance_dispatch_supports_presentation(VkInstanceDispatch* dispatch) {
    return dispatch.surface.get_surface_support != null
        && dispatch.surface.get_surface_capabilities != null
        && dispatch.surface.get_surface_formats != null
        && dispatch.surface.get_surface_present_modes != null;
}

fn bool device_dispatch_supports_presentation(VkDeviceDispatch* dispatch) {
    return dispatch.presentation.create_swapchain != null
        && dispatch.presentation.destroy_swapchain != null
        && dispatch.presentation.get_swapchain_images != null
        && dispatch.presentation.acquire_next_image != null
        && dispatch.presentation.queue_present != null;
}

fn bool device_dispatch_supports_descriptor_buffer(VkDeviceDispatch* dispatch) {
    return dispatch.descriptor_buffer.get_descriptor_set_layout_size != null
        && dispatch.descriptor_buffer.get_descriptor_set_layout_binding_offset != null
        && dispatch.descriptor_buffer.get_descriptor != null
        && dispatch.descriptor_buffer.cmd_bind_descriptor_buffers != null
        && dispatch.descriptor_buffer.cmd_set_descriptor_buffer_offsets != null;
}
fn bool device_dispatch_supports_generated_work(VkDeviceDispatch* dispatch) {
    return dispatch.generated_work.cmd_execute_generated_commands != null
        && dispatch.generated_work.get_memory_requirements != null
        && dispatch.generated_work.create_layout != null
        && dispatch.generated_work.destroy_layout != null;
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
