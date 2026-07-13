module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;

const ZString VALIDATION_LAYER      = "VK_LAYER_KHRONOS_validation";
const ZString DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils";
const ZString SURFACE_EXTENSION     = "VK_KHR_surface";

// Platform surface extensions enabled opportunistically (whatever the loader
// offers), so one binary serves Wayland-or-X11 sessions.
const ZString[3] PLATFORM_SURFACE_EXTENSIONS = {
    "VK_KHR_wayland_surface",
    "VK_KHR_xlib_surface",
    "VK_KHR_win32_surface",
};
const uint MAX_INSTANCE_EXTENSIONS = 8;

fn bool instance_supports_extension(ZString name) @private {
    uint count = 0;
    if (catch vk::try_enumerate_instance_extension_properties(null, &count, null)) return false;
    if (count == 0) return false;

    @pool() {
        vk::ExtensionProperties[] properties = mem::talloc_array(vk::ExtensionProperties, (sz)count);
        if (catch vk::try_enumerate_instance_extension_properties(null, &count, properties.ptr)) return false;
        for (uint i = 0; i < count; i++) {
            if (((ZString)&properties[i].extension_name[0]).str_view() == name.str_view()) return true;
        }
    };
    return false;
}

<* Create the Vulkan instance at API 1.3. Validation enables the Khronos layer
   and debug-utils; presentation enables VK_KHR_surface plus the
   loader-supported platform surface extensions. *>
fn vk::Instance? create_vk_instance(gpu::DeviceDesc* desc, VkDeviceState* state) {
    vk::Instance instance;
    @pool() {
        vk::ApplicationInfo app = vk::application_info()
            .set_application_name(desc.application_name == null ? (char*)"gpu.c3l" : (char*)desc.application_name)
            .set_api_version(API_VERSION_1_3);

        vk::InstanceCreateInfo create_info = vk::instance_create_info().set_application_info(&app);

        ZString[1] layers = { VALIDATION_LAYER };
        ZString[MAX_INSTANCE_EXTENSIONS] extensions;
        uint extension_count = 0;
        bool needs_debug_utils = desc.enable_validation
            || desc.enable_debug_names
            || desc.debug_callback != null;
        bool has_debug_utils = needs_debug_utils
            && instance_supports_extension(DEBUG_UTILS_EXTENSION);
        vk::DebugUtilsMessengerCreateInfoEXT messenger_info;
        if (desc.enable_validation) {
            create_info = create_info.set_enabled_layer_names(layers[..]);
            if (has_debug_utils) {
                messenger_info = debug_messenger_create_info(state);
                create_info = create_info.set_next(&messenger_info);
            }
        }
        if (has_debug_utils) {
            extensions[extension_count++] = DEBUG_UTILS_EXTENSION;
        }
        if (desc.enable_presentation) {
            if (!instance_supports_extension(SURFACE_EXTENSION)) return gpu::UNSUPPORTED_FEATURE~;
            extensions[extension_count++] = SURFACE_EXTENSION;
            foreach (platform_extension : PLATFORM_SURFACE_EXTENSIONS) {
                if (instance_supports_extension(platform_extension)) {
                    extensions[extension_count++] = platform_extension;
                }
            }
        }
        if (extension_count > 0) {
            create_info = create_info.set_enabled_extension_names(extensions[:extension_count]);
        }

        check_backend_create_result(vk::create_instance(&create_info, null, &instance))!;
    };
    return instance;
}
