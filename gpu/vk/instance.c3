module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;

const ZString VALIDATION_LAYER      = "VK_LAYER_KHRONOS_validation";
const ZString DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils";
const ZString SURFACE_EXTENSION     = "VK_KHR_surface";

const ZString[3] PLATFORM_SURFACE_EXTENSIONS = {
    "VK_KHR_wayland_surface",
    "VK_KHR_xlib_surface",
    "VK_KHR_win32_surface",
};
const uint MAX_INSTANCE_EXTENSIONS = 8;

struct VkInstanceDesc {
    bool enable_validation;
    bool enable_debug_utils;
    bool enable_available_surfaces;
    ZString application_name;
    vk::DebugUtilsMessengerCreateInfoEXT* debug_messenger_info;
}

fn bool instance_supports_extension(ZString name) @private {
    uint count = 0;
    if (catch vk::try_enumerate_instance_extension_properties(null, &count, null)) return false;
    if (count == 0) return false;

    @pool() {
        vk::ExtensionProperties[] properties =
            mem::talloc_array(vk::ExtensionProperties, (sz)count);
        if (catch vk::try_enumerate_instance_extension_properties(
            null,
            &count,
            properties.ptr,
        )) return false;
        for (uint i = 0; i < count; i++) {
            if (((ZString)&properties[i].extension_name[0]).str_view()
                == name.str_view()) return true;
        }
    };
    return false;
}

<* Create one Vulkan 1.3 discovery instance from backend-owned configuration. *>
fn vk::Instance? create_vk_instance(VkInstanceDesc* desc) {
    vk::Instance instance;
    @pool() {
        vk::ApplicationInfo app = vk::application_info()
            .set_application_name(
                desc.application_name == null
                    ? (char*)"gpu.c3l"
                    : (char*)desc.application_name,
            )
            .set_api_version(API_VERSION_1_3);
        vk::InstanceCreateInfo create_info =
            vk::instance_create_info().set_application_info(&app);

        ZString[1] layers = { VALIDATION_LAYER };
        ZString[MAX_INSTANCE_EXTENSIONS] extensions;
        uint extension_count = 0;
        bool has_debug_utils = desc.enable_debug_utils
            && instance_supports_extension(DEBUG_UTILS_EXTENSION);
        if (desc.enable_validation) {
            create_info = create_info.set_enabled_layer_names(layers[..]);
            if (has_debug_utils && desc.debug_messenger_info != null) {
                create_info = create_info.set_next(desc.debug_messenger_info);
            }
        }
        if (has_debug_utils) {
            extensions[extension_count++] = DEBUG_UTILS_EXTENSION;
        }

        if (desc.enable_available_surfaces
            && instance_supports_extension(SURFACE_EXTENSION)) {
            extensions[extension_count++] = SURFACE_EXTENSION;
            foreach (platform_extension : PLATFORM_SURFACE_EXTENSIONS) {
                if (instance_supports_extension(platform_extension)) {
                    extensions[extension_count++] = platform_extension;
                }
            }
        }
        if (extension_count > 0) {
            create_info =
                create_info.set_enabled_extension_names(extensions[:extension_count]);
        }

        check_backend_create_result(
            vk::create_instance(&create_info, null, &instance),
        )!;
    };
    return instance;
}
