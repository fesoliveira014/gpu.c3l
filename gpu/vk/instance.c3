module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;

alias InstanceExtensionQuery = fn bool(ZString name);

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

struct VkInstanceCapabilities {
    bool debug_utils;
    bool surface;
    bool win32_surface;
    bool wayland_surface;
    bool xlib_surface;
}

struct VkInstanceCreateResult {
    vk::Instance           instance;
    VkInstanceCapabilities capabilities;
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

fn VkInstanceCapabilities resolve_instance_capabilities(
    VkInstanceDesc* desc,
    InstanceExtensionQuery supports,
) {
    VkInstanceCapabilities capabilities;
    if (desc == null || supports == null) return capabilities;

    capabilities.debug_utils = desc.enable_debug_utils
        && supports(DEBUG_UTILS_EXTENSION);
    capabilities.surface = desc.enable_available_surfaces
        && supports(SURFACE_EXTENSION);
    if (capabilities.surface) {
        capabilities.wayland_surface =
            supports(PLATFORM_SURFACE_EXTENSIONS[0]);
        capabilities.xlib_surface =
            supports(PLATFORM_SURFACE_EXTENSIONS[1]);
        capabilities.win32_surface =
            supports(PLATFORM_SURFACE_EXTENSIONS[2]);
    }
    return capabilities;
}

fn bool should_create_debug_messenger(
    VkInstanceCapabilities capabilities,
    bool requested,
) => requested && capabilities.debug_utils;

<* Create one Vulkan 1.3 discovery instance from backend-owned configuration. *>
fn VkInstanceCreateResult? create_vk_instance(VkInstanceDesc* desc) {
    VkInstanceCreateResult result;
    VkInstanceCapabilities capabilities =
        resolve_instance_capabilities(desc, &instance_supports_extension);
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
        if (desc.enable_validation) {
            create_info = create_info.set_enabled_layer_names(layers[..]);
            if (capabilities.debug_utils && desc.debug_messenger_info != null) {
                create_info = create_info.set_next(desc.debug_messenger_info);
            }
        }
        if (capabilities.debug_utils) {
            extensions[extension_count++] = DEBUG_UTILS_EXTENSION;
        }
        if (capabilities.surface) {
            extensions[extension_count++] = SURFACE_EXTENSION;
            if (capabilities.wayland_surface) {
                extensions[extension_count++] = PLATFORM_SURFACE_EXTENSIONS[0];
            }
            if (capabilities.xlib_surface) {
                extensions[extension_count++] = PLATFORM_SURFACE_EXTENSIONS[1];
            }
            if (capabilities.win32_surface) {
                extensions[extension_count++] = PLATFORM_SURFACE_EXTENSIONS[2];
            }
        }
        if (extension_count > 0) {
            create_info =
                create_info.set_enabled_extension_names(extensions[:extension_count]);
        }

        check_backend_create_result(
            vk::create_instance(&create_info, null, &result.instance),
        )!;
    };
    result.capabilities = capabilities;
    return result;
}
