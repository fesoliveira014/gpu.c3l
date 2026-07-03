module gpu::vk;

import vk;
import std::io;

<* Messenger callback: prints validation messages to stderr and returns
   VK_FALSE, as the spec requires from application callbacks. *>
fn uint debug_callback(
    vk::DebugUtilsMessageSeverityFlagsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT     message_type,
    vk::DebugUtilsMessengerCallbackDataEXT* data,
    void*                                 user_data
) @private {
    ZString msg = data != null && data.p_message != null ? (ZString)data.p_message : (ZString)"<no message>";
    io::eprintfn("vk: %s", msg);
    return vk::FALSE;
}

<* Install the debug-utils messenger. Best-effort: a missing entry point or
   create failure leaves messenger null and falls back to layer stderr. *>
fn void setup_debug_messenger(VkDeviceState* state) @private {
    vk::load_extensions(state.instance);
    if (vk::extensions.vk_create_debug_utils_messenger_ext == null) {
        io::eprintfn("vk: debug-utils messenger entry point unavailable; relying on layer stderr");
        return;
    }

    uint severity_flags = (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT
                        | (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT;
    uint type_flags = (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.GENERAL_BIT_EXT
                    | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT
                    | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.PERFORMANCE_BIT_EXT;

    vk::DebugUtilsMessengerCreateInfoEXT create_info = vk::debug_utils_messenger_create_info_ext()
        .set_message_severity((vk::DebugUtilsMessageSeverityFlagsEXT)severity_flags)
        .set_message_type((vk::DebugUtilsMessageTypeFlagsEXT)type_flags)
        .set_pfn_user_callback((vk::PFN_vkDebugUtilsMessengerCallbackEXT)&debug_callback);

    vk::Result result = vk::extensions.vk_create_debug_utils_messenger_ext(state.instance, &create_info, null, &state.messenger);
    if (result != vk::Result.SUCCESS) {
        io::eprintfn("vk: vkCreateDebugUtilsMessengerEXT failed (%d); relying on layer stderr", (int)result);
        state.messenger = (vk::DebugUtilsMessengerEXT)null;
    }
}

<* Best-effort object naming: silently a no-op when the debug-utils entry
   point is unavailable or no name was given. *>
fn void set_object_name(VkDeviceState* state, vk::ObjectType object_type, ulong object_handle, ZString name) {
    if (name == null) return;
    if (vk::extensions.vk_set_debug_utils_object_name_ext == null) return;
    vk::DebugUtilsObjectNameInfoEXT info = vk::debug_utils_object_name_info_ext();
    info.object_type = object_type;
    info.object_handle = object_handle;
    info.p_object_name = (char*)name;
    if (catch vk::try_set_debug_utils_object_name_ext(state.device, &info)) return;
}

fn void destroy_debug_messenger(VkDeviceState* state) @private {
    if (state.messenger == null) return;
    if (vk::extensions.vk_destroy_debug_utils_messenger_ext != null) {
        vk::extensions.vk_destroy_debug_utils_messenger_ext(state.instance, state.messenger, null);
    }
    state.messenger = (vk::DebugUtilsMessengerEXT)null;
}
