module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;
import std::io;

struct VkRuntimeState {
    vk::Instance                            instance;
    vk::DebugUtilsMessengerEXT              messenger;
    vk::PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger;
    gpu::DebugMessageCallback               debug_callback;
    void*                                   debug_user_data;
    VkAdapterState[]                        adapters;
    bool                                    validation;
}

fn uint route_runtime_debug_message(
    VkRuntimeState* state,
    vk::DebugUtilsMessageSeverityFlagsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT message_type,
    vk::DebugUtilsMessengerCallbackDataEXT* data,
) {
    ZString text = data != null ? (ZString)data.p_message : null;
    if (state == null || state.debug_callback == null) {
        io::eprintfn("vk: %s", text == null ? (ZString)"<no message>" : text);
        return vk::FALSE;
    }
    gpu::DebugMessage message = {
        .severity             = debug_message_severity(severity),
        .category             = debug_message_category(message_type),
        .resource             = debug_resource(data),
        .backend_text         = text,
        .validation_id_name   = data != null ? (ZString)data.p_message_id_name : null,
        .validation_id_number = data != null ? data.message_id_number : 0,
    };
    gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    return vk::FALSE;
}

fn uint runtime_debug_callback(
    vk::DebugUtilsMessageSeverityFlagsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT message_type,
    vk::DebugUtilsMessengerCallbackDataEXT* data,
    void* user_data,
) {
    return route_runtime_debug_message(
        (VkRuntimeState*)user_data,
        severity,
        message_type,
        data,
    );
}

fn vk::DebugUtilsMessengerCreateInfoEXT runtime_debug_messenger_create_info(
    VkRuntimeState* state,
) {
    uint severity_flags = (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT
        | (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT;
    uint type_flags = (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.GENERAL_BIT_EXT
        | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT
        | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.PERFORMANCE_BIT_EXT;
    return vk::debug_utils_messenger_create_info_ext()
        .set_message_severity((vk::DebugUtilsMessageSeverityFlagsEXT)severity_flags)
        .set_message_type((vk::DebugUtilsMessageTypeFlagsEXT)type_flags)
        .set_pfn_user_callback((vk::PFN_vkDebugUtilsMessengerCallbackEXT)&runtime_debug_callback)
        .set_user_data(state);
}

fn bool create_runtime_debug_messenger(
    VkRuntimeState* state,
    vk::DebugUtilsMessengerCreateInfoEXT* create_info,
) {
    if (state.instance == null || create_info == null) return false;
    vk::PFN_vkCreateDebugUtilsMessengerEXT create_messenger =
        (vk::PFN_vkCreateDebugUtilsMessengerEXT)vk::get_instance_proc_addr(
            state.instance,
            "vkCreateDebugUtilsMessengerEXT",
        );
    state.destroy_messenger =
        (vk::PFN_vkDestroyDebugUtilsMessengerEXT)vk::get_instance_proc_addr(
            state.instance,
            "vkDestroyDebugUtilsMessengerEXT",
        );
    if (create_messenger == null || state.destroy_messenger == null) return false;
    vk::Result result = create_messenger(
        state.instance,
        create_info,
        null,
        &state.messenger,
    );
    if (result == vk::Result.SUCCESS) return true;
    state.messenger = (vk::DebugUtilsMessengerEXT)null;
    return false;
}

fn void destroy_runtime_debug_messenger(VkRuntimeState* state) {
    if (state.messenger == null) return;
    if (state.destroy_messenger != null) {
        state.destroy_messenger(state.instance, state.messenger, null);
    }
    state.messenger = (vk::DebugUtilsMessengerEXT)null;
}

fn void report_runtime_debug_messenger_failure(VkRuntimeState* state) {
    if (state.debug_callback == null) {
        io::eprintfn("vk: debug-utils messenger unavailable; relying on layer stderr");
        return;
    }
    gpu::DebugMessage message = {
        .severity     = gpu::DebugMessageSeverity.WARNING,
        .category     = { .backend },
        .operation    = "create_runtime",
        .backend_text = "debug-utils messenger unavailable",
    };
    gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
}

fn void destroy_vk_runtime_state(VkRuntimeState* state) {
    if (state == null) return;
    destroy_runtime_debug_messenger(state);
    if (state.instance != null) {
        vk::destroy_instance(state.instance, null);
        state.instance = (vk::Instance)null;
    }
    if (state.adapters.ptr != null) {
        mem::free(state.adapters.ptr);
        state.adapters = {};
    }
    mem::free(state);
}

fn void vk_destroy_runtime(void* backend_state) {
    destroy_vk_runtime_state((VkRuntimeState*)backend_state);
}

const gpu::RuntimeBackendVTable VK_RUNTIME_VTABLE @private = {
    .destroy_runtime         = &vk_destroy_runtime,
    .get_adapter_info        = &vk_get_adapter_info,
    .get_adapter_diagnostics = &vk_get_adapter_diagnostics,
};

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    VkRuntimeState* state = mem::new(VkRuntimeState);
    defer catch {
        destroy_vk_runtime_state(state);
    }

    state.debug_callback = desc.debug_callback;
    state.debug_user_data = desc.debug_user_data;
    state.validation = desc.enable_validation;
    vk::DebugUtilsMessengerCreateInfoEXT messenger_info =
        runtime_debug_messenger_create_info(state);
    VkInstanceDesc instance_desc = {
        .enable_validation         = desc.enable_validation,
        .enable_debug_utils        = desc.enable_validation
            || desc.debug_callback != null,
        .enable_available_surfaces = true,
        .application_name          = desc.application_name,
        .debug_messenger_info      = desc.enable_validation
            ? &messenger_info
            : null,
    };
    state.instance = create_vk_instance(&instance_desc)!;
    if (instance_desc.enable_debug_utils) {
        if (!create_runtime_debug_messenger(state, &messenger_info)) {
            report_runtime_debug_messenger_failure(state);
        }
    }

    enumerate_vk_adapters(state)!;
    return gpu::register_runtime(
        backend:       gpu::BackendKind.VULKAN,
        vtable:        &VK_RUNTIME_VTABLE,
        backend_state: state,
        adapter_count: (uint)state.adapters.len,
    );
}
