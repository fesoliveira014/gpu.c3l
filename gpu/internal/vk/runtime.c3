module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::io;

struct VkRuntimeState {
    vk::Instance               instance;
    VkInstanceDispatch         dispatch;
    VkInstanceCapabilities     instance_capabilities;
    vk::DebugUtilsMessengerEXT messenger;
    gpu::DebugMessageCallback  debug_callback;
    void*                      debug_user_data;
    VkAdapterState[]           adapters;
    DeviceValidationPolicy     validation_policy;
    bool                       debug_names;
    uint                       texture_heap_capacity;
    uint                       sampler_heap_capacity;
    uint                       texture_capacity;
    uint                       pipeline_capacity;
    char[]                     pipeline_cache_data;
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
    gpu::internal::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
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
    if (!instance_dispatch_supports_debug_messenger(&state.dispatch)) return false;
    vk::Result result = state.dispatch.debug.create_debug_messenger(
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
    if (state.dispatch.debug.destroy_debug_messenger != null) {
        state.dispatch.debug.destroy_debug_messenger(state.instance, state.messenger, null);
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
    gpu::internal::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
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
    if (state.pipeline_cache_data.ptr != null) {
        mem::free(state.pipeline_cache_data.ptr);
        state.pipeline_cache_data = {};
    }
    mem::free(state);
}

fn void vk_destroy_runtime(void* backend_state) {
    destroy_vk_runtime_state((VkRuntimeState*)backend_state);
}

const gpu::internal::RuntimeBackendVTable VK_RUNTIME_VTABLE @private = {
    .destroy_runtime         = &vk_destroy_runtime,
    .get_adapter_info        = &vk_get_adapter_info,
    .get_adapter_diagnostics = &vk_get_adapter_diagnostics,
    .create_win32_surface    = &vk_create_win32_surface,
    .create_wayland_surface  = &vk_create_wayland_surface,
    .create_x11_surface      = &vk_create_x11_surface,
    .destroy_surface         = &vk_destroy_surface,
    .supports_presentation   = &vk_supports_presentation,
    .supports_queue_requirements = &vk_supports_queue_requirements,
    .supports_resource_agnostic_texture_sync =
        &vk_supports_resource_agnostic_texture_sync,
    .supports_dynamic_color_state = &vk_supports_dynamic_color_state,
};

fn VkInstanceDesc runtime_instance_desc(
    gpu::RuntimeDesc* desc,
    vk::DebugUtilsMessengerCreateInfoEXT* messenger_info,
) {
    return {
        .enable_validation         = desc.enable_vulkan_validation,
        .enable_debug_utils        = desc.enable_vulkan_validation
            || desc.enable_debug_names
            || desc.debug_callback != null,
        .enable_available_surfaces = true,
        .application_name          = desc.application_name,
        .debug_messenger_info      = desc.enable_vulkan_validation
            ? messenger_info
            : null,
    };
}

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    VkRuntimeState* state = mem::new(VkRuntimeState);
    defer catch {
        destroy_vk_runtime_state(state);
    }

    state.debug_callback = desc.debug_callback;
    state.debug_user_data = desc.debug_user_data;
    state.validation_policy = {
        .contract        = desc.contract_validation,
        .track_lifetimes = desc.track_resource_lifetimes,
        .vulkan_layers   = desc.enable_vulkan_validation,
    };
    state.debug_names = desc.enable_debug_names;
    state.texture_heap_capacity = desc.texture_heap_capacity;
    state.sampler_heap_capacity = desc.sampler_heap_capacity;
    state.texture_capacity = desc.texture_capacity;
    state.pipeline_capacity = desc.pipeline_capacity;
    if (desc.pipeline_cache_data.len > 0) {
        if (desc.pipeline_cache_data.ptr == null) return gpu::INVALID_ARGUMENT~;
        state.pipeline_cache_data = mem::new_array(
            char,
            desc.pipeline_cache_data.len,
        );
        mem::copy(
            state.pipeline_cache_data.ptr,
            desc.pipeline_cache_data.ptr,
            desc.pipeline_cache_data.len,
        );
    }
    vk::DebugUtilsMessengerCreateInfoEXT messenger_info =
        runtime_debug_messenger_create_info(state);
    VkInstanceDesc instance_desc = runtime_instance_desc(desc, &messenger_info);
    VkInstanceCreateResult instance_result =
        create_vk_instance(&instance_desc)!;
    state.instance = instance_result.instance;
    state.instance_capabilities = instance_result.capabilities;
    VkInstanceDispatchPlan dispatch_plan = {
        .debug           = state.instance_capabilities.debug_utils,
        .surface         = state.instance_capabilities.surface,
        .win32_surface   = state.instance_capabilities.win32_surface,
        .wayland_surface = state.instance_capabilities.wayland_surface,
        .xlib_surface    = state.instance_capabilities.xlib_surface,
    };
    state.dispatch = load_vk_instance_dispatch(
        state.instance,
        &vk::get_instance_proc_addr,
        dispatch_plan,
    );
    if (should_create_debug_messenger(
        state.instance_capabilities,
        instance_desc.enable_debug_utils,
    )) {
        if (!create_runtime_debug_messenger(state, &messenger_info)) {
            report_runtime_debug_messenger_failure(state);
        }
    } else if (should_report_debug_messenger_unavailable(
        state.instance_capabilities,
        instance_desc.enable_debug_utils,
    )) {
        report_runtime_debug_messenger_failure(state);
    }

    enumerate_vk_adapters(state)!;
    return gpu::internal::register_runtime(
        backend:       gpu::BackendKind.VULKAN,
        vtable:        &VK_RUNTIME_VTABLE,
        backend_state: state,
        adapter_count: (uint)state.adapters.len,
    );
}
