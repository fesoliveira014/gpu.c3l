module gpu::vk;

import gpu;
import vk;
import vma;
import std::io;

<* Messenger callback: prints validation messages to stderr and returns
   VK_FALSE, as the spec requires from application callbacks. *>
const usz DEBUG_NAME_CAP = 64;

<* Truncating copy into a slot-owned buffer: no lifetime requirement on the
   caller's string. *>
fn void copy_debug_name(char* dst, ZString name) {
    if (name == null) {
        dst[0] = 0;
        return;
    }
    usz i = 0;
    while (i < DEBUG_NAME_CAP - 1 && name[i] != 0) {
        dst[i] = name[i];
        i++;
    }
    dst[i] = 0;
}

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

fn ZString slot_name(char* name) @private {
    return name[0] != 0 ? (ZString)name : (ZString)"(unnamed)";
}

<* Library-internal allocations (arenas, dedicated transfers) are exempt. *>
fn void mark_buffer_internal(VkDeviceState* state, gpu::BufferHandle handle) {
    if (try slot = state.buffers.get(handle)) slot.internal = true;
}

fn uint report_buffer_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.buffers.slots.ptr == null) return 0;
    foreach (i, &cell : state.buffers.slots[:state.buffers.next_new]) {
        if (!cell.used || cell.value.internal) continue;
        io::eprintfn("gpu: leaked buffer '%s' (index %d, generation %d)",
            slot_name(&cell.value.debug_name[0]), i, cell.generation);
        count++;
    }
    return count;
}

fn uint report_texture_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.textures.slots.ptr == null) return 0;
    foreach (i, &cell : state.textures.slots[:state.textures.next_new]) {
        if (!cell.used || cell.value.swapchain_owned) continue;
        io::eprintfn("gpu: leaked texture '%s' (%d bytes)",
            slot_name(&cell.value.debug_name[0]), cell.value.allocation_size);
        count++;
    }
    return count;
}

fn uint report_pipeline_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.pipeline_cache.entries.ptr == null) return 0;
    foreach (i, &entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
        if (!entry.used) continue;
        io::eprintfn("gpu: leaked pipeline cache entry %d (%d live aliases)", i, entry.refcount);
        count++;
    }
    return count;
}

fn uint report_shader_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.shaders.slots.ptr == null) return 0;
    foreach (i, &cell : state.shaders.slots[:state.shaders.next_new]) {
        if (!cell.used) continue;
        io::eprintfn("gpu: leaked shader (index %d)", i);
        count++;
    }
    return count;
}

fn uint report_semaphore_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.semaphores.slots.ptr == null) return 0;
    foreach (i, &cell : state.semaphores.slots[:state.semaphores.next_new]) {
        if (!cell.used) continue;
        io::eprintfn("gpu: leaked semaphore (index %d)", i);
        count++;
    }
    return count;
}

<* The device-owned default and helper contexts are exempt. *>
fn uint report_context_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.recording_contexts.slots.ptr == null) return 0;
    foreach (i, &cell : state.recording_contexts.slots[:state.recording_contexts.next_new]) {
        if (!cell.used || (uint)i == state.default_context.index || (uint)i == state.helper_context.index) continue;
        io::eprintfn("gpu: leaked recording context (index %d)", i);
        count++;
    }
    return count;
}

fn uint report_resource_leaks(VkDeviceState* state) {
    uint count = report_buffer_leaks(state);
    count += report_texture_leaks(state);
    count += report_pipeline_leaks(state);
    count += report_shader_leaks(state);
    count += report_semaphore_leaks(state);
    count += report_context_leaks(state);
    if (state.persistent_arena.virtual_block != null) {
        vma::Statistics stats;
        state.persistent_arena.virtual_block.get_statistics(&stats);
        if (stats.allocation_count > 0) {
            io::eprintfn("gpu: %d persistent spans still allocated", stats.allocation_count);
            count += (uint)stats.allocation_count;
        }
    }
    return count;
}

fn bool label_recording(gpu::CommandList* commands) @private {
    return commands.state == gpu::CommandState.RECORDING
        || commands.state == gpu::CommandState.RECORDING_RENDER_PASS;
}

fn void? vk_cmd_begin_label(gpu::CommandList* commands, ZString label, float[4] color) @private {
    if (!label_recording(commands)) return gpu::COMMAND_RECORDING_ERROR~;
    if (vk::extensions.vk_cmd_begin_debug_utils_label_ext == null) return;
    vk::DebugUtilsLabelEXT info = vk::debug_utils_label_ext();
    info.p_label_name = (char*)label;
    info.color = color;
    vk::extensions.vk_cmd_begin_debug_utils_label_ext((vk::CommandBuffer)commands.backend, &info);
}

fn void? vk_cmd_end_label(gpu::CommandList* commands) @private {
    if (!label_recording(commands)) return gpu::COMMAND_RECORDING_ERROR~;
    if (vk::extensions.vk_cmd_end_debug_utils_label_ext == null) return;
    vk::extensions.vk_cmd_end_debug_utils_label_ext((vk::CommandBuffer)commands.backend);
}
