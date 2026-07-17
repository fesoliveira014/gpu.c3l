module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::io;

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

fn gpu::DebugMessageSeverity debug_message_severity(vk::DebugUtilsMessageSeverityFlagsEXT severity) {
    uint flags = (uint)severity;
    if ((flags & (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT) != 0) {
        return gpu::DebugMessageSeverity.ERROR;
    }
    if ((flags & (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT) != 0) {
        return gpu::DebugMessageSeverity.WARNING;
    }
    if ((flags & (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.INFO_BIT_EXT) != 0) {
        return gpu::DebugMessageSeverity.INFO;
    }
    return gpu::DebugMessageSeverity.VERBOSE;
}

fn gpu::DebugMessageCategory debug_message_category(vk::DebugUtilsMessageTypeFlagsEXT message_type) {
    uint flags = (uint)message_type;
    gpu::DebugMessageCategory category = {};
    category.general = (flags & (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.GENERAL_BIT_EXT) != 0;
    category.validation = (flags & (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT) != 0;
    category.performance = (flags & (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.PERFORMANCE_BIT_EXT) != 0;
    category.backend = true;
    return category;
}

fn gpu::DebugResourceKind debug_resource_kind(vk::ObjectType object_type) {
    switch (object_type) {
        case vk::ObjectType.BUFFER:         return gpu::DebugResourceKind.GPU_SPAN;
        case vk::ObjectType.DEVICE:         return gpu::DebugResourceKind.DEVICE;
        case vk::ObjectType.IMAGE:
        case vk::ObjectType.IMAGE_VIEW:     return gpu::DebugResourceKind.TEXTURE;
        case vk::ObjectType.PIPELINE:       return gpu::DebugResourceKind.PIPELINE;
        case vk::ObjectType.SEMAPHORE:      return gpu::DebugResourceKind.NONE;
        case vk::ObjectType.SWAPCHAIN_KHR:  return gpu::DebugResourceKind.SWAPCHAIN;
        case vk::ObjectType.SHADER_MODULE:  return gpu::DebugResourceKind.SHADER;
        case vk::ObjectType.COMMAND_BUFFER: return gpu::DebugResourceKind.COMMAND_LIST;
        case vk::ObjectType.SAMPLER:        return gpu::DebugResourceKind.SAMPLER;
        default:                            return gpu::DebugResourceKind.NONE;
    }
}

fn gpu::DebugResourceRef debug_resource(vk::DebugUtilsMessengerCallbackDataEXT* data) {
    gpu::DebugResourceRef fallback = {};
    if (data == null || data.p_objects == null) return fallback;
    for (uint i = 0; i < data.object_count; i++) {
        vk::DebugUtilsObjectNameInfoEXT* object = &data.p_objects[i];
        gpu::DebugResourceKind kind = debug_resource_kind(object.object_type);
        if (object.p_object_name != null) {
            return {
                .kind       = kind,
                .debug_name = (ZString)object.p_object_name,
            };
        }
        if (fallback.kind == gpu::DebugResourceKind.NONE && kind != gpu::DebugResourceKind.NONE) {
            fallback.kind = kind;
        }
    }
    return fallback;
}

fn uint route_debug_message(
    VkDeviceState* state,
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

fn uint debug_callback(
    vk::DebugUtilsMessageSeverityFlagsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT     message_type,
    vk::DebugUtilsMessengerCallbackDataEXT* data,
    void*                                 user_data
) @private {
    return route_debug_message((VkDeviceState*)user_data, severity, message_type, data);
}

fn vk::DebugUtilsMessengerCreateInfoEXT debug_messenger_create_info(VkDeviceState* state) {
    uint severity_flags = (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT
                        | (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT;
    uint type_flags = (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.GENERAL_BIT_EXT
                    | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT
                    | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.PERFORMANCE_BIT_EXT;
    return vk::debug_utils_messenger_create_info_ext()
        .set_message_severity((vk::DebugUtilsMessageSeverityFlagsEXT)severity_flags)
        .set_message_type((vk::DebugUtilsMessageTypeFlagsEXT)type_flags)
        .set_pfn_user_callback((vk::PFN_vkDebugUtilsMessengerCallbackEXT)&debug_callback)
        .set_user_data(state);
}

fn void report_debug_messenger_failure(VkDeviceState* state, ZString text) {
    if (state.debug_callback == null) {
        io::eprintfn("vk: %s; relying on layer stderr", text);
        return;
    }
    gpu::DebugMessage message = {
        .severity     = gpu::DebugMessageSeverity.WARNING,
        .category     = { .backend },
        .operation    = "create_device",
        .backend_text = text,
    };
    gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
}

<* Emit exactly one public-contract ERROR when configured, return public_fault
   unchanged, and permit delivery while internal locks are held. *>
fn fault report_contract_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    gpu::DebugResourceKind resource_kind,
    bool has_public_identity,
    uint index,
    uint generation,
    ZString debug_name,
    ZString rejected_field,
    ZString invariant,
    ZString backend_text,
) {
    gpu::DebugMessage message = {
        .severity       = gpu::DebugMessageSeverity.ERROR,
        .category       = { .public_contract },
        .operation      = operation,
        .has_fault      = true,
        .public_fault   = public_fault,
        .resource       = {
            .kind                = resource_kind,
            .has_public_identity = has_public_identity,
            .index               = index,
            .generation          = generation,
            .debug_name          = debug_name,
        },
        .rejected_field = rejected_field,
        .invariant      = invariant,
        .backend_text   = backend_text,
    };
    gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    return public_fault;
}

fn bool report_resource_lifetime(
    VkDeviceState* state,
    gpu::DebugResourceKind resource_kind,
    bool has_public_identity,
    uint index,
    uint generation,
    ZString debug_name,
    ZString backend_text,
) @private {
    if (state.debug_callback == null) return false;
    gpu::DebugMessage message = {
        .severity  = gpu::DebugMessageSeverity.WARNING,
        .category  = { .resource_lifetime },
        .operation = "destroy_device",
        .resource  = {
            .kind                = resource_kind,
            .has_public_identity = has_public_identity,
            .index               = index,
            .generation          = generation,
            .debug_name          = debug_name,
        },
        .invariant    = "resource remains live at device destruction",
        .backend_text = backend_text,
    };
    gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    return true;
}

fn bool create_debug_messenger(
    VkDeviceState* state,
    vk::DebugUtilsMessengerCreateInfoEXT* create_info,
    vk::DebugUtilsMessengerEXT* out_messenger,
) {
    if (state == null || state.instance == null
        || create_info == null || out_messenger == null) return false;
    if (!instance_dispatch_supports_debug_messenger(&state.instance_dispatch)) return false;
    vk::Result result = state.instance_dispatch.debug.create_debug_messenger(
        state.instance,
        create_info,
        null,
        out_messenger,
    );
    if (result == vk::Result.SUCCESS) return true;
    *out_messenger = (vk::DebugUtilsMessengerEXT)null;
    return false;
}

<* Install the device debug messenger when debug-utils is available. *>
fn void setup_debug_messenger(VkDeviceState* state) @private {
    vk::DebugUtilsMessengerCreateInfoEXT create_info =
        debug_messenger_create_info(state);
    if (!create_debug_messenger(state, &create_info, &state.messenger)) {
        report_debug_messenger_failure(state, "debug-utils messenger unavailable");
    }
}

<* Best-effort object naming: silently a no-op when the debug-utils entry
   point is unavailable or no name was given. *>
fn void set_object_name(VkDeviceState* state, vk::ObjectType object_type, ulong object_handle, ZString name) {
    if (!state.debug_names) return;
    if (name == null) return;
    if (state.device_dispatch.debug.set_object_name == null) return;
    vk::DebugUtilsObjectNameInfoEXT info = vk::debug_utils_object_name_info_ext();
    info.object_type = object_type;
    info.object_handle = object_handle;
    info.p_object_name = (char*)name;
    if (state.device_dispatch.debug.set_object_name(state.device, &info) != vk::Result.SUCCESS) return;
}

fn void destroy_debug_messenger(VkDeviceState* state) @private {
    if (state.messenger == null) return;
    if (state.instance_dispatch.debug.destroy_debug_messenger != null) {
        state.instance_dispatch.debug.destroy_debug_messenger(
            state.instance,
            state.messenger,
            null,
        );
    }
    state.messenger = (vk::DebugUtilsMessengerEXT)null;
}

fn ZString slot_name(char* name) @private {
    return name[0] != 0 ? (ZString)name : (ZString)"(unnamed)";
}

<* Library-internal allocations (arenas, dedicated transfers) are exempt. *>
fn void mark_buffer_internal(VkDeviceState* state, BufferHandle handle) {
    if (try slot = state.buffers.get(handle)) slot.internal = true;
}

fn uint report_buffer_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.buffers.slots.ptr == null) return 0;
    foreach (i, &cell : state.buffers.slots[:state.buffers.next_new]) {
        if (!cell.used || cell.value.internal) continue;
        ZString name = slot_name(&cell.value.debug_name[0]);
        if (!report_resource_lifetime(
            state:               state,
            resource_kind:       gpu::DebugResourceKind.GPU_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          name,
            backend_text:        "GPU span storage remains live",
        )) {
            io::eprintfn("gpu: leaked GPU span storage '%s'", name);
        }
        count++;
    }
    return count;
}

fn uint report_allocation_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.allocations.slots.ptr == null) return 0;
    foreach (i, &cell : state.allocations.slots[:state.allocations.next_new]) {
        if (!cell.used) continue;
        ZString name = slot_name(&cell.value.backing.debug_name[0]);
        if (!report_resource_lifetime(
            state:               state,
            resource_kind:       gpu::DebugResourceKind.ALLOCATION,
            has_public_identity: true,
            index:               (uint)i,
            generation:          cell.generation,
            debug_name:          name,
            backend_text:        "allocation remains live",
        )) {
            io::eprintfn(
                "gpu: leaked allocation '%s' (index %d, generation %d)",
                name,
                i,
                cell.generation,
            );
        }
        count++;
    }
    return count;
}

fn uint report_texture_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.textures.slots.ptr == null) return 0;
    foreach (i, &cell : state.textures.slots[:state.textures.next_new]) {
        if (!cell.used || cell.value.swapchain_owned) continue;
        ZString name = slot_name(&cell.value.debug_name[0]);
        if (!report_resource_lifetime(
            state:               state,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               (uint)i,
            generation:          cell.generation,
            debug_name:          name,
            backend_text:        "texture remains live",
        )) {
            io::eprintfn("gpu: leaked texture '%s' (%d bytes)", name, cell.value.allocation_size);
        }
        count++;
    }
    return count;
}

fn uint report_pipeline_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.pipeline_cache.entries.ptr == null) return 0;
    foreach (i, &entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
        if (!entry.used) continue;
        bool has_public_identity = false;
        uint alias_index = 0;
        uint alias_generation = 0;
        if (state.pipelines.slots.ptr != null) {
            foreach (j, &cell : state.pipelines.slots[:state.pipelines.next_new]) {
                if (!cell.used || cell.value.entry != (uint)i) continue;
                has_public_identity = true;
                alias_index = (uint)j;
                alias_generation = cell.generation;
                break;
            }
        }
        if (!report_resource_lifetime(
            state:               state,
            resource_kind:       gpu::DebugResourceKind.PIPELINE,
            has_public_identity: has_public_identity,
            index:               alias_index,
            generation:          alias_generation,
            debug_name:          null,
            backend_text:        "pipeline cache entry retains live aliases",
        )) {
            io::eprintfn("gpu: leaked pipeline cache entry %d (%d live aliases)", i, entry.refcount);
        }
        count++;
    }
    return count;
}

fn uint report_shader_leaks(VkDeviceState* state) {
    uint count = 0;
    if (state.shaders.slots.ptr == null) return 0;
    foreach (i, &cell : state.shaders.slots[:state.shaders.next_new]) {
        if (!cell.used) continue;
        if (!report_resource_lifetime(
            state:               state,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: true,
            index:               (uint)i,
            generation:          cell.generation,
            debug_name:          null,
            backend_text:        "shader remains live",
        )) {
            io::eprintfn("gpu: leaked shader (index %d)", i);
        }
        count++;
    }
    return count;
}

fn uint report_resource_leaks(VkDeviceState* state) {
    uint count = report_allocation_leaks(state);
    count += report_buffer_leaks(state);
    count += report_texture_leaks(state);
    count += report_pipeline_leaks(state);
    count += report_shader_leaks(state);

    if (state.persistent_arena.virtual_block != null) {
        vma::Statistics stats;
        state.persistent_arena.virtual_block.get_statistics(&stats);
        if (stats.allocation_count > 0) {
            if (!report_resource_lifetime(
                state:               state,
                resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
                has_public_identity: false,
                index:               0,
                generation:          0,
                debug_name:          "persistent_arena",
                backend_text:        "persistent spans remain allocated",
            )) {
                io::eprintfn("gpu: %d persistent spans still allocated", stats.allocation_count);
            }
            count += (uint)stats.allocation_count;
        }
    }
    return count;
}

fn void? vk_cmd_begin_label(gpu::CommandList* commands, ZString label, float[4] color) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING
        && record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    if (!device_dispatch_supports_labels(&state.device_dispatch)) return;
    vk::DebugUtilsLabelEXT info = vk::debug_utils_label_ext();
    info.p_label_name = (char*)label;
    info.color = color;
    state.device_dispatch.debug.cmd_begin_label(record.command_buffer, &info);
}

fn void? vk_cmd_end_label(gpu::CommandList* commands) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING
        && record.state != CommandState.RECORDING_RENDER_PASS) return gpu::COMMAND_RECORDING_ERROR~;
    if (!device_dispatch_supports_labels(&state.device_dispatch)) return;
    state.device_dispatch.debug.cmd_end_label(record.command_buffer);
}
