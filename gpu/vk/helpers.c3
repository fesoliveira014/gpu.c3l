module gpu::vk @private;

import std::io;
import gpu @public;
import vk;

// Vulkan 1.3 in the (variant<<29)|(major<<22)|(minor<<12)|patch layout. The
// binding ships no API_VERSION_1_3 constant.
const uint API_VERSION_1_3 = (1u << 22) | (3u << 12);

enum WsiResultKind : int {
    SURFACE,
    SURFACE_ENUMERATION,
    SWAPCHAIN_CREATE,
    ENUMERATION,
    ACQUIRE,
    PRESENT,
    COMMON,
}

fn void? map_result(vk::Result result, bool report_unmapped) @private {
    switch (result) {
        case vk::Result.SUCCESS:
            return;
        case vk::Result.ERROR_OUT_OF_HOST_MEMORY:
            return gpu::OUT_OF_HOST_MEMORY~;
        case vk::Result.ERROR_OUT_OF_DEVICE_MEMORY:
            return gpu::OUT_OF_DEVICE_MEMORY~;
        case vk::Result.ERROR_DEVICE_LOST:
            return gpu::DEVICE_LOST~;
        case vk::Result.ERROR_SURFACE_LOST_KHR:
            return gpu::SURFACE_LOST~;
        case vk::Result.ERROR_FEATURE_NOT_PRESENT:
        case vk::Result.ERROR_EXTENSION_NOT_PRESENT:
        case vk::Result.ERROR_LAYER_NOT_PRESENT:
            return gpu::UNSUPPORTED_FEATURE~;
        default:
            if (report_unmapped) io::eprintfn("vk: unmapped VkResult %d", (int)result);
            return gpu::BACKEND_ERROR~;
    }
}

<* Map only context-free Vulkan results. Unknown results are backend errors,
   never implicit device loss. *>
fn void? check(vk::Result result) @private {
    return map_result(result, true);
}

fn void observe_device_result(VkDeviceState* state, vk::Result result) @private {
    if (state != null && result == vk::Result.ERROR_DEVICE_LOST) {
        gpu::mark_device_lost(&state.owner);
    }
}

fn void? check_device_result(VkDeviceState* state, vk::Result result) @private {
    observe_device_result(state, result);
    return check(result);
}

fn ZString backend_result_text(vk::Result result) @private {
    switch (result) {
        case vk::Result.ERROR_OUT_OF_HOST_MEMORY:   return "Vulkan host memory exhausted";
        case vk::Result.ERROR_OUT_OF_DEVICE_MEMORY: return "Vulkan device memory exhausted";
        case vk::Result.ERROR_DEVICE_LOST:          return "Vulkan device lost";
        case vk::Result.ERROR_SURFACE_LOST_KHR:     return "Vulkan surface lost";
        case vk::Result.ERROR_FEATURE_NOT_PRESENT:  return "Vulkan feature not present";
        case vk::Result.ERROR_EXTENSION_NOT_PRESENT: return "Vulkan extension not present";
        case vk::Result.ERROR_LAYER_NOT_PRESENT:    return "Vulkan layer not present";
        default:                                    return "unmapped Vulkan result";
    }
}

fn void? check_backend_result(VkDeviceState* state, ZString operation, vk::Result result) {
    observe_device_result(state, result);
    bool stderr_fallback = state == null || state.debug_callback == null;
    fault public_fault = @catch(map_result(result, stderr_fallback));
    if (!public_fault) return;
    if (!stderr_fallback) {
        gpu::DebugMessage message = {
            .severity             = gpu::DebugMessageSeverity.ERROR,
            .category             = { .backend },
            .operation            = operation,
            .has_fault            = true,
            .public_fault         = public_fault,
            .backend_text         = backend_result_text(result),
            .validation_id_number = (int)result,
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return public_fault~;
}

fn fault wait_result_fault(vk::Result result) {
    if (result == vk::Result.TIMEOUT) return gpu::WAIT_TIMEOUT;
    return @catch(map_result(result, false));
}

fn void? check_wait_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    fault public_fault = @catch(check_wait(result));
    if (!public_fault) return;
    if (public_fault != gpu::WAIT_TIMEOUT) {
        return check_backend_result(state, operation, result);
    }
    return public_fault~;
}

fn void? check_backend_create_result(vk::Result result) @private {
    switch (result) {
        case vk::Result.ERROR_INCOMPATIBLE_DRIVER:
        case vk::Result.ERROR_INITIALIZATION_FAILED:
            return gpu::UNSUPPORTED_BACKEND~;
        default:
            return check(result);
    }
}

fn void? check_physical_device_enumeration_result(vk::Result result) @private {
    if (result == vk::Result.ERROR_INITIALIZATION_FAILED) return gpu::UNSUPPORTED_BACKEND~;
    return check_enumeration_result(result);
}

fn void? map_wsi_result(WsiResultKind kind, vk::Result result, bool report_unmapped) @private {
    switch (kind) {
        case WsiResultKind.SURFACE:
            switch (result) {
                case vk::Result.ERROR_SURFACE_LOST_KHR:         return gpu::SURFACE_LOST~;
                case vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR: return gpu::INVALID_RESOURCE_STATE~;
                default:                                        return map_result(result, report_unmapped);
            }
        case WsiResultKind.SURFACE_ENUMERATION:
            if (result == vk::Result.ERROR_SURFACE_LOST_KHR) return gpu::SURFACE_LOST~;
            if (result == vk::Result.INCOMPLETE) return gpu::BACKEND_ERROR~;
            return map_result(result, report_unmapped);
        case WsiResultKind.SWAPCHAIN_CREATE:
            switch (result) {
                case vk::Result.ERROR_SURFACE_LOST_KHR:              return gpu::SURFACE_LOST~;
                case vk::Result.ERROR_OUT_OF_DATE_KHR:               return gpu::SWAPCHAIN_OUT_OF_DATE~;
                case vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR:      return gpu::INVALID_RESOURCE_STATE~;
                case vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: return gpu::UNSUPPORTED_FEATURE~;
                default:                                             return map_result(result, report_unmapped);
            }
        case WsiResultKind.ENUMERATION:
            if (result == vk::Result.INCOMPLETE) return gpu::BACKEND_ERROR~;
            return map_result(result, report_unmapped);
        case WsiResultKind.ACQUIRE:
            switch (result) {
                case vk::Result.SUCCESS:
                case vk::Result.SUBOPTIMAL_KHR:         return;
                case vk::Result.ERROR_SURFACE_LOST_KHR: return gpu::SURFACE_LOST~;
                case vk::Result.TIMEOUT:
                case vk::Result.NOT_READY:              return gpu::WAIT_TIMEOUT~;
                case vk::Result.ERROR_OUT_OF_DATE_KHR:  return gpu::SWAPCHAIN_OUT_OF_DATE~;
                default:                                return map_result(result, report_unmapped);
            }
        case WsiResultKind.PRESENT:
            switch (result) {
                case vk::Result.SUCCESS:
                case vk::Result.SUBOPTIMAL_KHR:         return;
                case vk::Result.ERROR_SURFACE_LOST_KHR: return gpu::SURFACE_LOST~;
                case vk::Result.ERROR_OUT_OF_DATE_KHR:  return gpu::SWAPCHAIN_OUT_OF_DATE~;
                default:                                return map_result(result, report_unmapped);
            }
        default:
            return map_result(result, report_unmapped);
    }
}

fn void? check_surface_result(vk::Result result) @private {
    return map_wsi_result(WsiResultKind.SURFACE, result, true);
}

fn void? check_surface_enumeration_result(vk::Result result) @private {
    return map_wsi_result(WsiResultKind.SURFACE_ENUMERATION, result, true);
}

fn void? check_swapchain_create_result(vk::Result result) @private {
    return map_wsi_result(WsiResultKind.SWAPCHAIN_CREATE, result, true);
}

fn void? check_acquire_result(vk::Result result) @private {
    return map_wsi_result(WsiResultKind.ACQUIRE, result, true);
}

fn void? check_present_result(vk::Result result) @private {
    return map_wsi_result(WsiResultKind.PRESENT, result, true);
}

fn void? check_texture_create_result(vk::Result result) @private {
    switch (result) {
        case vk::Result.ERROR_FORMAT_NOT_SUPPORTED:
        case vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:
            return gpu::UNSUPPORTED_FEATURE~;
        default:
            return check(result);
    }
}

fn void? check_texture_create_device_result(VkDeviceState* state, vk::Result result) @private {
    observe_device_result(state, result);
    return check_texture_create_result(result);
}

fn void? check_shader_create_result(vk::Result result) @private {
    switch (result) {
        case vk::Result.ERROR_INVALID_SHADER_NV:
        case vk::Result.ERROR_VALIDATION_FAILED_EXT:
            return gpu::SHADER_INVALID~;
        default:
            return check(result);
    }
}

fn void? check_shader_create_device_result(VkDeviceState* state, vk::Result result) @private {
    observe_device_result(state, result);
    return check_shader_create_result(result);
}

fn void? check_pipeline_create_result(vk::Result result) @private {
    switch (result) {
        case vk::Result.SUCCESS:
            return;
        case vk::Result.ERROR_OUT_OF_HOST_MEMORY:
        case vk::Result.ERROR_OUT_OF_DEVICE_MEMORY:
        case vk::Result.ERROR_DEVICE_LOST:
            return check(result);
        default:
            return gpu::PIPELINE_CREATE_FAILED~;
    }
}

fn void? check_pipeline_create_device_result(VkDeviceState* state, vk::Result result) @private {
    observe_device_result(state, result);
    return check_pipeline_create_result(result);
}

fn void? check_descriptor_pool_create_result(vk::Result result) @private {
    if (result == vk::Result.ERROR_FRAGMENTATION) return gpu::DESCRIPTOR_HEAP_FULL~;
    return check(result);
}

fn void? check_descriptor_allocate_result(vk::Result result) @private {
    switch (result) {
        case vk::Result.ERROR_OUT_OF_POOL_MEMORY:
        case vk::Result.ERROR_FRAGMENTED_POOL:
            return gpu::DESCRIPTOR_HEAP_FULL~;
        default:
            return check(result);
    }
}

fn void? check_pipeline_cache_data_result(vk::Result result) @private {
    if (result == vk::Result.INCOMPLETE) return gpu::INVALID_ARGUMENT~;
    return check(result);
}

fn void? check_enumeration_result(vk::Result result) @private {
    if (result == vk::Result.INCOMPLETE) return gpu::BACKEND_ERROR~;
    return check(result);
}

fn bool pipeline_cache_blob_retryable(vk::Result result) @private {
    switch (result) {
        case vk::Result.SUCCESS:
        case vk::Result.ERROR_OUT_OF_HOST_MEMORY:
        case vk::Result.ERROR_OUT_OF_DEVICE_MEMORY:
        case vk::Result.ERROR_DEVICE_LOST:
            return false;
        default:
            return true;
    }
}

<* Translate bounded host-wait results through the common result taxonomy. *>
fn void? check_wait(vk::Result result) @private {
    fault public_fault = wait_result_fault(result);
    if (!public_fault) return;
    return public_fault~;
}

<* Pure translation of a public execution stage to a synchronization2 stage
   mask. Unit-testable without a device. *>
fn vk::PipelineStageFlags2 stage_to_vk(gpu::Stage stage) {
    switch (stage) {
        case gpu::Stage.HOST:             return vk::PipelineStageFlagBits2.STAGE_2_HOST_BIT;
        case gpu::Stage.TRANSFER:         return vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT;
        case gpu::Stage.COMPUTE_SHADER:   return vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT;
        case gpu::Stage.VERTEX_SHADER:    return vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT;
        case gpu::Stage.FRAGMENT_SHADER:  return vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT;
        case gpu::Stage.COLOR_ATTACHMENT: return vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case gpu::Stage.DEPTH_STENCIL:
            return (vk::PipelineStageFlags2)((ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT);
        case gpu::Stage.INDIRECT_COMMAND: return vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
        case gpu::Stage.PRESENT:          return vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
        case gpu::Stage.NONE:             return vk::PipelineStageFlagBits2.STAGE_2_NONE;
        default:                          return vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
    }
}

<* Pure translation of a public hazard to a synchronization2 access mask. *>
fn vk::AccessFlags2 hazard_to_vk_access(gpu::Hazard hazard) {
    switch (hazard) {
        case gpu::Hazard.HOST_WRITE:     return vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT;
        case gpu::Hazard.HOST_READ:      return vk::AccessFlagBits2.ACCESS_2_HOST_READ_BIT;
        case gpu::Hazard.TRANSFER_READ:  return vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT;
        case gpu::Hazard.TRANSFER_WRITE: return vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT;
        case gpu::Hazard.SHADER_READ:    return vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT;
        case gpu::Hazard.SHADER_WRITE:   return vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT;
        case gpu::Hazard.COLOR_READ:     return vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        case gpu::Hazard.COLOR_WRITE:    return vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        case gpu::Hazard.DEPTH_READ:     return vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        case gpu::Hazard.DEPTH_WRITE:    return vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case gpu::Hazard.INDIRECT_READ:  return vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        case gpu::Hazard.PRESENT_READ:   return vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT;
        case gpu::Hazard.NONE:           return vk::AccessFlagBits2.ACCESS_2_NONE;
        case gpu::Hazard.SHADER_READ_WRITE:
            return (vk::AccessFlags2)((ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
                | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT);
        case gpu::Hazard.COLOR_READ_WRITE:
            return (vk::AccessFlags2)((ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
                | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        case gpu::Hazard.DEPTH_READ_WRITE:
            return (vk::AccessFlags2)((ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        default:                         return vk::AccessFlagBits2.ACCESS_2_NONE;
    }
}

<* Pure translation of a public texture layout to a Vulkan image layout. *>
fn vk::ImageLayout layout_to_vk(gpu::TextureLayout layout) {
    switch (layout) {
        case gpu::TextureLayout.UNDEFINED:        return vk::ImageLayout.UNDEFINED;
        case gpu::TextureLayout.GENERAL:          return vk::ImageLayout.GENERAL;
        case gpu::TextureLayout.COLOR_ATTACHMENT: return vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.DEPTH_STENCIL:    return vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.SHADER_READ:      return vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL;
        case gpu::TextureLayout.TRANSFER_SRC:     return vk::ImageLayout.TRANSFER_SRC_OPTIMAL;
        case gpu::TextureLayout.TRANSFER_DST:     return vk::ImageLayout.TRANSFER_DST_OPTIMAL;
        case gpu::TextureLayout.PRESENT:          return vk::ImageLayout.PRESENT_SRC_KHR;
        default:                                  return vk::ImageLayout.UNDEFINED;
    }
}

fn gpu::PresentMode present_mode_from_vk(vk::PresentModeKHR mode) {
    switch (mode) {
        case vk::PresentModeKHR.IMMEDIATE_KHR: return gpu::PresentMode.IMMEDIATE;
        case vk::PresentModeKHR.MAILBOX_KHR:   return gpu::PresentMode.MAILBOX;
        default:                               return gpu::PresentMode.FIFO;
    }
}
