module gpu::internal::vk @private;

import gpu::internal @public;
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
        gpu::internal::mark_device_lost(&state.owner);
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
        gpu::internal::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
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

fn vk::PipelineStageFlags2 barrier_before_stages_to_vk(
    gpu::StageMask stages,
    bool generated_work = false,
) {
    if (stages.all || stages.present) {
        return vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
    }

    ulong mask;
    if (stages.host) mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_HOST_BIT;
    if (stages.transfer) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT;
    }
    if (stages.compute) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT;
    }
    if (stages.vertex_shader) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT;
    }
    if (stages.fragment_shader) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if (stages.color_output) {
        mask |=
            (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    if (stages.depth_output) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
            | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    }
    if (stages.indirect) {
        mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
        if (generated_work) {
            mask |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
        }
    }
    if (stages.acceleration_structure_build) {
        mask |= (ulong)vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    return (vk::PipelineStageFlags2)mask;
}

fn vk::PipelineStageFlags2 barrier_after_stages_to_vk(
    gpu::StageMask stages,
    bool generated_work = false,
) {
    return barrier_before_stages_to_vk(stages, generated_work);
}

fn vk::AccessFlags2 barrier_before_access_to_vk(
    gpu::StageMask stages,
    bool ray_queries_enabled = false,
) {
    ulong mask;
    if (stages.all) mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_WRITE_BIT;
    if (stages.host) mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT;
    if (stages.transfer) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT;
    }
    if (stages.compute || stages.vertex_shader || stages.fragment_shader) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT;
        if (ray_queries_enabled) {
            mask |= (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        }
    }
    if (stages.color_output) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (stages.depth_output) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    if (stages.present) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT;
    }
    if (stages.acceleration_structure_build) {
        mask |= (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
            | (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    }
    return (vk::AccessFlags2)mask;
}

fn vk::AccessFlags2 barrier_after_access_to_vk(
    gpu::StageMask stages,
    bool generated_work = false,
    bool ray_queries_enabled = false,
) {
    ulong mask;
    if (stages.all) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT
            | (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_WRITE_BIT;
    }
    if (stages.host) mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_READ_BIT;
    if (stages.transfer) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT
            | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT;
    }
    if (stages.compute || stages.vertex_shader || stages.fragment_shader) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
            | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT;
        if (ray_queries_enabled) {
            mask |= (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        }
    }
    if (stages.color_output) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
            | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (stages.depth_output) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    if (stages.present) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT;
    }
    if (stages.indirect) {
        mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        if (generated_work) {
            mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV;
        }
    }
    if (stages.acceleration_structure_build) {
        mask |= (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
            | (ulong)vk::AccessFlagBits2
                .ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    }
    return (vk::AccessFlags2)mask;
}

fn vk::MemoryBarrier2 barrier_to_vk(
    gpu::Barrier* barrier,
    bool generated_work = false,
    bool ray_queries_enabled = false,
) {
    return vk::memory_barrier2()
        .set_src_stage_mask(barrier_before_stages_to_vk(
            barrier.before,
            generated_work,
        ))
        .set_src_access_mask(barrier_before_access_to_vk(
            barrier.before,
            ray_queries_enabled,
        ))
        .set_dst_stage_mask(barrier_after_stages_to_vk(
            barrier.after,
            generated_work,
        ))
        .set_dst_access_mask(barrier_after_access_to_vk(
            barrier.after,
            generated_work,
            ray_queries_enabled,
        ));
}

fn bool barrier_stage_mask_valid(gpu::StageMask stages) @private {
    return gpu::internal::stage_mask_valid(stages);
}

fn ZString barrier_masks_rejected_field(gpu::Barrier* barrier) {
    if (barrier == null) return "Barrier";
    if (!barrier_stage_mask_valid(barrier.before)) return "Barrier.before";
    if (!barrier_stage_mask_valid(barrier.after)) return "Barrier.after";
    return null;
}

fn bool barrier_masks_valid(gpu::Barrier* barrier) {
    return barrier_masks_rejected_field(barrier) == null;
}

fn bool queue_flags_include(
    vk::QueueFlags flags,
    vk::QueueFlagBits required,
) @private {
    return ((uint)flags & (uint)required) != 0;
}

fn bool barrier_stage_mask_supported(
    gpu::StageMask stages,
    vk::QueueFlags flags,
    bool ray_queries = false,
) @private {
    bool graphics = queue_flags_include(flags, vk::QueueFlagBits.GRAPHICS_BIT);
    bool compute = queue_flags_include(flags, vk::QueueFlagBits.COMPUTE_BIT);
    bool transfer = queue_flags_include(flags, vk::QueueFlagBits.TRANSFER_BIT);
    if (!graphics && !compute && !transfer) return false;
    if (stages.compute && !compute) return false;
    if ((stages.vertex_shader
            || stages.fragment_shader
            || stages.color_output
            || stages.depth_output)
        && !graphics) {
        return false;
    }
    if ((stages.indirect || stages.acceleration_structure_build)
        && !graphics && !compute) return false;
    if (stages.acceleration_structure_build && !ray_queries) return false;
    return true;
}

<* Validate and lower the first destination stages of a queue submission wait. *>
fn vk::PipelineStageFlags2? submit_wait_stages_to_vk(
    gpu::StageMask stages,
    vk::QueueFlags flags,
    bool generated_work = false,
    bool ray_queries = false,
) {
    if (!barrier_stage_mask_valid(stages)
        || stages.host
        || stages.present
        || !barrier_stage_mask_supported(stages, flags, ray_queries)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return barrier_before_stages_to_vk(stages, generated_work);
}

fn ZString barrier_queue_rejected_field(
    gpu::Barrier* barrier,
    vk::QueueFlags flags,
    bool ray_queries = false,
) {
    if (!barrier_stage_mask_supported(barrier.before, flags, ray_queries)) {
        return "Barrier.before";
    }
    if (!barrier_stage_mask_supported(barrier.after, flags, ray_queries)) {
        return "Barrier.after";
    }
    return null;
}

fn bool barrier_queue_flags_valid(
    gpu::Barrier* barrier,
    vk::QueueFlags flags,
) {
    return barrier_queue_rejected_field(barrier, flags) == null;
}

fn gpu::PresentMode present_mode_from_vk(vk::PresentModeKHR mode) {
    switch (mode) {
        case vk::PresentModeKHR.IMMEDIATE_KHR: return gpu::PresentMode.IMMEDIATE;
        case vk::PresentModeKHR.MAILBOX_KHR:   return gpu::PresentMode.MAILBOX;
        default:                               return gpu::PresentMode.FIFO;
    }
}
