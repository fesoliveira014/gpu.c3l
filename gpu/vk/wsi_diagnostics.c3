module gpu::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

struct WsiDiagnosticContext {
    VkDeviceState* state;
    ZString        operation;
    bool           has_public_identity;
    uint           index;
    uint           generation;
    ZString        debug_name;
}

fn fault report_wsi_contract(
    WsiDiagnosticContext* context,
    fault public_fault,
    ZString rejected_field,
    ZString invariant,
    ZString backend_text = null,
) {
    return report_contract_failure(
        state:               context.state,
        operation:           context.operation,
        public_fault:        public_fault,
        resource_kind:       gpu::DebugResourceKind.SWAPCHAIN,
        has_public_identity: context.has_public_identity,
        index:               context.index,
        generation:          context.generation,
        debug_name:          context.debug_name,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        backend_text,
    );
}

fn fault report_wsi_backend_failure(
    WsiDiagnosticContext* context,
    fault public_fault,
    ZString backend_text,
    int raw_result = 0,
) {
    gpu::DebugMessage message = {
        .severity             = gpu::DebugMessageSeverity.ERROR,
        .category             = { .backend },
        .operation            = context.operation,
        .has_fault            = true,
        .public_fault         = public_fault,
        .resource             = {
            .kind                = gpu::DebugResourceKind.SWAPCHAIN,
            .has_public_identity = context.has_public_identity,
            .index               = context.index,
            .generation          = context.generation,
            .debug_name          = context.debug_name,
        },
        .backend_text         = backend_text,
        .validation_id_number = raw_result,
    };
    gpu::internal::dispatch_debug_message(context.state.debug_callback, context.state.debug_user_data, &message);
    return public_fault;
}

fn ZString wsi_backend_text(vk::Result result) @private {
    switch (result) {
        case vk::Result.TIMEOUT:                             return "Vulkan image acquire timed out";
        case vk::Result.NOT_READY:                           return "Vulkan image acquire is not ready";
        case vk::Result.ERROR_OUT_OF_DATE_KHR:               return "Vulkan swapchain is out of date";
        case vk::Result.ERROR_SURFACE_LOST_KHR:              return "Vulkan surface was lost";
        case vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR:      return "native window is already in use";
        case vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: return "swapchain image usage is unsupported";
        case vk::Result.INCOMPLETE:                          return "Vulkan enumeration was incomplete";
        default:                                             return backend_result_text(result);
    }
}

fn void? check_wsi_backend_result(
    WsiDiagnosticContext* context,
    WsiResultKind kind,
    vk::Result result,
) {
    observe_device_result(context.state, result);
    bool stderr_fallback = context.state == null || context.state.debug_callback == null;
    fault public_fault = @catch(map_wsi_result(kind, result, stderr_fallback));
    if (!public_fault) return;
    if (kind == WsiResultKind.ACQUIRE
        && (result == vk::Result.TIMEOUT || result == vk::Result.NOT_READY)) {
        return public_fault~;
    }
    if (!stderr_fallback) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: public_fault,
            backend_text: wsi_backend_text(result),
            raw_result:   (int)result,
        )~;
    }
    return public_fault~;
}

fn SwapchainSlot*? resolve_swapchain_for_operation(
    WsiDiagnosticContext* context,
    gpu::SwapchainHandle handle,
) {
    SwapchainSlot*? resolved = context.state.presentation.swapchains.get(handle);
    if (catch lookup_fault = resolved) {
        return report_wsi_contract(
            context:        context,
            public_fault:   lookup_fault,
            rejected_field: "SwapchainHandle",
            invariant:      "must identify a live swapchain",
        )~;
    }
    SwapchainSlot* slot = resolved;
    context.has_public_identity = true;
    context.index = handle.index;
    context.generation = handle.generation;
    context.debug_name = slot.requested.debug_name;
    return slot;
}

fn TextureSlot*? resolve_swapchain_texture_for_operation(
    WsiDiagnosticContext* context,
    gpu::TextureHandle handle,
) {
    TextureSlot*? resolved = context.state.textures.get(handle);
    if (catch lookup_fault = resolved) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: lookup_fault,
            backend_text: "swapchain image table invariant failed",
        )~;
    }
    return resolved;
}

fn void? check_swapchain_acquire_lifecycle(
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
) {
    if (slot.info.dormant || slot.swapchain == null) {
        return gpu::SWAPCHAIN_OUT_OF_DATE~;
    }
    if (slot.acquire_pending) {
        return report_wsi_contract(
            context:        context,
            public_fault:   gpu::INVALID_RESOURCE_STATE,
            rejected_field: "swapchain acquire lifecycle",
            invariant:      "previous acquired image must be submitted and presented first",
        )~;
    }
}

fn void? check_swapchain_present_contract(
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
) {
    fault guard_fault = @catch(swapchain_present_guard(
        acquire_pending:      slot.acquire_pending,
        readiness_consumed:   slot.readiness_consumed,
        expected_acquisition: slot.acquisition,
        provided_acquisition: swapchain_readiness_acquisition(
            image.readiness,
        ),
        expected_completion:  slot.render_completion,
        provided_completion:  render_completion,
    ));
    if (guard_fault) {
        bool wrong_completion = guard_fault == gpu::INVALID_ARGUMENT;
        return report_wsi_contract(
            context:        context,
            public_fault:   guard_fault,
            rejected_field: wrong_completion
                ? "render_completion"
                : "acquired image readiness",
            invariant:      wrong_completion
                ? "must be returned by the readiness-consuming submission"
                : "must match the pending acquisition after readiness submission",
        )~;
    }
    if (image.index != slot.pending_image
        || image.texture != slot.textures[slot.pending_image]) {
        return report_wsi_contract(
            context:        context,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "AcquiredImage",
            invariant:      "must match the current acquired swapchain image",
        )~;
    }
}
