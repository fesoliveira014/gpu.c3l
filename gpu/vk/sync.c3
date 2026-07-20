module gpu::vk @private;

import gpu @public;
import vk;

struct TextureUseScope {
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2        access;
    vk::ImageLayout         layout;
}

struct TextureBarrierRejection {
    ZString field;
    ZString invariant;
}

fn bool texture_format_is_depth_stencil(gpu::Format format) @private {
    return format == gpu::Format.D32_FLOAT
        || format == gpu::Format.D24_UNORM_S8_UINT;
}

fn TextureUseScope? texture_use_to_vk(
    gpu::TextureUse use,
    gpu::Format format,
) {
    bool depth_stencil = texture_format_is_depth_stencil(format);
    switch (use) {
        case gpu::TextureUse.UNDEFINED:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = vk::ImageLayout.UNDEFINED,
            };
        case gpu::TextureUse.TRANSFER_SOURCE:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT,
                .layout = vk::ImageLayout.TRANSFER_SRC_OPTIMAL,
            };
        case gpu::TextureUse.TRANSFER_DESTINATION:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT,
                .layout = vk::ImageLayout.TRANSFER_DST_OPTIMAL,
            };
        case gpu::TextureUse.SAMPLED_COMPUTE:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .layout = depth_stencil
                    ? vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            };
        case gpu::TextureUse.SAMPLED_FRAGMENT:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .layout = depth_stencil
                    ? vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            };
        case gpu::TextureUse.STORAGE_COMPUTE:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
                .access = (vk::AccessFlags2)(
                    (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_READ_BIT
                    | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_WRITE_BIT
                ),
                .layout = vk::ImageLayout.GENERAL,
            };
        case gpu::TextureUse.COLOR_ATTACHMENT:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access = (vk::AccessFlags2)(
                    (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
                    | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                ),
                .layout = vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
            };
        case gpu::TextureUse.DEPTH_ATTACHMENT:
            return {
                .stages = (vk::PipelineStageFlags2)(
                    (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                    | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT
                ),
                .access = (vk::AccessFlags2)(
                    (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                    | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                ),
                .layout = vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
        case gpu::TextureUse.PRESENT:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = vk::ImageLayout.PRESENT_SRC_KHR,
            };
        default:
            return gpu::INVALID_ARGUMENT~;
    }
}

fn bool texture_use_known(gpu::TextureUse use) @private {
    switch (use) {
        case gpu::TextureUse.UNDEFINED:
        case gpu::TextureUse.TRANSFER_SOURCE:
        case gpu::TextureUse.TRANSFER_DESTINATION:
        case gpu::TextureUse.SAMPLED_COMPUTE:
        case gpu::TextureUse.SAMPLED_FRAGMENT:
        case gpu::TextureUse.STORAGE_COMPUTE:
        case gpu::TextureUse.COLOR_ATTACHMENT:
        case gpu::TextureUse.DEPTH_ATTACHMENT:
        case gpu::TextureUse.PRESENT:
            return true;
        default:
            return false;
    }
}

fn bool texture_use_stage_supported_by_queue(
    gpu::TextureUse use,
    vk::QueueFlags flags,
) @private {
    bool graphics = queue_flags_include(flags, vk::QueueFlagBits.GRAPHICS_BIT);
    bool compute = queue_flags_include(flags, vk::QueueFlagBits.COMPUTE_BIT);
    bool transfer = queue_flags_include(flags, vk::QueueFlagBits.TRANSFER_BIT);
    bool admitted = graphics || compute || transfer;
    switch (use) {
        case gpu::TextureUse.UNDEFINED:
        case gpu::TextureUse.TRANSFER_SOURCE:
        case gpu::TextureUse.TRANSFER_DESTINATION:
            return admitted;
        case gpu::TextureUse.SAMPLED_COMPUTE:
        case gpu::TextureUse.STORAGE_COMPUTE:
            return compute;
        case gpu::TextureUse.SAMPLED_FRAGMENT:
        case gpu::TextureUse.COLOR_ATTACHMENT:
        case gpu::TextureUse.DEPTH_ATTACHMENT:
        case gpu::TextureUse.PRESENT:
            return graphics;
        default:
            return false;
    }
}

fn TextureBarrierRejection texture_barrier_queue_rejection(
    gpu::TextureBarrier* barrier,
    vk::QueueFlags flags,
) @private {
    if (!texture_use_stage_supported_by_queue(barrier.before, flags)) {
        return {
            .field     = "TextureBarrier.before",
            .invariant = "before use stage must be supported by the recording queue",
        };
    }
    if (!texture_use_stage_supported_by_queue(barrier.after, flags)) {
        return {
            .field     = "TextureBarrier.after",
            .invariant = "after use stage must be supported by the recording queue",
        };
    }
    return {};
}

fn bool texture_use_matches_format(
    gpu::TextureUse use,
    gpu::Format format,
) @private {
    bool depth_stencil = texture_format_is_depth_stencil(format);
    switch (use) {
        case gpu::TextureUse.COLOR_ATTACHMENT:
        case gpu::TextureUse.PRESENT:
            return !depth_stencil;
        case gpu::TextureUse.DEPTH_ATTACHMENT:
            return depth_stencil;
        default:
            return true;
    }
}

fn bool texture_use_supported_by_slot(
    gpu::TextureUse use,
    TextureSlot* slot,
) @private {
    if (!texture_use_matches_format(use, slot.format)) return false;
    switch (use) {
        case gpu::TextureUse.UNDEFINED:
            return true;
        case gpu::TextureUse.TRANSFER_SOURCE:
            return slot.usage.transfer_src;
        case gpu::TextureUse.TRANSFER_DESTINATION:
            return slot.usage.transfer_dst;
        case gpu::TextureUse.SAMPLED_COMPUTE:
        case gpu::TextureUse.SAMPLED_FRAGMENT:
            return slot.usage.sampled;
        case gpu::TextureUse.STORAGE_COMPUTE:
            return slot.usage.storage;
        case gpu::TextureUse.COLOR_ATTACHMENT:
            return slot.usage.color_attach;
        case gpu::TextureUse.DEPTH_ATTACHMENT:
            return slot.usage.depth_attach;
        case gpu::TextureUse.PRESENT:
            return slot.swapchain_owned;
        default:
            return false;
    }
}

fn vk::ImageSubresourceRange? texture_transition_range(
    gpu::TextureViewDesc view,
    TextureSlot* slot,
) @private {
    if (view.format != gpu::Format.UNDEFINED && view.format != slot.format) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (view.base_mip >= slot.mip_levels
        || view.base_layer >= slot.array_layers) {
        return gpu::INVALID_ARGUMENT~;
    }

    uint mip_count = view.mip_count == 0
        ? slot.mip_levels - view.base_mip
        : view.mip_count;
    uint layer_count = view.layer_count == 0
        ? slot.array_layers - view.base_layer
        : view.layer_count;
    if (mip_count > slot.mip_levels - view.base_mip
        || layer_count > slot.array_layers - view.base_layer) {
        return gpu::INVALID_ARGUMENT~;
    }

    return {
        .aspect_mask      = format_aspect(slot.format),
        .base_mip_level   = view.base_mip,
        .level_count      = mip_count,
        .base_array_layer = view.base_layer,
        .layer_count      = layer_count,
    };
}

fn TextureBarrierRejection texture_barrier_rejection(
    gpu::TextureBarrier* barrier,
    TextureSlot* slot,
) @private {
    if (barrier == null) {
        return {
            .field     = "TextureBarrier",
            .invariant = "barrier must be non-null",
        };
    }
    if (slot == null) return {};
    if (!texture_use_known(barrier.before)) {
        return {
            .field     = "TextureBarrier.before",
            .invariant = "before use must be valid",
        };
    }
    if (!texture_use_known(barrier.after)
        || barrier.after == gpu::TextureUse.UNDEFINED) {
        return {
            .field     = "TextureBarrier.after",
            .invariant = "after use must be a valid concrete texture use",
        };
    }
    if (!texture_use_supported_by_slot(barrier.before, slot)) {
        return {
            .field     = "TextureBarrier.before",
            .invariant = "before use must be supported by the texture",
        };
    }
    if (!texture_use_supported_by_slot(barrier.after, slot)) {
        return {
            .field     = "TextureBarrier.after",
            .invariant = "after use must be supported by the texture",
        };
    }
    if (catch texture_transition_range(barrier.view, slot)) {
        return {
            .field     = "TextureBarrier.view",
            .invariant = "view must select a valid texture subresource range",
        };
    }
    return {};
}

fn vk::ImageMemoryBarrier2? texture_barrier_to_vk(
    gpu::TextureBarrier* barrier,
    TextureSlot* slot,
) {
    if (barrier == null || slot == null
        || barrier.after == gpu::TextureUse.UNDEFINED
        || !texture_use_matches_format(barrier.before, slot.format)
        || !texture_use_matches_format(barrier.after, slot.format)) {
        return gpu::INVALID_ARGUMENT~;
    }

    TextureUseScope before = texture_use_to_vk(
        barrier.before,
        slot.format,
    )!;
    TextureUseScope after = texture_use_to_vk(
        barrier.after,
        slot.format,
    )!;
    vk::ImageSubresourceRange range = texture_transition_range(
        barrier.view,
        slot,
    )!;

    return vk::image_memory_barrier2()
        .set_src_stage_mask(before.stages)
        .set_src_access_mask(before.access)
        .set_dst_stage_mask(after.stages)
        .set_dst_access_mask(after.access)
        .set_old_layout(before.layout)
        .set_new_layout(after.layout)
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_image(slot.image)
        .set_subresource_range(range);
}

fn void? vk_cmd_barrier(
    gpu::CommandList* commands,
    gpu::Barrier* barrier,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    ZString rejected_field = barrier_masks_rejected_field(barrier);
    ZString invariant;
    if (rejected_field != null) {
        if (rejected_field.str_view() == "Barrier") {
            invariant = "barrier must be non-null";
        } else if (rejected_field.str_view() == "Barrier.hazards") {
            invariant = "hazard flags must be known and match the consumer stages";
        } else {
            invariant = "stage mask must be nonempty and noncontradictory";
        }
    } else {
        QueueCompletionState* completion = queue_completion_state(
            state,
            record.queue.id,
        )!;
        rejected_field = barrier_queue_rejected_field(
            barrier,
            completion.queue_flags,
        );
        if (rejected_field != null) {
            invariant = rejected_field.str_view() == "Barrier.hazards"
                ? "hazard flags must be supported by the recording queue"
                : "stage mask must be supported by the recording queue";
        }
    }
    if (rejected_field != null) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_barrier",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           invariant,
            backend_text:        null,
        )~;
    }

    bool descriptor_buffer = state.strict_heap.implementation
        == VkHeapImplementation.DESCRIPTOR_BUFFER;
    vk::MemoryBarrier2 native_barrier = barrier_to_vk(
        barrier,
        descriptor_buffer,
        state.generated_work.enabled,
    );
    vk::DependencyInfo dependency =
        vk::dependency_info().set_memory_barriers((&native_barrier)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dependency);
}

fn void? vk_cmd_texture_barrier(
    gpu::CommandList* commands,
    gpu::TextureBarrier* barrier,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TextureBarrierRejection rejection = texture_barrier_rejection(
        barrier,
        null,
    );
    if (rejection.field != null) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_texture_barrier",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejection.field,
            invariant:           rejection.invariant,
            backend_text:        null,
        )~;
    }

    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(
        state,
        record,
        reference_checkpoint,
    );
    TextureSlot* slot = state.textures.get(barrier.texture)!;
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        handle:    barrier.texture,
        operation: "cmd_texture_barrier",
    )!;
    rejection = texture_barrier_rejection(barrier, slot);
    if (rejection.field == null) {
        QueueCompletionState* completion = queue_completion_state(
            state,
            record.queue.id,
        )!;
        rejection = texture_barrier_queue_rejection(
            barrier,
            completion.queue_flags,
        );
    }
    if (rejection.field != null) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_texture_barrier",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               barrier.texture.index,
            generation:          barrier.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      rejection.field,
            invariant:           rejection.invariant,
            backend_text:        null,
        )~;
    }

    vk::ImageMemoryBarrier2 native_barrier = texture_barrier_to_vk(
        barrier,
        slot,
    )!;
    vk::DependencyInfo dependency = vk::dependency_info()
        .set_image_memory_barriers((&native_barrier)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dependency);
}

fn vk::Semaphore? create_private_timeline(VkDeviceState* state) @private {
    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value(0);
    vk::SemaphoreCreateInfo info = vk::semaphore_create_info()
        .set_next(&type_info);
    vk::Semaphore semaphore;
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &info,
        p_allocator:   null,
        p_semaphore:   &semaphore,
    ))!;
    return semaphore;
}

fn bool? vk_poll_completion(
    gpu::Device* device,
    uint queue_id,
    ulong sequence,
) @private {
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(device);
    QueueCompletionState* completion = published_queue_completion_state(
        state,
        queue_id,
        sequence,
    )!;
    ulong current;
    check_backend_result(
        state:     state,
        operation: "poll_completion",
        result:    vk::get_semaphore_counter_value(
            state.device,
            completion.timeline,
            &current,
        ),
    )!;
    if (current < sequence) return false;
    observe_queue_completion(completion, current);
    release_completed_submitted_commands(
        state,
        queue_id,
        current,
    );
    return true;
}

fn void? vk_wait_completion(
    gpu::Device* device,
    uint queue_id,
    ulong sequence,
    ulong timeout_ns,
) @private {
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.submit_stats.completion_wait_calls++;
    QueueCompletionState* completion = published_queue_completion_state(
        state,
        queue_id,
        sequence,
    )!;
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&completion.timeline)[:1])
        .set_values((&sequence)[:1]);
    check_wait_backend_result(
        state:     state,
        operation: "wait_completion",
        result:    vk::wait_semaphores(
            state.device,
            &wait_info,
            timeout_ns,
        ),
    )!;
    observe_queue_completion(completion, sequence);
    release_completed_submitted_commands(
        state,
        queue_id,
        sequence,
    );
}
