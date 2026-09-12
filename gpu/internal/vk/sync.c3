module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

alias BarrierEmitFn = fn void (
    void*, vk::CommandBuffer, vk::DependencyInfo*
);
alias SemaphoreWaitFn @private = fn vk::Result (
    vk::Device,
    vk::SemaphoreWaitInfo*,
    ulong,
);

enum TextureStateSide {
    BEFORE,
    AFTER,
}

struct TextureStateScope {
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2        access;
    vk::ImageLayout         layout;
}

struct LoweredTextureBarrier {
    vk::ImageMemoryBarrier2 native;
}

struct BarrierEmitOps {
    void*         user_data;
    BarrierEmitFn emit;
}

fn bool texture_format_is_depth_stencil(gpu::Format format) @private {
    return gpu::internal::format_is_depth_stencil(format);
}

fn vk::AccessFlags2 texture_access_to_vk(
    gpu::TextureLayout layout,
    gpu::TextureAccess access,
) {
    ulong mask;
    switch (layout) {
        case gpu::TextureLayout.TRANSFER_SOURCE:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT;
            }
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            if (access.write) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT;
            }
        case gpu::TextureLayout.SAMPLED:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT;
            }
        case gpu::TextureLayout.STORAGE:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_READ_BIT;
            }
            if (access.write) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            }
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            }
            if (access.write) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            }
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            if (access.write) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
        case gpu::TextureLayout.GENERAL:
            if (access.read) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT;
            }
            if (access.write) {
                mask |= (ulong)vk::AccessFlagBits2.ACCESS_2_MEMORY_WRITE_BIT;
            }
        default:
            return vk::AccessFlagBits2.ACCESS_2_NONE;
    }
    return (vk::AccessFlags2)mask;
}

<* Unified mode keeps every layout but UNDEFINED in GENERAL. *>
fn vk::ImageLayout texture_layout_to_vk(
    gpu::TextureLayout layout,
    gpu::Format format,
    bool unified = false,
) {
    if (unified && layout != gpu::TextureLayout.UNDEFINED) {
        return vk::ImageLayout.GENERAL;
    }
    switch (layout) {
        case gpu::TextureLayout.UNDEFINED:
            return vk::ImageLayout.UNDEFINED;
        case gpu::TextureLayout.TRANSFER_SOURCE:
            return vk::ImageLayout.TRANSFER_SRC_OPTIMAL;
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            return vk::ImageLayout.TRANSFER_DST_OPTIMAL;
        case gpu::TextureLayout.SAMPLED:
            return texture_format_is_depth_stencil(format)
                ? vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL
                : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL;
        case gpu::TextureLayout.STORAGE:
            return vk::ImageLayout.GENERAL;
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            return vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            return vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.PRESENT:
            return vk::ImageLayout.PRESENT_SRC_KHR;
        case gpu::TextureLayout.GENERAL:
            return vk::ImageLayout.GENERAL;
        default:
            return {};
    }
}

fn TextureStateScope texture_state_to_vk(
    gpu::TextureState state,
    gpu::Format format,
    TextureStateSide side,
    gpu::StageMask paired_stages,
    bool task_shaders = false,
    bool unified = false,
) {
    vk::ImageLayout layout = texture_layout_to_vk(
        state.layout,
        format,
        unified,
    );
    if (unified && state.layout != gpu::TextureLayout.UNDEFINED) {
        return {
            .stages = barrier_before_stages_to_vk(state.stages),
            .access = texture_access_to_vk(gpu::TextureLayout.GENERAL, state.access),
            .layout = layout,
        };
    }
    switch (state.layout) {
        case gpu::TextureLayout.UNDEFINED:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = layout,
            };
        case gpu::TextureLayout.TRANSFER_SOURCE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.SAMPLED:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.STORAGE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages, false, task_shaders),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.GENERAL:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.PRESENT:
            return {
                // Pair the consumer stage to link the transition with readiness_before's stage-limited acquire wait.
                .stages = side == TextureStateSide.BEFORE
                    ? barrier_before_stages_to_vk(paired_stages, false, task_shaders)
                    : vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = layout,
            };
        default:
            return {};
    }
}

fn vk::ImageSubresourceRange? resolve_texture_barrier_range(
    gpu::TextureViewDesc view,
    TextureSlot* slot,
) @private {
    if (view.aspect != gpu::TextureAspect.AUTO
        || view.base_mip >= slot.mip_levels
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

fn void real_barrier_emit(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::DependencyInfo* dependency,
) @private {
    (void)user_data;
    vk::cmd_pipeline_barrier2(command_buffer, dependency);
}

fn BarrierEmitOps barrier_emit_ops_real() @private {
    return { .emit = &real_barrier_emit };
}

fn void emit_barrier(
    CommandRecord* commands,
    vk::MemoryBarrier2* native_barrier,
    BarrierEmitOps* ops,
) {
    CommandRecord* record = commands;
    vk::DependencyInfo dependency =
        vk::dependency_info().set_memory_barriers(native_barrier[:1]);
    ops.emit(ops.user_data, record.command_buffer, &dependency);
}

fn void? vk_cmd_barrier(
    CommandRecord* commands,
    gpu::Barrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    return vk_cmd_barrier_with_ops(commands, barrier, &ops);
}

fn LoweredTextureBarrier? lower_texture_barrier(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
) {
    if (barrier == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    TextureSlot* slot = state.textures.get(barrier.texture)!;
    vk::ImageSubresourceRange range = resolve_texture_barrier_range(
        barrier.view,
        slot,
    )!;
    TextureStateScope before = texture_state_to_vk(
        state:         barrier.before,
        format:        slot.format,
        side:          TextureStateSide.BEFORE,
        paired_stages: barrier.after.stages,
        task_shaders:  state.task_shaders_enabled,
        unified:       state.unified_layouts,
    );
    TextureStateScope after = texture_state_to_vk(
        state:         barrier.after,
        format:        slot.format,
        side:          TextureStateSide.AFTER,
        paired_stages: barrier.before.stages,
        task_shaders:  state.task_shaders_enabled,
        unified:       state.unified_layouts,
    );
    vk::ImageMemoryBarrier2 native = vk::image_memory_barrier2()
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
    return { .native = native };
}

fn void emit_texture_barrier(
    CommandRecord* commands,
    LoweredTextureBarrier* lowered,
    BarrierEmitOps* ops,
) {
    CommandRecord* record = commands;
    vk::DependencyInfo dependency = vk::dependency_info()
        .set_image_memory_barriers((&lowered.native)[:1]);
    ops.emit(ops.user_data, record.command_buffer, &dependency);
}

fn void? vk_cmd_texture_barrier_with_ops(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
    BarrierEmitOps* ops,
) {
    LoweredTextureBarrier lowered = lower_texture_barrier(
        commands,
        barrier,
    )!;
    emit_texture_barrier(commands, &lowered, ops);
}

fn void? vk_cmd_texture_barrier(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    return vk_cmd_texture_barrier_with_ops(commands, barrier, &ops);
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

fn bool? poll_queue_completion_with_ops(
    VkDeviceState* state,
    QueueCompletionState* completion,
    uint queue_id,
    ulong sequence,
    CompletionRetirementOps* ops,
) @private {
    if (completion.retired_sequence.load(AtomicOrdering.ACQUIRE) >= sequence) {
        return true;
    }
    ulong current;
    check_backend_result(
        state:     state,
        operation: "poll_completion",
        result:    ops.query(
            state.device,
            completion.timeline,
            &current,
        ),
    )!;
    if (current < sequence) return false;
    retire_observed_completion_and_drain_with_ops(
        state:             state,
        observed_queue_id: queue_id,
        observed_completed: current,
        ops:               ops,
    )!;
    return completion.retired_sequence.load(AtomicOrdering.ACQUIRE) >= sequence;
}

fn bool? poll_queue_completion_with_query(
    VkDeviceState* state,
    QueueCompletionState* completion,
    uint queue_id,
    ulong sequence,
    SemaphoreCounterQueryFn query_fn,
) @private {
    CompletionRetirementOps ops = { .query = query_fn };
    return poll_queue_completion_with_ops(
        state:      state,
        completion: completion,
        queue_id:   queue_id,
        sequence:   sequence,
        ops:        &ops,
    );
}

fn bool? vk_poll_completion_with_query(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
    SemaphoreCounterQueryFn query_fn,
) @private {
    QueueCompletionState* completion = published_queue_completion_state(
        state,
        queue_id,
        sequence,
    )!;
    return poll_queue_completion_with_query(
        state:      state,
        completion: completion,
        queue_id:   queue_id,
        sequence:   sequence,
        query_fn:   query_fn,
    );
}

fn bool? vk_poll_completion_with_ops(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
    CompletionRetirementOps* ops,
) {
    QueueCompletionState* completion = published_queue_completion_state(
        state,
        queue_id,
        sequence,
    )!;
    return poll_queue_completion_with_ops(
        state:      state,
        completion: completion,
        queue_id:   queue_id,
        sequence:   sequence,
        ops:        ops,
    );
}

fn bool? vk_poll_completion(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
) {
    return vk_poll_completion_with_query(
        state:    state,
        queue_id: queue_id,
        sequence: sequence,
        query_fn: &real_semaphore_counter_query,
    );
}

fn vk::Result real_semaphore_wait(
    vk::Device device,
    vk::SemaphoreWaitInfo* wait_info,
    ulong timeout_ns,
) @private {
    return vk::wait_semaphores(device, wait_info, timeout_ns);
}

fn void? vk_wait_completion_with_wait(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
    ulong timeout_ns,
    SemaphoreWaitFn wait_fn,
) @private {
    QueueCompletionState* completion = published_queue_completion_state(
        state,
        queue_id,
        sequence,
    )!;
    if (completion.retired_sequence.load(AtomicOrdering.ACQUIRE) >= sequence) {
        return;
    }
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&completion.timeline)[:1])
        .set_values((&sequence)[:1]);
    check_wait_backend_result(
        state:     state,
        operation: "wait_completion",
        result:    wait_fn(
            state.device,
            &wait_info,
            timeout_ns,
        ),
    )!;
    retire_observed_completion_and_drain_with_query(
        state:              state,
        observed_queue_id:  queue_id,
        observed_completed: sequence,
        query_fn:           &real_semaphore_counter_query,
    )!;
}

fn void? vk_wait_completion(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
    ulong timeout_ns,
) {
    return vk_wait_completion_with_wait(
        state:      state,
        queue_id:   queue_id,
        sequence:   sequence,
        timeout_ns: timeout_ns,
        wait_fn:    &real_semaphore_wait,
    );
}

fn void? vk_cmd_barrier_with_ops(
    CommandRecord* commands,
    gpu::Barrier* barrier,
    BarrierEmitOps* ops,
) {
    if (barrier == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    vk::MemoryBarrier2 native_barrier = barrier_to_vk(
        barrier,
        state.generated_work.enabled,
        state.ray_queries_enabled,
        state.task_shaders_enabled,
    );
    emit_barrier(commands, &native_barrier, ops);
}
