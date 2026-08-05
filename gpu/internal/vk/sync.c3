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

const gpu::StageMask TEXTURE_STATE_STAGE_MASK @private = {
    .all,
    .host,
    .transfer,
    .compute,
    .vertex_shader,
    .fragment_shader,
    .ray_tracing,
    .color_output,
    .depth_output,
    .present,
};
const gpu::StageMask TEXTURE_ALL_STAGE @private = { .all };
const gpu::StageMask TEXTURE_TRANSFER_STAGE @private = { .transfer };
const gpu::StageMask TEXTURE_SHADER_STAGE_MASK @private = {
    .compute,
    .vertex_shader,
    .fragment_shader,
    .ray_tracing,
};
const gpu::StageMask TEXTURE_COLOR_OUTPUT_STAGE @private = { .color_output };
const gpu::StageMask TEXTURE_DEPTH_OUTPUT_STAGE @private = { .depth_output };
const gpu::TextureAccess TEXTURE_ACCESS_MASK @private = { .read, .write };
const gpu::TextureAccess TEXTURE_READ_ACCESS @private = { .read };
const gpu::TextureAccess TEXTURE_WRITE_ACCESS @private = { .write };

enum TextureStateSide {
    BEFORE,
    AFTER,
}

enum TextureStateMember {
    LAYOUT,
    STAGES,
    ACCESS,
}

struct TextureStateScope {
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2        access;
    vk::ImageLayout         layout;
}

struct LoweredTextureBarrier {
    TextureSlot*            slot;
    vk::ImageMemoryBarrier2 native;
}

struct TextureBarrierRejection {
    ZString field;
    ZString invariant;
}

struct BarrierEmitOps {
    void*         user_data;
    BarrierEmitFn emit;
}

fn bool texture_format_is_depth_stencil(gpu::Format format) @private {
    return format == gpu::Format.D32_FLOAT;
}

fn bool texture_layout_known(gpu::TextureLayout layout) @private {
    switch (layout) {
        case gpu::TextureLayout.UNDEFINED:
        case gpu::TextureLayout.TRANSFER_SOURCE:
        case gpu::TextureLayout.TRANSFER_DESTINATION:
        case gpu::TextureLayout.SAMPLED:
        case gpu::TextureLayout.STORAGE:
        case gpu::TextureLayout.COLOR_ATTACHMENT:
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
        case gpu::TextureLayout.PRESENT:
            return true;
        default:
            return false;
    }
}

<*
 @require ((uint)access & ~(uint)TEXTURE_ACCESS_MASK) == 0 : "known texture access bits"
*>
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
        default:
            return vk::AccessFlagBits2.ACCESS_2_NONE;
    }
    return (vk::AccessFlags2)mask;
}

fn vk::ImageLayout texture_layout_to_vk(
    gpu::TextureLayout layout,
    gpu::Format format,
) {
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
        default:
            return {};
    }
}

fn TextureStateScope texture_state_to_vk(
    gpu::TextureState state,
    gpu::Format format,
    TextureStateSide side,
    gpu::StageMask paired_stages,
) {
    vk::ImageLayout layout = texture_layout_to_vk(
        state.layout,
        format,
    );
    switch (state.layout) {
        case gpu::TextureLayout.UNDEFINED:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = layout,
            };
        case gpu::TextureLayout.TRANSFER_SOURCE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.SAMPLED:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.STORAGE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = layout,
            };
        case gpu::TextureLayout.PRESENT:
            return {
                // Pair the consumer stage to link the transition with readiness_before's stage-limited acquire wait.
                .stages = side == TextureStateSide.BEFORE
                    ? barrier_before_stages_to_vk(paired_stages)
                    : vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = layout,
            };
        default:
            return {};
    }
}

fn ZString texture_state_field(
    TextureStateSide side,
    TextureStateMember member,
) @private {
    switch (member) {
        case TextureStateMember.LAYOUT:
            return side == TextureStateSide.BEFORE
                ? "TextureBarrier.before.layout"
                : "TextureBarrier.after.layout";
        case TextureStateMember.STAGES:
            return side == TextureStateSide.BEFORE
                ? "TextureBarrier.before.stages"
                : "TextureBarrier.after.stages";
        case TextureStateMember.ACCESS:
            return side == TextureStateSide.BEFORE
                ? "TextureBarrier.before.access"
                : "TextureBarrier.after.access";
    }
}

fn bool texture_shader_stages_supported(
    gpu::StageMask stages,
    vk::QueueFlags flags,
    bool           ray_tracing_pipelines,
) @private {
    bool graphics = queue_flags_include(flags, vk::QueueFlagBits.GRAPHICS_BIT);
    bool compute = queue_flags_include(flags, vk::QueueFlagBits.COMPUTE_BIT);
    if (stages.compute && !compute) return false;
    if ((stages.vertex_shader || stages.fragment_shader) && !graphics) {
        return false;
    }
    if (stages.ray_tracing && (!(graphics || compute)
            || !ray_tracing_pipelines)) {
        return false;
    }
    return true;
}

fn TextureBarrierRejection texture_state_rejection(
    gpu::TextureState texture_state,
    TextureSlot* slot,
    vk::QueueFlags flags,
    bool ray_tracing_pipelines,
    bool before,
) @private {
    TextureStateSide side = before
        ? TextureStateSide.BEFORE
        : TextureStateSide.AFTER;
    if (!texture_layout_known(texture_state.layout)) {
        return {
            .field     = texture_state_field(side, TextureStateMember.LAYOUT),
            .invariant = "layout must be a known TextureLayout value",
        };
    }
    uint stages = (uint)texture_state.stages;
    uint access = (uint)texture_state.access;
    if ((stages & (uint)TEXTURE_STATE_STAGE_MASK) != stages
        || texture_state.stages.host
        || texture_state.stages.present
        || (texture_state.stages.all
            && stages != (uint)TEXTURE_ALL_STAGE)) {
        return {
            .field     = texture_state_field(side, TextureStateMember.STAGES),
            .invariant = "stages must contain known non-host pipeline stages and all must be exclusive",
        };
    }
    if ((access & (uint)TEXTURE_ACCESS_MASK) != access) {
        return {
            .field     = texture_state_field(side, TextureStateMember.ACCESS),
            .invariant = "access must contain only read and write bits",
        };
    }

    bool graphics = queue_flags_include(flags, vk::QueueFlagBits.GRAPHICS_BIT);
    bool compute = queue_flags_include(flags, vk::QueueFlagBits.COMPUTE_BIT);
    bool transfer = queue_flags_include(flags, vk::QueueFlagBits.TRANSFER_BIT)
        || graphics || compute;
    bool shader_queue = graphics || compute;
    bool shader_stages = texture_state.stages.compute
        || texture_state.stages.vertex_shader
        || texture_state.stages.fragment_shader
        || texture_state.stages.ray_tracing;
    bool exact_read = access == (uint)TEXTURE_READ_ACCESS;
    bool exact_write = access == (uint)TEXTURE_WRITE_ACCESS;
    bool concrete_access = access != 0;
    bool all = texture_state.stages.all;
    bool depth_stencil = texture_format_is_depth_stencil(slot.format);

    switch (texture_state.layout) {
        case gpu::TextureLayout.UNDEFINED:
            if (side == TextureStateSide.AFTER) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "UNDEFINED is source-only",
                };
            }
            if (stages != 0) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "UNDEFINED requires empty stages",
                };
            }
            if (access != 0) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "UNDEFINED requires empty access",
                };
            }
        case gpu::TextureLayout.TRANSFER_SOURCE:
            if (!(all || (texture_state.stages.transfer
                    && stages == (uint)TEXTURE_TRANSFER_STAGE))
                || !transfer) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "transfer source requires transfer or all stages on a transfer-capable queue",
                };
            }
            if (!exact_read) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "transfer source requires read-only access",
                };
            }
            if (!slot.usage.transfer_src) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "texture usage must include transfer_src",
                };
            }
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            if (!(all || (texture_state.stages.transfer
                    && stages == (uint)TEXTURE_TRANSFER_STAGE))
                || !transfer) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "transfer destination requires transfer or all stages on a transfer-capable queue",
                };
            }
            if (!exact_write) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "transfer destination requires write-only access",
                };
            }
            if (!slot.usage.transfer_dst) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "texture usage must include transfer_dst",
                };
            }
        case gpu::TextureLayout.SAMPLED:
            if (!(all || (shader_stages
                    && (stages & ~(uint)TEXTURE_SHADER_STAGE_MASK) == 0))
                || !shader_queue
                || !texture_shader_stages_supported(
                    texture_state.stages,
                    flags,
                    ray_tracing_pipelines,
                )) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "sampled textures require supported shader stages or all",
                };
            }
            if (!exact_read) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "sampled textures require read-only access",
                };
            }
            if (!slot.usage.sampled) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "texture usage must include sampled",
                };
            }
        case gpu::TextureLayout.STORAGE:
            if (!(all || (shader_stages
                    && (stages & ~(uint)TEXTURE_SHADER_STAGE_MASK) == 0))
                || !shader_queue
                || !texture_shader_stages_supported(
                    texture_state.stages,
                    flags,
                    ray_tracing_pipelines,
                )) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "storage textures require supported shader stages or all",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "storage textures require read, write, or read-write access",
                };
            }
            if (!slot.usage.storage) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "texture usage must include storage",
                };
            }
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            if (!(all || (texture_state.stages.color_output
                    && stages == (uint)TEXTURE_COLOR_OUTPUT_STAGE)) || !graphics) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "color attachment requires color_output or all on a graphics queue",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "color attachment requires read, write, or read-write access",
                };
            }
            if (depth_stencil || !slot.usage.color_attach) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "color attachment requires a non-depth color_attach texture",
                };
            }
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            if (!(all || (texture_state.stages.depth_output
                    && stages == (uint)TEXTURE_DEPTH_OUTPUT_STAGE)) || !graphics) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "depth attachment requires depth_output or all on a graphics queue",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "depth attachment requires read, write, or read-write access",
                };
            }
            if (!depth_stencil || !slot.usage.depth_attach) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "depth attachment requires a depth_attach depth texture",
                };
            }
        case gpu::TextureLayout.PRESENT:
            if (stages != 0) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.STAGES),
                    .invariant = "PRESENT requires empty public stages",
                };
            }
            if (access != 0) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.ACCESS),
                    .invariant = "PRESENT requires empty public access",
                };
            }
            if (!slot.swapchain_owned || depth_stencil || !graphics) {
                return {
                    .field     = texture_state_field(side, TextureStateMember.LAYOUT),
                    .invariant = "PRESENT requires a swapchain color texture on a graphics queue",
                };
            }
        default:
            unreachable();
    }
    return {};
}

fn vk::ImageSubresourceRange? resolve_texture_barrier_range(
    gpu::TextureViewDesc view,
    TextureSlot* slot,
) @private {
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

fn fault report_texture_barrier_rejection(
    VkDeviceState* state,
    gpu::TextureBarrier* barrier,
    TextureSlot* slot,
    TextureBarrierRejection rejection,
    fault public_fault = gpu::INVALID_ARGUMENT,
) @private {
    return report_contract_failure(
        state:               state,
        operation:           "cmd_texture_barrier",
        public_fault:        public_fault,
        resource_kind:       barrier == null
            ? gpu::DebugResourceKind.NONE
            : gpu::DebugResourceKind.TEXTURE,
        has_public_identity: slot != null,
        index:               slot == null ? 0 : barrier.texture.index,
        generation:          slot == null ? 0 : barrier.texture.generation,
        debug_name:          slot == null ? null : slot_name(&slot.debug_name[0]),
        rejected_field:      rejection.field,
        invariant:           rejection.invariant,
        backend_text:        null,
    );
}

fn LoweredTextureBarrier? validate_and_lower_resolved_texture_barrier(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureBarrier* barrier,
    TextureSlot* slot,
    ZString operation,
) {
    validate_texture_queue_access(
        state:       state,
        queue_roles: record.queue.roles,
        slot:        slot,
        handle:      barrier.texture,
        operation:   operation,
    )!;

    vk::ImageSubresourceRange? range_result = resolve_texture_barrier_range(
        barrier.view,
        slot,
    );
    if (catch range_fault = range_result) {
        TextureBarrierRejection rejection = {
            .field     = "TextureBarrier.view",
            .invariant = "view must select a valid texture subresource range without format reinterpretation",
        };
        return report_texture_barrier_rejection(
            state:        state,
            barrier:      barrier,
            slot:         slot,
            rejection:    rejection,
            public_fault: range_fault,
        )~;
    }
    vk::ImageSubresourceRange range = range_result;
    QueueCompletionState* completion = queue_completion_state(
        state,
        record.queue.id,
    )!;

    TextureBarrierRejection rejection = texture_state_rejection(
        texture_state: barrier.before,
        slot:          slot,
        flags:         completion.queue_flags,
        ray_tracing_pipelines: state.ray_tracing_pipelines_enabled,
        before:        true,
    );
    if (rejection.field != null) {
        return report_texture_barrier_rejection(
            state:     state,
            barrier:   barrier,
            slot:      slot,
            rejection: rejection,
        )~;
    }
    TextureStateScope before = texture_state_to_vk(
        state:         barrier.before,
        format:        slot.format,
        side:          TextureStateSide.BEFORE,
        paired_stages: barrier.after.stages,
    );

    rejection = texture_state_rejection(
        texture_state: barrier.after,
        slot:          slot,
        flags:         completion.queue_flags,
        ray_tracing_pipelines: state.ray_tracing_pipelines_enabled,
        before:        false,
    );
    if (rejection.field != null) {
        return report_texture_barrier_rejection(
            state:     state,
            barrier:   barrier,
            slot:      slot,
            rejection: rejection,
        )~;
    }
    TextureStateScope after = texture_state_to_vk(
        state:         barrier.after,
        format:        slot.format,
        side:          TextureStateSide.AFTER,
        paired_stages: barrier.before.stages,
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
    return { .slot = slot, .native = native };
}

fn LoweredTextureBarrier? validate_track_and_lower_texture_barrier(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureBarrier* barrier,
    ZString operation,
) {
    if (barrier == null) {
        TextureBarrierRejection rejection = {
            .field     = "TextureBarrier",
            .invariant = "barrier must be non-null",
        };
        return report_texture_barrier_rejection(
            state:     state,
            barrier:   barrier,
            slot:      null,
            rejection: rejection,
        )~;
    }

    TextureSlot*? slot_result = resolve_tracked_texture_command_reference(
        state,
        record,
        barrier.texture,
    );
    if (catch handle_fault = slot_result) {
        TextureBarrierRejection rejection = {
            .field     = "TextureBarrier.texture",
            .invariant = "must identify one live texture",
        };
        return report_texture_barrier_rejection(
            state:        state,
            barrier:      barrier,
            slot:         null,
            rejection:    rejection,
            public_fault: handle_fault,
        )~;
    }
    return validate_and_lower_resolved_texture_barrier(
        state:     state,
        record:    record,
        barrier:   barrier,
        slot:      slot_result,
        operation: operation,
    );
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

fn void? vk_cmd_barrier_with_ops(
    CommandRecord* commands,
    gpu::Barrier* barrier,
    BarrierEmitOps* ops,
) @private {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    ZString rejected_field = barrier_masks_rejected_field(barrier);
    ZString invariant;
    if (rejected_field != null) {
        if (rejected_field.str_view() == "Barrier") {
            invariant = "barrier must be non-null";
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
            state.acceleration_structures_enabled,
            state.ray_tracing_pipelines_enabled,
        );
        if (rejected_field != null) {
            invariant = "stage mask must be supported by the recording queue";
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

    vk::MemoryBarrier2 native_barrier = barrier_to_vk(
        barrier,
        state.generated_work.enabled,
        state.ray_queries_enabled,
    );
    emit_barrier(commands, &native_barrier, ops);
}

fn void? trusted_barrier_with_ops(
    CommandRecord* commands,
    gpu::Barrier* barrier,
    BarrierEmitOps* ops,
) {
    if (barrier == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    vk::MemoryBarrier2 native_barrier = barrier_to_vk(
        barrier,
        state.generated_work.enabled,
        state.ray_queries_enabled,
    );
    emit_barrier(commands, &native_barrier, ops);
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

fn void? trusted_barrier(
    CommandRecord* commands,
    gpu::Barrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    return trusted_barrier_with_ops(commands, barrier, &ops);
}

fn void? full_barrier(
    CommandRecord* commands,
    gpu::Barrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    return vk_cmd_barrier_with_ops(commands, barrier, &ops);
}

fn LoweredTextureBarrier? lower_texture_barrier_trusted(
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
    );
    TextureStateScope after = texture_state_to_vk(
        state:         barrier.after,
        format:        slot.format,
        side:          TextureStateSide.AFTER,
        paired_stages: barrier.before.stages,
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
    return { .slot = slot, .native = native };
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

fn void? trusted_texture_barrier_with_ops(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
    BarrierEmitOps* ops,
) {
    LoweredTextureBarrier lowered = lower_texture_barrier_trusted(
        commands,
        barrier,
    )!;
    emit_texture_barrier(commands, &lowered, ops);
}

fn void? full_texture_barrier_with_ops(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
    BarrierEmitOps* ops,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    LoweredTextureBarrier lowered = validate_track_and_lower_texture_barrier(
        state:     state,
        record:    record,
        barrier:   barrier,
        operation: "cmd_texture_barrier",
    )!;
    emit_texture_barrier(commands, &lowered, ops);
}

fn void? trusted_texture_barrier(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    return trusted_texture_barrier_with_ops(commands, barrier, &ops);
}

fn void? full_texture_barrier(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
) {
    BarrierEmitOps ops = barrier_emit_ops_real();
    return full_texture_barrier_with_ops(commands, barrier, &ops);
}

fn void? barrier_with_injected_ops(
    CommandRecord* commands,
    gpu::Barrier* barrier,
    BarrierEmitOps* ops,
) @if(env::TESTING) {
    if (commands.device_state.validation_policy.contract
            == gpu::ContractValidation.FULL) {
        return vk_cmd_barrier_with_ops(commands, barrier, ops);
    }
    return trusted_barrier_with_ops(commands, barrier, ops);
}

fn void? texture_barrier_with_injected_ops(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
    BarrierEmitOps* ops,
) @if(env::TESTING) {
    if (commands.device_state.validation_policy.contract
            == gpu::ContractValidation.FULL) {
        return full_texture_barrier_with_ops(commands, barrier, ops);
    }
    return trusted_texture_barrier_with_ops(commands, barrier, ops);
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
