module gpu::vk @private;

import gpu @public;
import vk;

alias TextureBarrierEmitHook @if(env::TESTING) = fn void (
    void*, vk::ImageMemoryBarrier2*
);

const uint TEXTURE_STATE_STAGE_BITS @private = 0x01FF;
const uint TEXTURE_ACCESS_BITS @private = 0x03;

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

struct TextureBarrierWorkCounters @if(env::TESTING) {
    ulong helper_calls;
    ulong handle_resolutions;
    ulong access_validations;
    ulong range_resolutions;
    ulong state_lowerings;
    ulong native_assemblies;
    ulong native_emissions;
}

fn bool texture_format_is_depth_stencil(gpu::Format format) @private {
    return format == gpu::Format.D32_FLOAT
        || format == gpu::Format.D24_UNORM_S8_UINT;
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

fn TextureStateScope texture_state_to_vk(
    gpu::TextureState state,
    gpu::Format format,
) {
    bool depth_stencil = texture_format_is_depth_stencil(format);
    switch (state.layout) {
        case gpu::TextureLayout.UNDEFINED:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_NONE,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = vk::ImageLayout.UNDEFINED,
            };
        case gpu::TextureLayout.TRANSFER_SOURCE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = vk::ImageLayout.TRANSFER_SRC_OPTIMAL,
            };
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = vk::ImageLayout.TRANSFER_DST_OPTIMAL,
            };
        case gpu::TextureLayout.SAMPLED:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = depth_stencil
                    ? vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            };
        case gpu::TextureLayout.STORAGE:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = vk::ImageLayout.GENERAL,
            };
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
            };
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            return {
                .stages = barrier_before_stages_to_vk(state.stages),
                .access = texture_access_to_vk(state.layout, state.access),
                .layout = vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
        case gpu::TextureLayout.PRESENT:
            return {
                .stages = vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .access = vk::AccessFlagBits2.ACCESS_2_NONE,
                .layout = vk::ImageLayout.PRESENT_SRC_KHR,
            };
        default:
            return {};
    }
}

fn ZString texture_state_field(bool before, ZString field) @private {
    if (field.str_view() == "layout") {
        return before
            ? "TextureBarrier.before.layout"
            : "TextureBarrier.after.layout";
    }
    if (field.str_view() == "stages") {
        return before
            ? "TextureBarrier.before.stages"
            : "TextureBarrier.after.stages";
    }
    return before
        ? "TextureBarrier.before.access"
        : "TextureBarrier.after.access";
}

fn bool texture_shader_stages_supported(
    gpu::StageMask stages,
    vk::QueueFlags flags,
) @private {
    bool graphics = queue_flags_include(flags, vk::QueueFlagBits.GRAPHICS_BIT);
    bool compute = queue_flags_include(flags, vk::QueueFlagBits.COMPUTE_BIT);
    if (stages.compute && !compute) return false;
    if ((stages.vertex_shader || stages.fragment_shader) && !graphics) {
        return false;
    }
    return true;
}

fn TextureBarrierRejection texture_state_rejection(
    gpu::TextureState texture_state,
    TextureSlot* slot,
    vk::QueueFlags flags,
    bool before,
) @private {
    if (!texture_layout_known(texture_state.layout)) {
        return {
            .field     = texture_state_field(before, "layout"),
            .invariant = "layout must be a known TextureLayout value",
        };
    }
    uint stages = (uint)texture_state.stages;
    uint access = (uint)texture_state.access;
    if ((stages & TEXTURE_STATE_STAGE_BITS) != stages
        || texture_state.stages.host
        || texture_state.stages.present
        || (texture_state.stages.all && stages != 1)) {
        return {
            .field     = texture_state_field(before, "stages"),
            .invariant = "stages must contain known non-host pipeline stages and all must be exclusive",
        };
    }
    if ((access & TEXTURE_ACCESS_BITS) != access) {
        return {
            .field     = texture_state_field(before, "access"),
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
        || texture_state.stages.fragment_shader;
    bool exact_read = access == 1;
    bool exact_write = access == 2;
    bool concrete_access = access != 0;
    bool all = texture_state.stages.all;
    bool depth_stencil = texture_format_is_depth_stencil(slot.format);

    switch (texture_state.layout) {
        case gpu::TextureLayout.UNDEFINED:
            if (!before) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "UNDEFINED is source-only",
                };
            }
            if (stages != 0) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "UNDEFINED requires empty stages",
                };
            }
            if (access != 0) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "UNDEFINED requires empty access",
                };
            }
        case gpu::TextureLayout.TRANSFER_SOURCE:
            if (!(all || (texture_state.stages.transfer && stages == 1u << 2))
                || !transfer) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "transfer source requires transfer or all stages on a transfer-capable queue",
                };
            }
            if (!exact_read) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "transfer source requires read-only access",
                };
            }
            if (!slot.usage.transfer_src) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "texture usage must include transfer_src",
                };
            }
        case gpu::TextureLayout.TRANSFER_DESTINATION:
            if (!(all || (texture_state.stages.transfer && stages == 1u << 2))
                || !transfer) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "transfer destination requires transfer or all stages on a transfer-capable queue",
                };
            }
            if (!exact_write) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "transfer destination requires write-only access",
                };
            }
            if (!slot.usage.transfer_dst) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "texture usage must include transfer_dst",
                };
            }
        case gpu::TextureLayout.SAMPLED:
            if (!(all || (shader_stages
                    && (stages & ~((1u << 3) | (1u << 4) | (1u << 5))) == 0))
                || !shader_queue
                || !texture_shader_stages_supported(texture_state.stages, flags)) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "sampled textures require supported shader stages or all",
                };
            }
            if (!exact_read) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "sampled textures require read-only access",
                };
            }
            if (!slot.usage.sampled) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "texture usage must include sampled",
                };
            }
        case gpu::TextureLayout.STORAGE:
            if (!(all || (shader_stages
                    && (stages & ~((1u << 3) | (1u << 4) | (1u << 5))) == 0))
                || !shader_queue
                || !texture_shader_stages_supported(texture_state.stages, flags)) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "storage textures require supported shader stages or all",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "storage textures require read, write, or read-write access",
                };
            }
            if (!slot.usage.storage) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "texture usage must include storage",
                };
            }
        case gpu::TextureLayout.COLOR_ATTACHMENT:
            if (!(all || (texture_state.stages.color_output
                    && stages == 1u << 6)) || !graphics) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "color attachment requires color_output or all on a graphics queue",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "color attachment requires read, write, or read-write access",
                };
            }
            if (depth_stencil || !slot.usage.color_attach) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "color attachment requires a non-depth color_attach texture",
                };
            }
        case gpu::TextureLayout.DEPTH_ATTACHMENT:
            if (!(all || (texture_state.stages.depth_output
                    && stages == 1u << 7)) || !graphics) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "depth attachment requires depth_output or all on a graphics queue",
                };
            }
            if (!concrete_access) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "depth attachment requires read, write, or read-write access",
                };
            }
            if (!depth_stencil || !slot.usage.depth_attach) {
                return {
                    .field     = texture_state_field(before, "layout"),
                    .invariant = "depth attachment requires a depth_attach depth texture",
                };
            }
        case gpu::TextureLayout.PRESENT:
            if (stages != 0) {
                return {
                    .field     = texture_state_field(before, "stages"),
                    .invariant = "PRESENT requires empty public stages",
                };
            }
            if (access != 0) {
                return {
                    .field     = texture_state_field(before, "access"),
                    .invariant = "PRESENT requires empty public access",
                };
            }
            if (!slot.swapchain_owned || depth_stencil || !graphics) {
                return {
                    .field     = texture_state_field(before, "layout"),
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

fn void note_texture_barrier_helper(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_helper_calls.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_texture_barrier_access_validation(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_access_validations.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_texture_barrier_range_resolution(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_range_resolutions.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_texture_barrier_state_lowering(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_state_lowerings.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_texture_barrier_native_assembly(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_native_assemblies.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_texture_barrier_native_emission(VkDeviceState* state) @private {
    $if env::TESTING:
        (void)state.texture_barrier_native_emissions.add(1, AtomicOrdering.RELAXED);
    $endif
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

fn LoweredTextureBarrier? validate_and_lower_texture_barrier(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureBarrier* barrier,
    ZString operation,
) {
    note_texture_barrier_helper(state);
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

    TextureSlot*? slot_result = resolve_texture_command_reference(
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
    TextureSlot* slot = slot_result;
    note_texture_barrier_access_validation(state);
    validate_texture_queue_access(
        state:       state,
        queue_roles: record.queue.roles,
        slot:        slot,
        handle:      barrier.texture,
        operation:   operation,
    )!;

    note_texture_barrier_range_resolution(state);
    vk::ImageSubresourceRange? range_result = resolve_texture_barrier_range(
        barrier.view,
        slot,
    );
    if (catch range_result) {
        TextureBarrierRejection rejection = {
            .field     = "TextureBarrier.view",
            .invariant = "view must select a valid texture subresource range without format reinterpretation",
        };
        return report_texture_barrier_rejection(
            state:     state,
            barrier:   barrier,
            slot:      slot,
            rejection: rejection,
        )~;
    }
    vk::ImageSubresourceRange range = range_result;
    QueueCompletionState* completion = queue_completion_state(
        state,
        record.queue.id,
    )!;

    note_texture_barrier_state_lowering(state);
    TextureBarrierRejection rejection = texture_state_rejection(
        texture_state: barrier.before,
        slot:          slot,
        flags:         completion.queue_flags,
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
        barrier.before,
        slot.format,
    );

    note_texture_barrier_state_lowering(state);
    rejection = texture_state_rejection(
        texture_state: barrier.after,
        slot:          slot,
        flags:         completion.queue_flags,
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
        barrier.after,
        slot.format,
    );

    note_texture_barrier_native_assembly(state);
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

fn void? vk_cmd_barrier(
    gpu::CommandEncoderState* commands,
    gpu::Barrier* barrier,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
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
    gpu::note_command_native_call();
}

fn void? vk_cmd_texture_barrier(
    gpu::CommandEncoderState* commands,
    gpu::TextureBarrier* barrier,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(
        state,
        record,
        reference_checkpoint,
    );
    LoweredTextureBarrier lowered = validate_and_lower_texture_barrier(
        state:     state,
        record:    record,
        barrier:   barrier,
        operation: "cmd_texture_barrier",
    )!;
    vk::DependencyInfo dependency = vk::dependency_info()
        .set_image_memory_barriers((&lowered.native)[:1]);
    note_texture_barrier_native_emission(state);
    $if env::TESTING:
        if (state.texture_barrier_emit_hook != null) {
            state.texture_barrier_emit_hook(
                state.texture_barrier_emit_user_data,
                &lowered.native,
            );
        } else {
            vk::cmd_pipeline_barrier2(record.command_buffer, &dependency);
        }
    $else
        vk::cmd_pipeline_barrier2(record.command_buffer, &dependency);
    $endif
    gpu::note_command_native_call();
}

fn TextureBarrierWorkCounters texture_barrier_work_counters(
    VkDeviceState* state,
) @if(env::TESTING) {
    return {
        .helper_calls = state.texture_barrier_helper_calls.load(
            AtomicOrdering.RELAXED,
        ),
        .handle_resolutions = state.textures.get_calls.load(
            AtomicOrdering.RELAXED,
        ),
        .access_validations = state.texture_barrier_access_validations.load(
            AtomicOrdering.RELAXED,
        ),
        .range_resolutions = state.texture_barrier_range_resolutions.load(
            AtomicOrdering.RELAXED,
        ),
        .state_lowerings = state.texture_barrier_state_lowerings.load(
            AtomicOrdering.RELAXED,
        ),
        .native_assemblies = state.texture_barrier_native_assemblies.load(
            AtomicOrdering.RELAXED,
        ),
        .native_emissions = state.texture_barrier_native_emissions.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

fn void reset_texture_barrier_work_counters(
    VkDeviceState* state,
) @if(env::TESTING) {
    state.texture_barrier_helper_calls.store(0, AtomicOrdering.RELAXED);
    state.textures.get_calls.store(0, AtomicOrdering.RELAXED);
    state.texture_barrier_access_validations.store(0, AtomicOrdering.RELAXED);
    state.texture_barrier_range_resolutions.store(0, AtomicOrdering.RELAXED);
    state.texture_barrier_state_lowerings.store(0, AtomicOrdering.RELAXED);
    state.texture_barrier_native_assemblies.store(0, AtomicOrdering.RELAXED);
    state.texture_barrier_native_emissions.store(0, AtomicOrdering.RELAXED);
}

fn void set_texture_barrier_emit_hook(
    VkDeviceState* state,
    TextureBarrierEmitHook hook,
    void* user_data,
) @if(env::TESTING) {
    state.texture_barrier_emit_hook = hook;
    state.texture_barrier_emit_user_data = user_data;
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
