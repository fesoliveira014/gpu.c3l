module gpu::vk @private;

import gpu @public;
import vk;
import std::io;

fn void? vk_cmd_buffer_barrier(
    gpu::CommandList* commands,
    gpu::BufferBarrier* barrier,
) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           &barrier.span,
        required:       0,
        operation:      "cmd_buffer_barrier",
        rejected_field: "BufferBarrier.span",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      &barrier.span,
        operation: "cmd_buffer_barrier",
    )!;

    vk::BufferMemoryBarrier2 bar = vk::buffer_memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard))
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_buffer(resolved.slot.buffer)
        .set_offset((vk::DeviceSize)resolved.offset)
        .set_size((vk::DeviceSize)resolved.size);

    vk::DependencyInfo dep = vk::dependency_info().set_buffer_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
}

fn void? vk_cmd_global_barrier(gpu::CommandList* commands, gpu::GlobalBarrier* barrier) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    vk::MemoryBarrier2 bar = vk::memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard));
    vk::DependencyInfo dep = vk::dependency_info().set_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
}

<* Record-time layout for `tex` on `list`: its own pending transition if it
   has recorded one, else the committed tracked layout. *>
fn gpu::TextureLayout effective_layout(CommandRecord* record, TextureSlot* slot, gpu::TextureHandle tex) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        if (record.pending_layouts[i].texture == tex) return record.pending_layouts[i].latest;
    }
    return texture_layout(slot);
}

<* Record or update one texture's pending transition, growing as needed. *>
fn void record_pending_layout(VkDeviceState* state, CommandRecord* record, gpu::TextureHandle tex, gpu::TextureLayout old_layout, gpu::TextureLayout new_layout) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        if (record.pending_layouts[i].texture == tex) {
            record.pending_layouts[i].latest = new_layout;
            return;
        }
    }
    ensure_pending_layout_capacity(state, record);
    record.pending_layouts[record.pending_layout_count] = {
        .texture = tex,
        .first_old = old_layout,
        .latest = new_layout,
    };
    record.pending_layout_count++;
}

<* Replay a submitted list's pending transitions onto tracked state, in
   record order; called under the queue mutex once queue_submit2 succeeds.
   Best-effort drift check: a mismatch against the layout seen at record time
   means a concurrent recorder raced this texture; only validation logs it,
   matching today's relaxed-atomic looseness. *>
fn void commit_pending_layouts(VkDeviceState* state, CommandRecord* record) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        PendingLayout* entry = &record.pending_layouts[i];
        if (try slot = state.textures.get(entry.texture)) {
            if (state.validation && texture_layout(slot) != entry.first_old) {
                io::eprintfn(
                    "gpu: texture layout drift at commit (texture index %d): expected %d, observed %d",
                    entry.texture.index, (int)entry.first_old, (int)texture_layout(slot),
                );
            }
            set_texture_layout(slot, entry.latest);
        }
    }
    record.pending_layout_count = 0;
}

<* Whole-image barrier; a barrier whose old_layout disagrees with the
   effective layout (this list's own pending transition, else the tracked
   layout) faults INVALID_RESOURCE_STATE. The transition is recorded as
   pending, not committed until the list submits. *>
fn void? vk_cmd_texture_barrier(gpu::CommandList* commands, gpu::TextureBarrier* barrier) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    TextureSlot* slot = state.textures.get(barrier.texture)!;
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        handle:    barrier.texture,
        operation: "cmd_texture_barrier",
    )!;
    if (barrier.old_layout != effective_layout(record, slot, barrier.texture)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_texture_barrier",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               barrier.texture.index,
            generation:          barrier.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureBarrier.old_layout",
            invariant:           "old_layout must match the command list effective layout",
            backend_text:        "texture barrier old layout disagrees with tracked layout",
        )~;
    }
    // Track the transition before emitting its GPU-visible barrier so the
    // backend record always describes everything in the command buffer.
    record_pending_layout(
        state:      state,
        record:     record,
        tex:        barrier.texture,
        old_layout: barrier.old_layout,
        new_layout: barrier.new_layout,
    );

    vk::ImageMemoryBarrier2 bar = vk::image_memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard))
        .set_old_layout(layout_to_vk(barrier.old_layout))
        .set_new_layout(layout_to_vk(barrier.new_layout))
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_image(slot.image)
        .set_subresource_range({
            .aspect_mask      = format_aspect(slot.format),
            .base_mip_level   = 0,
            .level_count      = slot.mip_levels,
            .base_array_layer = 0,
            .layer_count      = slot.array_layers,
        });

    vk::DependencyInfo dep = vk::dependency_info().set_image_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
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
    observe_queue_completion(state, completion, current);
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
    observe_queue_completion(state, completion, sequence);
    release_completed_submitted_commands(
        state,
        queue_id,
        sequence,
    );
}
