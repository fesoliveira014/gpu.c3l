module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::env;
import std::atomic::types;

alias CompletionRetirementLockHook @if(env::TESTING) = fn void (void*);

const uint SUBMITTED_COMMAND_DRAIN_THRESHOLD = 64;

struct CommandResourceReference {
    ulong         owner;
    uint          index;
    uint          generation;
    Atomic{uint}* tracked_command_references;
}

struct SubmittedCommandBatch {
    gpu::CompletionPoint          completion;
    gpu::internal::CommandRecord* record;
}

struct CommandBufferStats {
    ulong allocated;
    ulong freed;
    uint  pending;
}

fn bool same_resource_reference(
    CommandResourceReference* left,
    CommandResourceReference* right,
) {
    return left.owner == right.owner
        && left.index == right.index
        && left.generation == right.generation;
}

fn void? retain_tracked_command_reference(
    VkDeviceState* state,
    CommandResourceReference reference,
) {
    if (reference.owner == state.buffers.owner) {
        BufferHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.buffers.get(handle)!.tracked_command_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    if (reference.owner == state.allocations.owner) {
        gpu::GpuAllocation allocation = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.allocations.get(allocation)!.tracked_command_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    if (reference.owner == state.textures.owner) {
        gpu::TextureHandle texture = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.textures.get(texture)!.tracked_command_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    if (reference.owner == state.attachment_views.owner) {
        gpu::AttachmentViewHandle view = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.attachment_views.get(view)!.tracked_command_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    if (reference.owner == state.pipelines.owner) {
        gpu::PipelineHandle pipeline = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.pipelines.get(pipeline)!.tracked_command_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    return gpu::INVALID_HANDLE~;
}

fn void release_tracked_command_reference(
    VkDeviceState* state,
    CommandResourceReference reference,
) {
    if (reference.tracked_command_references != null) {
        reference.tracked_command_references.sub(1, AtomicOrdering.RELAXED);
        note_command_reference_release(state);
        note_command_reference_work_release(state);
        return;
    }
    if (reference.owner == state.buffers.owner) {
        BufferHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.buffers.get(handle)!!.tracked_command_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
        note_command_reference_release(state);
        note_command_reference_work_release(state);
    } else if (reference.owner == state.allocations.owner) {
        gpu::GpuAllocation allocation = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.allocations.get(allocation)!!.tracked_command_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
        note_command_reference_release(state);
        note_command_reference_work_release(state);
    } else if (reference.owner == state.textures.owner) {
        gpu::TextureHandle texture = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.textures.get(texture)!!.tracked_command_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
        note_command_reference_release(state);
        note_command_reference_work_release(state);
    } else if (reference.owner == state.attachment_views.owner) {
        gpu::AttachmentViewHandle view = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.attachment_views.get(view)!!.tracked_command_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
        note_command_reference_release(state);
        note_command_reference_work_release(state);
    } else if (reference.owner == state.pipelines.owner) {
        gpu::PipelineHandle pipeline = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.pipelines.get(pipeline)!!.tracked_command_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
        note_command_reference_release(state);
        note_command_reference_work_release(state);
    }
}

fn void release_tracked_command_references(
    VkDeviceState* state,
    CommandResourceReference[] references,
    uint count,
) {
    for (uint i = 0; i < count; i++) {
        release_tracked_command_reference(state, references[i]);
    }
}

fn void? ensure_submitted_command_capacity(
    VkDeviceState* state,
    uint additional,
) {
    usz required = (usz)state.submitted_command_count
        + (usz)state.submitted_command_reserved
        + (usz)additional;
    if (required <= state.submitted_commands.len) return;
    return gpu::DEVICE_BUSY~;
}

fn uint? prepare_submitted_commands(
    VkDeviceState* state,
    gpu::internal::CommandRecord*[] records,
) {
    uint additional = (uint)records.len;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    ensure_submitted_command_capacity(state, additional)!;
    state.submitted_command_reserved += additional;
    return additional;
}

fn void cancel_submitted_command_reservation(
    VkDeviceState* state,
    uint reservation,
) {
    if (reservation == 0) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.submitted_command_reserved -= reservation;
}

fn void publish_submitted_commands(
    VkDeviceState* state,
    gpu::internal::CommandRecord*[] records,
    gpu::CompletionPoint completion,
    uint reservation,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.submitted_command_reserved -= reservation;
    foreach (record : records) {
        state.submitted_commands[state.submitted_command_count++] = {
            .completion = completion,
            .record     = record,
        };
        VkCommandRecord* backend = backend_command(record);
        CommandAllocatorSlot* slot = state.command_allocators.get(
            backend.allocator,
        )!!;
        slot.mutex.lock()!!;
        slot.live_executables--;
        slot.in_flight_buffers++;
        slot.mutex.unlock();
    }
}

fn void release_submitted_command_batch(
    VkDeviceState* state,
    uint index,
) {
    SubmittedCommandBatch* batch = &state.submitted_commands[index];
    gpu::internal::CommandRecord* record = batch.record;
    VkCommandRecord* backend = backend_command(record);
    CommandAllocatorSlot* slot = state.command_allocators.get(
        backend.allocator,
    )!!;
    slot.mutex.lock()!!;
    defer slot.mutex.unlock();
    record.state = gpu::internal::CommandState.INACTIVE;
    CommandScratch* scratch = backend.scratch;
    clear_command_scratch_references(state, scratch);
    recycle_generated_preprocess_indices_locked(
        state,
        slot,
        scratch.generated_preprocess_indices,
        scratch.generated_preprocess_count,
    );
    return_allocator_buffer_locked(slot, backend.allocator_buffer);
    slot.in_flight_buffers--;
    gpu::CommandListHandle identity = record.identity;
    gpu::internal::CommandRecord ownership = *record;
    state.submitted_command_count--;
    *batch = state.submitted_commands[state.submitted_command_count];
    state.submitted_commands[state.submitted_command_count] = {};
    state.command_mutex.lock()!!;
    (void)state.commands.free(identity);
    state.command_mutex.unlock();
    gpu::internal::release_command_record_ownership(&ownership);
}

fn void release_completed_submitted_commands_locked(
    VkDeviceState* state,
    uint queue_id,
    ulong completed_sequence,
) {
    uint i;
    while (i < state.submitted_command_count) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
        if (gpu::internal::completion_point_queue_id(batch.completion) != queue_id
            || gpu::internal::completion_point_sequence(batch.completion)
                > completed_sequence) {
            i++;
            continue;
        }
        release_submitted_command_batch(state, i);
    }
}

fn void note_completion_counter_query(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_native_counter_queries.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_completion_wait_call(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_native_wait_calls.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_completion_retirement_pass(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_retirement_passes.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_completion_retirement_lock_acquisition(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_retirement_lock_acquisitions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_completion_cached_poll_hit(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_cached_poll_hits.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_completion_cached_wait_hit(VkDeviceState* state) {
    $if env::TESTING || $feature(COMPLETION_RETIREMENT_STATS):
        (void)state.completion_cached_wait_hits.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

<* Retires through min(native_completed, published_sequence): native progress
   never crosses a submission publication gap. Requires `resource_mutex`. *>
fn void retire_queue_through_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    uint queue_id,
    ulong native_completed,
) {
    ulong published = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong target = native_completed < published
        ? native_completed
        : published;
    ulong retired = completion.retired_sequence.load(AtomicOrdering.RELAXED);
    if (target <= retired) return;

    note_completion_retirement_pass(state);
    release_completed_submitted_commands_locked(state, queue_id, target);
    completion.retired_sequence.store(target, AtomicOrdering.RELEASE);
}

<* Every first successful completion observation drains all pending queue
   identities before publishing retired prefixes; cached hits rely on this. *>
fn void? retire_observed_completion_and_drain_with_query(
    VkDeviceState* state,
    uint observed_queue_id,
    ulong observed_completed,
    SemaphoreCounterQueryFn query_fn,
) {
    bool[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] queried = {};
    ulong[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] completed = {};
    queried[observed_queue_id] = true;
    completed[observed_queue_id] = observed_completed;

    $if env::TESTING:
        if (state.completion_retirement_lock_hook != null) {
            state.completion_retirement_lock_hook(
                state.completion_retirement_lock_user_data,
            );
        }
    $endif

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    note_completion_retirement_lock_acquisition(state);
    for (uint i = 0; i < state.submitted_command_count; i++) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
        uint queue_id = gpu::internal::completion_point_queue_id(batch.completion);
        if (queried[queue_id]) continue;
        QueueCompletionState* completion = queue_completion_state(
            state,
            queue_id,
        )!;
        note_completion_counter_query(state);
        check_device_result(
            state,
            query_fn(
                state.device,
                completion.timeline,
                &completed[queue_id],
            ),
        )!;
        queried[queue_id] = true;
    }
    for (uint queue_id = 0;
        queue_id < gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES;
        queue_id++) {
        if (!queried[queue_id]) continue;
        QueueCompletionState* completion = queue_completion_state(
            state,
            queue_id,
        )!;
        retire_queue_through_locked(
            state:            state,
            completion:       completion,
            queue_id:         queue_id,
            native_completed: completed[queue_id],
        );
    }
}

fn void set_completion_retirement_lock_hook(
    VkDeviceState* state,
    CompletionRetirementLockHook hook,
    void* user_data,
) @if(env::TESTING) {
    state.completion_retirement_lock_hook = hook;
    state.completion_retirement_lock_user_data = user_data;
}

fn void? drain_completed_submitted_commands_with_query(
    VkDeviceState* state,
    SemaphoreCounterQueryFn query_fn,
) {
    bool[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] queried = {};
    ulong[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] completed = {};
    for (uint i = 0; i < state.submitted_command_count; i++) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
        uint queue_id = gpu::internal::completion_point_queue_id(batch.completion);
        QueueCompletionState* completion = queue_completion_state(state, queue_id)!;
        if (!queried[queue_id]) {
            note_completion_counter_query(state);
            check_device_result(
                state,
                query_fn(
                    state.device,
                    completion.timeline,
                    &completed[queue_id],
                ),
            )!;
            queried[queue_id] = true;
        }
    }
    for (uint queue_id = 0;
        queue_id < gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES;
        queue_id++) {
        if (!queried[queue_id]) continue;
        QueueCompletionState* completion = queue_completion_state(
            state,
            queue_id,
        )!;
        retire_queue_through_locked(
            state:            state,
            completion:       completion,
            queue_id:         queue_id,
            native_completed: completed[queue_id],
        );
    }
}

fn void? drain_completed_submitted_commands(VkDeviceState* state) {
    return drain_completed_submitted_commands_with_query(
        state,
        &real_semaphore_counter_query,
    );
}

fn void? drain_submitted_commands_if_needed_with_query(
    VkDeviceState* state,
    SemaphoreCounterQueryFn query_fn,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (state.submitted_command_count
        < SUBMITTED_COMMAND_DRAIN_THRESHOLD) return;
    drain_completed_submitted_commands_with_query(state, query_fn)!;
}

fn void? drain_submitted_commands_if_needed(VkDeviceState* state) {
    return drain_submitted_commands_if_needed_with_query(
        state,
        &real_semaphore_counter_query,
    );
}

fn CompletionWorkCounters completion_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(COMPLETION_RETIREMENT_STATS)) {
    return {
        .native_counter_queries = state.completion_native_counter_queries.load(
            AtomicOrdering.RELAXED,
        ),
        .native_wait_calls = state.completion_native_wait_calls.load(
            AtomicOrdering.RELAXED,
        ),
        .retirement_passes = state.completion_retirement_passes.load(
            AtomicOrdering.RELAXED,
        ),
        .retirement_lock_acquisitions =
            state.completion_retirement_lock_acquisitions.load(
                AtomicOrdering.RELAXED,
            ),
        .cached_poll_hits = state.completion_cached_poll_hits.load(
            AtomicOrdering.RELAXED,
        ),
        .cached_wait_hits = state.completion_cached_wait_hits.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

fn void reset_completion_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(COMPLETION_RETIREMENT_STATS)) {
    state.completion_native_counter_queries.store(0, AtomicOrdering.RELAXED);
    state.completion_native_wait_calls.store(0, AtomicOrdering.RELAXED);
    state.completion_retirement_passes.store(0, AtomicOrdering.RELAXED);
    state.completion_retirement_lock_acquisitions.store(
        0,
        AtomicOrdering.RELAXED,
    );
    state.completion_cached_poll_hits.store(0, AtomicOrdering.RELAXED);
    state.completion_cached_wait_hits.store(0, AtomicOrdering.RELAXED);
}

fn void destroy_submitted_commands(VkDeviceState* state) {
    while (state.submitted_command_count > 0) {
        release_submitted_command_batch(state, 0);
    }
    if (state.submitted_commands.ptr != null) {
        mem::free(state.submitted_commands.ptr);
    }
    state.submitted_commands = {};
    state.submitted_command_count = 0;
    state.submitted_command_reserved = 0;
}

fn CommandBufferStats command_buffer_stats(VkDeviceState* state) {
    state.resource_mutex.lock()!!;
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    defer state.resource_mutex.unlock();
    return {
        .allocated = state.command_buffers_allocated,
        .freed     = state.command_buffers_freed,
        .pending   = state.submitted_command_count,
    };
}
