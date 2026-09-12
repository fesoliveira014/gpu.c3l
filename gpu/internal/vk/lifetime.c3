module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::env;
import std::core::mem::alloc;
import std::atomic::types;

alias BeforeCompletionRetirementFn @feat(TESTING) = fn void (void*);

const uint SUBMITTED_COMMAND_DRAIN_THRESHOLD = 64;

struct SparseBindReference {
    Atomic{uint}*       retained_counter;
    ulong               sequence;
    SparseBindReference* next;
}

struct CompletionRetirementOps {
    void* user_data @feat(TESTING);
    SemaphoreCounterQueryFn query;
    BeforeCompletionRetirementFn before_retirement @feat(TESTING);
}

fn void publish_pending_commands_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    CommandRecord*[] records,
    gpu::CompletionPoint point,
) {
    foreach (record : records) {
        activate_pending_acceleration_structure_builds(record);
        record.submitted_completion = point;
        record.next_submitted = null;
        if (completion.submitted_tail == null) {
            completion.submitted_head = record;
        } else {
            completion.submitted_tail.next_submitted = record;
        }
        completion.submitted_tail = record;
        completion.submitted_count++;
        CommandAllocatorSlot* slot = record_allocator_slot(state, record);
        // The live executable forces destroy to release its claim before this retirement-held pin proceeds.
        pin_command_allocator_slot(slot);
        slot.mutex.lock()!!;
        slot.live_executables--;
        slot.in_flight_buffers++;
        unlock_command_allocator(slot);
    }
}

fn void publish_sparse_bind_reference_locked(
    QueueCompletionState* completion,
    SparseBindReference* reference,
    gpu::CompletionPoint point,
) {
    reference.sequence = gpu::internal::completion_point_sequence(point);
    reference.next = null;
    if (completion.sparse_bind_tail == null) {
        completion.sparse_bind_head = reference;
    } else {
        completion.sparse_bind_tail.next = reference;
    }
    completion.sparse_bind_tail = reference;
    completion.sparse_bind_count++;
}

fn SparseBindReference* release_sparse_bind_reference_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    SparseBindReference* reference,
) {
    SparseBindReference* next = reference.next;
    reference.retained_counter.sub(1, AtomicOrdering.RELAXED);
    alloc::free(state.host_allocator, reference);
    completion.sparse_bind_count--;
    return next;
}

fn CommandRecord* release_pending_command_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    CommandRecord* record,
) {
    CommandRecord* next = record.next_submitted;
    CommandAllocatorSlot* slot = record_allocator_slot(state, record);
    // The in-flight buffer forces destroy to release its claim before this retirement-held pin proceeds.
    pin_command_allocator_slot(slot);
    slot.mutex.lock()!!;
    record.state = gpu::internal::CommandState.INACTIVE;
    complete_pending_acceleration_structure_builds(record);
    CommandScratch* scratch = record.scratch;

    recycle_generated_preprocess_indices_locked(
        state,
        slot,
        scratch.generated_preprocess_indices,
        scratch.generated_preprocess_count,
    );
    return_allocator_buffer_locked(slot, record.allocator_buffer);
    slot.in_flight_buffers--;
    unlock_command_allocator(slot);
    gpu::internal::CommandListHandle identity = record.identity;
    gpu::Device owner = record.device;
    if (record.init_textures.len > 0) {
        release_pending_texture_inits(state, record.init_textures, false);
        record.init_textures = {};
    }
    record.submitted_completion = {};
    record.next_submitted = null;
    completion.submitted_count--;
    state.command_mutex.lock()!!;
    (void)state.commands.free(identity);
    state.command_mutex.unlock();
    gpu::internal::release_command_record_ownership(owner);
    return next;
}

<* Retires no further than the published prefix. Requires retirement_mutex. *>
fn void retire_queue_through_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
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

    while (completion.submitted_head != null) {
        CommandRecord* record = completion.submitted_head;
        if (gpu::internal::completion_point_sequence(
            record.submitted_completion,
        ) > target) {
            break;
        }
        completion.submitted_head = release_pending_command_locked(
            state,
            completion,
            record,
        );
    }
    if (completion.submitted_head == null) {
        completion.submitted_tail = null;
    }
    while (completion.sparse_bind_head != null
        && completion.sparse_bind_head.sequence <= target) {
        completion.sparse_bind_head =
            release_sparse_bind_reference_locked(
                state,
                completion,
                completion.sparse_bind_head,
            );
    }
    if (completion.sparse_bind_head == null) {
        completion.sparse_bind_tail = null;
    }
    completion.retired_sequence.store(target, AtomicOrdering.RELEASE);
}

fn void retire_observed_queue(
    VkDeviceState* state,
    QueueCompletionState* completion,
    ulong completed,
) @private {
    completion.retirement_mutex.lock()!!;
    defer completion.retirement_mutex.unlock();
    retire_queue_through_locked(state, completion, completed);
}

fn bool? query_pending_queue(
    VkDeviceState* state,
    QueueCompletionState* completion,
    SemaphoreCounterQueryFn query_fn,
    ulong* completed,
) @private {
    completion.retirement_mutex.lock()!!;
    defer completion.retirement_mutex.unlock();
    ulong published = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong retired = completion.retired_sequence.load(
        AtomicOrdering.RELAXED,
    );
    if (completion.submitted_head == null && published <= retired) return false;

    check_device_result(
        state,
        query_fn(
            state.device,
            completion.timeline,
            completed,
        ),
    )!;
    return true;
}

fn void? query_pending_queues_before_retirement(
    VkDeviceState* state,
    bool has_observed,
    uint observed_queue_id,
    ulong observed_completed,
    SemaphoreCounterQueryFn query_fn,
) @private {
    bool[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] queried = {};
    ulong[gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES] completed = {};
    if (has_observed) {
        queried[observed_queue_id] = true;
        completed[observed_queue_id] = observed_completed;
    }

    foreach (queue_id, &completion : state.queue_completions) {
        if (has_observed && (uint)queue_id == observed_queue_id) continue;
        queried[queue_id] = query_pending_queue(
            state,
            completion,
            query_fn,
            &completed[queue_id],
        )!;
    }
    foreach (queue_id, &completion : state.queue_completions) {
        if (!queried[queue_id]) continue;
        retire_observed_queue(
            state,
            completion,
            completed[queue_id],
        );
    }
}

<* Every first successful completion observation drains all pending queue
   identities before publishing retired prefixes; cached hits rely on this. *>
fn void? retire_observed_completion_and_drain_with_ops(
    VkDeviceState* state,
    uint observed_queue_id,
    ulong observed_completed,
    CompletionRetirementOps* ops,
) {
    $if env::TESTING:
        if (ops.before_retirement != null) {
            ops.before_retirement(ops.user_data);
        }
    $endif
    (void)queue_completion_state(state, observed_queue_id)!;
    query_pending_queues_before_retirement(
        state:              state,
        has_observed:       true,
        observed_queue_id:  observed_queue_id,
        observed_completed: observed_completed,
        query_fn:           ops.query,
    )!;
}

fn void? retire_observed_completion_and_drain_with_query(
    VkDeviceState* state,
    uint observed_queue_id,
    ulong observed_completed,
    SemaphoreCounterQueryFn query_fn,
) {
    CompletionRetirementOps ops = { .query = query_fn };
    retire_observed_completion_and_drain_with_ops(
        state:              state,
        observed_queue_id:  observed_queue_id,
        observed_completed: observed_completed,
        ops:                &ops,
    )!;
}

<* Observe the queue's timeline and retire everything it has completed. *>
fn void? retire_queue_latest_with_query(
    VkDeviceState* state,
    uint queue_id,
    SemaphoreCounterQueryFn query_fn,
) {
    QueueCompletionState* completion = queue_completion_state(state, queue_id)!;
    ulong current;
    check_backend_result(
        state:     state,
        operation: "begin_commands",
        result:    query_fn(state.device, completion.timeline, &current),
    )!;
    retire_observed_completion_and_drain_with_query(
        state:              state,
        observed_queue_id:  queue_id,
        observed_completed: current,
        query_fn:           query_fn,
    )!;
}

fn void? drain_completed_submitted_commands_with_query(
    VkDeviceState* state,
    SemaphoreCounterQueryFn query_fn,
) {
    query_pending_queues_before_retirement(
        state:              state,
        has_observed:       false,
        observed_queue_id:  0,
        observed_completed: 0,
        query_fn:           query_fn,
    )!;
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
    foreach (&completion : state.queue_completions) {
        drain_queue_if_needed_with_query(state, completion, query_fn)!;
    }
}

fn void? drain_queue_if_needed_with_query(
    VkDeviceState* state,
    QueueCompletionState* completion,
    SemaphoreCounterQueryFn query_fn,
) {
    completion.retirement_mutex.lock()!!;
    defer completion.retirement_mutex.unlock();
    if (completion.submitted_count + completion.sparse_bind_count
        < SUBMITTED_COMMAND_DRAIN_THRESHOLD) return;

    ulong completed;
    check_device_result(
        state,
        query_fn(
            state.device,
            completion.timeline,
            &completed,
        ),
    )!;
    retire_queue_through_locked(state, completion, completed);
}

fn void? drain_submitted_commands_if_needed(VkDeviceState* state) {
    return drain_submitted_commands_if_needed_with_query(
        state,
        &real_semaphore_counter_query,
    );
}

fn void destroy_submitted_commands(VkDeviceState* state) {
    foreach (&completion : state.queue_completions) {
        bool locked = completion.retirement_mutex.is_initialized();
        if (locked) completion.retirement_mutex.lock()!!;
        while (completion.submitted_head != null) {
            completion.submitted_head = release_pending_command_locked(
                state,
                completion,
                completion.submitted_head,
            );
        }
        completion.submitted_tail = null;
        completion.submitted_count = 0;
        while (completion.sparse_bind_head != null) {
            completion.sparse_bind_head =
                release_sparse_bind_reference_locked(
                    state,
                    completion,
                    completion.sparse_bind_head,
                );
        }
        completion.sparse_bind_tail = null;
        completion.sparse_bind_count = 0;
        if (locked) completion.retirement_mutex.unlock();
    }
}
