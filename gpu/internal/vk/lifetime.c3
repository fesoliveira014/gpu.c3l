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
    Atomic{uint}* retained_counter;
}

fn bool same_resource_reference(
    CommandResourceReference* left,
    CommandResourceReference* right,
) {
    return left.owner == right.owner
        && left.index == right.index
        && left.generation == right.generation;
}

fn Atomic{uint}*? tracked_command_reference_counter(
    VkDeviceState* state,
    CommandResourceReference reference,
) {
    if (reference.owner == state.buffers.owner) {
        BufferHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        BufferSlot* slot = state.buffers.get(handle)!;
        return &slot.tracked_command_references;
    }
    if (reference.owner == state.allocations.owner) {
        gpu::GpuAllocation allocation = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        AllocationSlot* slot = state.allocations.get(allocation)!;
        return &slot.tracked_command_references;
    }
    if (reference.owner == state.textures.owner) {
        gpu::TextureHandle texture = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        TextureSlot* slot = state.textures.get(texture)!;
        return &slot.tracked_command_references;
    }
    if (reference.owner == state.attachment_views.owner) {
        gpu::AttachmentViewHandle view = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        AttachmentViewSlot* slot = state.attachment_views.get(view)!;
        return &slot.tracked_command_references;
    }
    if (reference.owner == state.pipelines.owner) {
        gpu::PipelineHandle pipeline = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        PipelineSlot* slot = state.pipelines.get(pipeline)!;
        return &slot.tracked_command_references;
    }
    return gpu::INVALID_HANDLE~;
}

fn void? retain_tracked_command_reference(
    VkDeviceState* state,
    CommandResourceReference* reference,
) {
    reference.retained_counter =
        tracked_command_reference_counter(state, *reference)!;
    reference.retained_counter.add(1, AtomicOrdering.RELAXED);
}

<* @require reference.retained_counter != null : "retained counter" *>
fn void release_tracked_command_reference(
    CommandResourceReference reference,
) {
    reference.retained_counter.sub(1, AtomicOrdering.RELAXED);
}

fn void release_tracked_command_references(
    CommandResourceReference[] references,
    uint count,
) {
    for (uint i = 0; i < count; i++) {
        release_tracked_command_reference(references[i]);
    }
}

fn void publish_pending_commands_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    CommandRecord*[] records,
    gpu::CompletionPoint point,
) {
    foreach (record : records) {
        record.submitted_completion = point;
        record.next_submitted = null;
        if (completion.submitted_tail == null) {
            completion.submitted_head = record;
        } else {
            completion.submitted_tail.next_submitted = record;
        }
        completion.submitted_tail = record;
        completion.submitted_count++;
        CommandAllocatorSlot* slot = state.command_allocators.get(
            record.allocator,
        )!!;
        slot.mutex.lock()!!;
        slot.live_executables--;
        slot.in_flight_buffers++;
        slot.mutex.unlock();
    }
}

fn CommandRecord* release_pending_command_locked(
    VkDeviceState* state,
    QueueCompletionState* completion,
    CommandRecord* record,
) {
    CommandRecord* next = record.next_submitted;
    CommandAllocatorSlot* slot = state.command_allocators.get(
        record.allocator,
    )!!;
    slot.mutex.lock()!!;
    record.state = gpu::internal::CommandState.INACTIVE;
    // Reference gates plus retirement/destroy serialization make resource_mutex unnecessary.
    CommandScratch* scratch = record.scratch;
    clear_command_scratch_references(scratch);
    recycle_generated_preprocess_indices_locked(
        state,
        slot,
        scratch.generated_preprocess_indices,
        scratch.generated_preprocess_count,
    );
    return_allocator_buffer_locked(slot, record.allocator_buffer);
    slot.in_flight_buffers--;
    slot.mutex.unlock();
    gpu::internal::CommandListHandle identity = record.identity;
    gpu::Device owner = record.device;
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
    if (completion.submitted_head == null) return false;

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
fn void? retire_observed_completion_and_drain_with_query(
    VkDeviceState* state,
    uint observed_queue_id,
    ulong observed_completed,
    SemaphoreCounterQueryFn query_fn,
) {
    $if env::TESTING:
        if (state.completion_retirement_lock_hook != null) {
            state.completion_retirement_lock_hook(
                state.completion_retirement_lock_user_data,
            );
        }
    $endif

    (void)queue_completion_state(state, observed_queue_id)!;
    query_pending_queues_before_retirement(
        state:              state,
        has_observed:       true,
        observed_queue_id:  observed_queue_id,
        observed_completed: observed_completed,
        query_fn:           query_fn,
    )!;
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
    if (completion.submitted_count < SUBMITTED_COMMAND_DRAIN_THRESHOLD) return;

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
        if (locked) completion.retirement_mutex.unlock();
    }
}
