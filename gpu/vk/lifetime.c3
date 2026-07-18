module gpu::vk @private;

import gpu @public;
import vk;
import std::atomic::types;

const usz COMMAND_REFERENCE_INITIAL_CAPACITY = 16;
const usz SUBMITTED_REFERENCE_INITIAL_CAPACITY = 16;
const uint SUBMITTED_REFERENCE_DRAIN_THRESHOLD = 64;

struct CommandResourceReference {
    ulong owner;
    uint  index;
    uint  generation;
}

struct SubmittedReferenceBatch {
    gpu::CompletionPoint       completion;
    CommandResourceReference[] references;
    uint                       reference_count;
}

fn bool same_resource_reference(
    CommandResourceReference* left,
    CommandResourceReference* right,
) {
    return left.owner == right.owner
        && left.index == right.index
        && left.generation == right.generation;
}

fn void? retain_validation_reference(
    VkDeviceState* state,
    CommandResourceReference reference,
) {
    if (reference.owner == state.buffers.owner) {
        BufferHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.buffers.get(handle)!.validation_references.add(
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
        state.allocations.get(allocation)!.validation_references.add(
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
        state.textures.get(texture)!.validation_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    if (reference.owner == state.persistent_arena.allocations.owner) {
        PersistentAllocationHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.persistent_arena.allocations.get(handle)!
            .validation_references.add(
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
        state.pipelines.get(pipeline)!.validation_references.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return;
    }
    return gpu::INVALID_HANDLE~;
}

fn void release_validation_reference(
    VkDeviceState* state,
    CommandResourceReference reference,
) {
    if (reference.owner == state.buffers.owner) {
        BufferHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.buffers.get(handle)!!.validation_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
    } else if (reference.owner == state.allocations.owner) {
        gpu::GpuAllocation allocation = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.allocations.get(allocation)!!.validation_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
    } else if (reference.owner == state.textures.owner) {
        gpu::TextureHandle texture = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.textures.get(texture)!!.validation_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
    } else if (reference.owner == state.pipelines.owner) {
        gpu::PipelineHandle pipeline = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.pipelines.get(pipeline)!!.validation_references.sub(
            1,
            AtomicOrdering.RELAXED,
        );
    } else if (reference.owner == state.persistent_arena.allocations.owner) {
        PersistentAllocationHandle handle = {
            .owner = reference.owner, .index = reference.index,
            .generation = reference.generation,
        };
        state.persistent_arena.allocations.get(handle)!!
            .validation_references.sub(
                1,
                AtomicOrdering.RELAXED,
            );
    }
}

fn void release_validation_references(
    VkDeviceState* state,
    CommandResourceReference[] references,
    uint count,
) {
    for (uint i = 0; i < count; i++) {
        release_validation_reference(state, references[i]);
    }
}

fn void ensure_submitted_reference_capacity(
    VkDeviceState* state,
    uint additional,
) {
    usz required = (usz)state.submitted_reference_count
        + (usz)state.submitted_reference_reserved
        + (usz)additional;
    if (required <= state.submitted_references.len) return;
    usz capacity = state.submitted_references.len == 0
        ? SUBMITTED_REFERENCE_INITIAL_CAPACITY
        : state.submitted_references.len * 2;
    while (capacity < required) {
        capacity *= 2;
    }
    SubmittedReferenceBatch[] grown = alloc::new_array(
        state.host_allocator,
        SubmittedReferenceBatch,
        (sz)capacity,
    );
    for (uint i = 0; i < state.submitted_reference_count; i++) {
        grown[i] = state.submitted_references[i];
    }
    if (state.submitted_references.ptr != null) {
        alloc::free(state.host_allocator, state.submitted_references.ptr);
    }
    state.submitted_references = grown;
}

fn uint prepare_submitted_references(
    VkDeviceState* state,
    CommandRecord*[] records,
) {
    if (!state.validation) return 0;
    uint additional;
    foreach (record : records) {
        if (record.reference_count > 0) additional++;
    }
    if (additional == 0) return 0;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    ensure_submitted_reference_capacity(state, additional);
    state.submitted_reference_reserved += additional;
    return additional;
}

fn void cancel_submitted_reference_reservation(
    VkDeviceState* state,
    uint reservation,
) {
    if (reservation == 0) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.submitted_reference_reserved -= reservation;
}

fn void publish_submitted_references(
    VkDeviceState* state,
    CommandRecord*[] records,
    gpu::CompletionPoint completion,
    uint reservation,
) {
    if (!state.validation) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.submitted_reference_reserved -= reservation;
    foreach (record : records) {
        if (record.reference_count == 0) continue;
        state.submitted_references[state.submitted_reference_count++] = {
            .completion      = completion,
            .references      = record.references,
            .reference_count = record.reference_count,
        };
        record.references = {};
        record.reference_count = 0;
    }
}

fn void release_submitted_reference_batch(
    VkDeviceState* state,
    uint index,
) {
    SubmittedReferenceBatch* batch = &state.submitted_references[index];
    release_validation_references(
        state,
        batch.references,
        batch.reference_count,
    );
    alloc::free(state.host_allocator, batch.references.ptr);
    state.submitted_reference_count--;
    *batch = state.submitted_references[state.submitted_reference_count];
    state.submitted_references[state.submitted_reference_count] = {};
}

fn void release_completed_submission_references_locked(
    VkDeviceState* state,
    uint queue_id,
    ulong completed_sequence,
) {
    uint i;
    while (i < state.submitted_reference_count) {
        SubmittedReferenceBatch* batch = &state.submitted_references[i];
        if (gpu::completion_point_queue_id(batch.completion) != queue_id
            || gpu::completion_point_sequence(batch.completion)
                > completed_sequence) {
            i++;
            continue;
        }
        release_submitted_reference_batch(state, i);
    }
}

fn void release_completed_submission_references(
    VkDeviceState* state,
    uint queue_id,
    ulong completed_sequence,
) {
    if (!state.validation) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    release_completed_submission_references_locked(
        state,
        queue_id,
        completed_sequence,
    );
}

fn void? drain_completed_submission_references_with_query(
    VkDeviceState* state,
    FrameCounterQueryFn query_fn,
) {
    if (!state.validation) return;
    bool[gpu::MAX_SELECTED_QUEUE_IDENTITIES] queried;
    ulong[gpu::MAX_SELECTED_QUEUE_IDENTITIES] completed;
    uint i;
    while (i < state.submitted_reference_count) {
        SubmittedReferenceBatch* batch = &state.submitted_references[i];
        uint queue_id = gpu::completion_point_queue_id(batch.completion);
        QueueCompletionState* completion = queue_completion_state(state, queue_id)!;
        if (!queried[queue_id]) {
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
        if (completed[queue_id]
            < gpu::completion_point_sequence(batch.completion)) {
            i++;
            continue;
        }
        release_submitted_reference_batch(state, i);
    }
}

fn void? drain_completed_submission_references(VkDeviceState* state) {
    return drain_completed_submission_references_with_query(
        state,
        &real_frame_counter_query,
    );
}

fn void? drain_submission_references_if_needed(VkDeviceState* state) {
    if (!state.validation) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (state.submitted_reference_count
        < SUBMITTED_REFERENCE_DRAIN_THRESHOLD) return;
    drain_completed_submission_references(state)!;
}

fn void destroy_submitted_references(VkDeviceState* state) {
    for (uint i = 0; i < state.submitted_reference_count; i++) {
        SubmittedReferenceBatch* batch = &state.submitted_references[i];
        release_validation_references(
            state,
            batch.references,
            batch.reference_count,
        );
        alloc::free(state.host_allocator, batch.references.ptr);
    }
    if (state.submitted_references.ptr != null) {
        alloc::free(state.host_allocator, state.submitted_references.ptr);
    }
    state.submitted_references = {};
    state.submitted_reference_count = 0;
    state.submitted_reference_reserved = 0;
}
