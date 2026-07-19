module gpu::vk @private;

import gpu @public;
import vk;
import std::atomic::types;

const usz COMMAND_REFERENCE_INITIAL_CAPACITY = 16;
const usz SUBMITTED_COMMAND_INITIAL_CAPACITY = 16;
const uint SUBMITTED_COMMAND_DRAIN_THRESHOLD = 64;

struct CommandResourceReference {
    ulong owner;
    uint  index;
    uint  generation;
}

struct SubmittedCommandBatch {
    gpu::CompletionPoint       completion;
    RecordingContextHandle     context;
    vk::CommandPool            command_pool;
    vk::CommandBuffer          command_buffer;
    CommandResourceReference[] references;
    uint                       reference_count;
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

fn void ensure_submitted_command_capacity(
    VkDeviceState* state,
    uint additional,
) {
    usz required = (usz)state.submitted_command_count
        + (usz)state.submitted_command_reserved
        + (usz)additional;
    if (required <= state.submitted_commands.len) return;
    usz capacity = state.submitted_commands.len == 0
        ? SUBMITTED_COMMAND_INITIAL_CAPACITY
        : state.submitted_commands.len * 2;
    while (capacity < required) {
        capacity *= 2;
    }
    SubmittedCommandBatch[] grown = alloc::new_array(
        state.host_allocator,
        SubmittedCommandBatch,
        (sz)capacity,
    );
    for (uint i = 0; i < state.submitted_command_count; i++) {
        grown[i] = state.submitted_commands[i];
    }
    if (state.submitted_commands.ptr != null) {
        alloc::free(state.host_allocator, state.submitted_commands.ptr);
    }
    state.submitted_commands = grown;
}

fn uint prepare_submitted_commands(
    VkDeviceState* state,
    CommandRecord*[] records,
) {
    uint additional = (uint)records.len;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    ensure_submitted_command_capacity(state, additional);
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
    CommandRecord*[] records,
    gpu::CompletionPoint completion,
    uint reservation,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.submitted_command_reserved -= reservation;
    foreach (record : records) {
        state.submitted_commands[state.submitted_command_count++] = {
            .context         = record.context,
            .command_pool    = record.command_pool,
            .command_buffer  = record.command_buffer,
            .completion      = completion,
            .references      = record.references,
            .reference_count = record.reference_count,
        };
        record.references = {};
        record.reference_count = 0;
    }
}

fn void release_submitted_command_batch(
    VkDeviceState* state,
    uint index,
) {
    SubmittedCommandBatch* batch = &state.submitted_commands[index];
    release_validation_references(
        state,
        batch.references,
        batch.reference_count,
    );
    alloc::free(state.host_allocator, batch.references.ptr);
    enqueue_retired_command_buffer_locked(
        state:          state,
        context_handle: batch.context,
        pool:           batch.command_pool,
        buffer:         batch.command_buffer,
    );
    state.submitted_command_count--;
    *batch = state.submitted_commands[state.submitted_command_count];
    state.submitted_commands[state.submitted_command_count] = {};
}

fn void release_completed_submitted_commands_locked(
    VkDeviceState* state,
    uint queue_id,
    ulong completed_sequence,
) {
    uint i;
    while (i < state.submitted_command_count) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
        if (gpu::completion_point_queue_id(batch.completion) != queue_id
            || gpu::completion_point_sequence(batch.completion)
                > completed_sequence) {
            i++;
            continue;
        }
        release_submitted_command_batch(state, i);
    }
}

fn void release_completed_submitted_commands(
    VkDeviceState* state,
    uint queue_id,
    ulong completed_sequence,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    release_completed_submitted_commands_locked(
        state,
        queue_id,
        completed_sequence,
    );
}

fn void? drain_completed_submitted_commands_with_query(
    VkDeviceState* state,
    SemaphoreCounterQueryFn query_fn,
) {
    bool[gpu::MAX_SELECTED_QUEUE_IDENTITIES] queried;
    ulong[gpu::MAX_SELECTED_QUEUE_IDENTITIES] completed;
    uint i;
    while (i < state.submitted_command_count) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
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
        release_submitted_command_batch(state, i);
    }
}

fn void? drain_completed_submitted_commands(VkDeviceState* state) {
    return drain_completed_submitted_commands_with_query(
        state,
        &real_semaphore_counter_query,
    );
}

fn void? drain_submitted_commands_if_needed(VkDeviceState* state) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (state.submitted_command_count
        < SUBMITTED_COMMAND_DRAIN_THRESHOLD) return;
    drain_completed_submitted_commands_with_query(state, &real_semaphore_counter_query)!;
}

fn void destroy_submitted_commands(VkDeviceState* state) {
    for (uint i = 0; i < state.submitted_command_count; i++) {
        SubmittedCommandBatch* batch = &state.submitted_commands[i];
        release_validation_references(
            state,
            batch.references,
            batch.reference_count,
        );
        alloc::free(state.host_allocator, batch.references.ptr);
    }
    if (state.submitted_commands.ptr != null) {
        alloc::free(state.host_allocator, state.submitted_commands.ptr);
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
