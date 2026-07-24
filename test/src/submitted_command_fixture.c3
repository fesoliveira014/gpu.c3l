module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn void init_submitted_command_fixture(
    vk::VkDeviceState* state,
    vk::CommandCell[] cells,
) {
    state.command_mutex.init()!!;
    state.commands = {
        .slots = cells,
        .owner = 1,
    };
}

fn gpu::internal::CommandRecord* submitted_command_fixture_record(
    vk::VkDeviceState* state,
    gpu::CommandAllocatorHandle allocator,
    uint allocator_buffer,
) {
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator,
    )!!;
    gpu::internal::CommandRecord* record = state.commands.claim(
        device:        gpu::DEVICE_INVALID,
        ops:           null,
        backend_state: state,
        value: {
            .allocator        = allocator,
            .allocator_buffer = allocator_buffer,
            .scratch          = &slot.scratch[allocator_buffer],
        },
    )!!;
    record.state = gpu::internal::CommandState.SUBMITTED;
    return record;
}

fn void append_submitted_command_fixture(
    vk::QueueCompletionState* completion,
    gpu::internal::CommandRecord* record,
    gpu::CompletionPoint point,
) {
    vk::VkCommandRecord* backend = vk::backend_command(record);
    backend.submitted = {
        .completion = point,
        .record     = record,
    };
    backend.next_submitted = null;
    if (completion.submitted_tail == null) {
        completion.submitted_head = record;
    } else {
        vk::backend_command(completion.submitted_tail).next_submitted = record;
    }
    completion.submitted_tail = record;
    completion.submitted_count++;
}
