module gpu_test;

import gpu @public;
import gpu::vk;
import vk;

fn void attachment_view_table_is_generation_checked() @test {
    vk::AttachmentViewCell[1] cells;
    vk::AttachmentViewTable table = {
        .slots = cells[..],
        .owner = 41,
    };
    vk::AttachmentViewSlot slot = {
        .texture      = { .owner = 7, .index = 3, .generation = 1 },
        .format       = gpu::Format.RGBA8_UNORM,
        .sample_count = gpu::SampleCount.ONE,
        .width        = 16,
        .height       = 8,
    };

    gpu::AttachmentViewHandle first = table.alloc(slot)!!;
    assert(first.owner == 41 && first.index == 0,
        "attachment table publishes its exact owner and slot");
    assert(table.get(first)!!.width == 16,
        "attachment metadata remains immutable while live");
    table.free(first)!!;
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "destroyed attachment generations are stale");

    gpu::AttachmentViewHandle second = table.alloc(slot)!!;
    assert(second.index == first.index
        && second.generation != first.generation,
        "recycled attachment slots advance generation");
}

fn void attachment_view_table_rejects_foreign_and_full_handles() @test {
    vk::AttachmentViewCell[1] cells;
    vk::AttachmentViewTable table = {
        .slots = cells[..],
        .owner = 73,
    };
    gpu::AttachmentViewHandle live = table.alloc({})!!;
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "live attachment capacity faults deterministically");

    gpu::AttachmentViewHandle foreign = live;
    foreign.owner++;
    assert(@catch(table.get(foreign)) == gpu::INVALID_HANDLE,
        "foreign attachment owners cannot alias local slots");
}

fn void attachment_view_generation_exhaustion_retires_slot() @test {
    vk::AttachmentViewCell[1] cells;
    cells[0].generation = vk::HANDLE_GENERATION_MAX - 1;
    vk::AttachmentViewTable table = {
        .slots     = cells[..],
        .owner     = 91,
        .free_head = 1,
        .next_new  = 1,
    };

    gpu::AttachmentViewHandle final = table.alloc({})!!;
    assert(final.generation == vk::HANDLE_GENERATION_MAX,
        "the final attachment generation can be published once");
    table.free(final)!!;
    assert(cells[0].generation == vk::HANDLE_GENERATION_MAX
        && table.free_head == 0,
        "an exhausted attachment slot is permanently retired");
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "an exhausted attachment generation has no successor");
}

fn void warm_command_begin_reuses_context_owned_storage() @test {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "warm_command_reuse_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state_ptr(
        &device,
    );

    gpu::CommandList first = gpu::begin_commands(queue)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::discard_executable_commands(&first_executable)!!;
    vk::reset_command_recording_work_counters(state);
    gpu::RecordingWorkCounters before = vk::command_recording_work_counters(
        state,
    );

    gpu::CommandList second = gpu::begin_commands(queue)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::discard_executable_commands(&second_executable)!!;
    gpu::RecordingWorkCounters after = vk::command_recording_work_counters(
        state,
    );

    assert(after.command_buffer_allocations
        == before.command_buffer_allocations,
        "warm begin reuses the context-owned native command buffer");
    assert(after.command_buffer_frees == before.command_buffer_frees,
        "normal discard never frees a reusable command buffer");
    assert(after.command_buffer_resets == before.command_buffer_resets + 1,
        "warm begin individually resets the completed command buffer");
    assert(after.host_allocations == before.host_allocations,
        "warm begin/discard retains reusable host bookkeeping");
}

fn void generated_scratch_reservation_is_explicit_and_bounded() @test {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "generated_scratch_reservation_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::GeneratedScratchDesc reservation = {
        .max_commands_per_list   = 8,
        .max_preprocess_bytes    = 4096,
        .preprocess_buffer_count = 1,
    };
    gpu::reserve_generated_scratch(queue, reservation)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state_ptr(
        &device,
    );
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    assert(@catch(gpu::reserve_generated_scratch(queue, reservation))
        == gpu::RESOURCE_IN_USE,
        "reservation replacement requires a quiescent context");

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::RecordingContextState* context = state.recording_contexts.get(
        record.context,
    )!!;
    vk::GeneratedPreprocessBuffer* reserved =
        &context.reserved_preprocess[0];
    vk::MemoryRequirements requirements = {
        .size             = 1024,
        .alignment        = 1,
        .memory_type_bits = 1 << reserved.memory_type,
    };
    gpu::RecordingWorkCounters before = vk::command_recording_work_counters(
        state,
    );
    vk::GeneratedPreprocessBuffer* acquired =
        vk::acquire_generated_preprocess_buffer(
            state,
            record,
            &requirements,
            8,
        )!!;
    gpu::RecordingWorkCounters after = vk::command_recording_work_counters(
        state,
    );
    assert(acquired.address == reserved.address,
        "generated recording consumes the exact reserved buffer");
    assert(after.host_allocations == before.host_allocations
        && after.vma_allocations == before.vma_allocations,
        "reserved acquisition performs no host or VMA allocation");
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state,
        record,
        &requirements,
        9,
    )) == gpu::CAPACITY_EXCEEDED,
        "work above the explicit command count faults deterministically");
    vk::MemoryRequirements oversized = requirements;
    oversized.size = reservation.max_preprocess_bytes + 1;
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state,
        record,
        &oversized,
        1,
    )) == gpu::CAPACITY_EXCEEDED,
        "preprocess bytes above the reservation fault deterministically");
    vk::MemoryRequirements incompatible = requirements;
    incompatible.memory_type_bits = 0;
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state,
        record,
        &incompatible,
        1,
    )) == gpu::CAPACITY_EXCEEDED,
        "incompatible preprocess memory requirements fault without allocation");
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state,
        record,
        &requirements,
        1,
    )) == gpu::CAPACITY_EXCEEDED,
        "an in-use reserved buffer cannot be aliased by one command list");
    assert(@catch(gpu::reserve_generated_scratch(queue, {
        .max_preprocess_bytes    = 4096,
        .preprocess_buffer_count = 1,
    })) == gpu::INVALID_ARGUMENT,
        "invalid replacement leaves the live reservation unchanged");

    gpu::discard_commands(&commands)!!;
    gpu::CommandList retry = gpu::begin_commands(queue)!!;
    vk::CommandRecord* retry_record = vk::resolve_command(&retry)!!;
    vk::GeneratedPreprocessBuffer* reacquired =
        vk::acquire_generated_preprocess_buffer(
            state,
            retry_record,
            &requirements,
            1,
        )!!;
    assert(reacquired.address == reserved.address,
        "discard returns reserved preprocess storage for retry");
    gpu::discard_commands(&retry)!!;
    gpu::reserve_generated_scratch(queue, {})!!;
}
