module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::vk @public;
import vk;
import std::atomic;


fn ulong command_submit_count(gpu::Device* device) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(*device);
    return vk::submit_stats(state).queue_submits;
}

fn void record_empty_executables(
    gpu::CommandAllocator* allocator,
    gpu::ExecutableCommandList[] out_lists,
) {
    foreach (&out_list : out_lists) {
        gpu::CommandList commands = gpu::begin_commands(allocator)!!;
        *out_list = gpu::end_commands(&commands)!!;
    }
}


fn void command_tokens_are_small_owner_values() @test {
    assert(gpu::CommandList::size <= 32,
        "recording token stays within four machine words");
    assert(gpu::ExecutableCommandList::size <= 32,
        "executable token stays within four machine words");
}

fn void discard_consumes_recording_and_executable_lists() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;

    gpu::CommandList recording = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&recording)!!;
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "a recording command list blocks device destruction");
    gpu::discard_commands(&recording)!!;
    assert(!recording.handle.is_valid(), "discard consumes a recording list");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "an executable command list blocks device destruction");
    gpu::ExecutableCommandList stale = executable;
    gpu::discard_executable_commands(&executable)!!;
    assert(!executable.handle.is_valid(), "discard consumes an executable list");
    assert(@catch(gpu::discard_executable_commands(&stale)) == gpu::INVALID_HANDLE,
        "a consumed executable-list alias is stale");

    gpu::destroy_command_allocator(&allocator)!!;
    destroy_test_device(&device)!!;
}

fn void copy_upload_to_readback() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_command_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "upload",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::GpuSpan up_span = upload.span;
    gpu::GpuSpan rb_span = readback.span;
    char[] up_mapping = gpu::get_span_mapping(&device, up_span)!!;
    char* src = up_mapping.ptr;
    for (uint i = 0; i < 256; i++) src[i] = (char)i;
    gpu::flush_mapped_span(&device, upload.span)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferCopyDesc copy = { .src = up_span, .dst = rb_span };
    gpu::reset_command_resolution_stats();
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::Barrier bar = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &bar)!!;

    assert_warm_command_resolution(gpu::command_resolution_stats(), 2);
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.TRANSFER,
        &submit,
    )!!;
    gpu::wait_completion(completion, 1_000_000_000)!!;

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char[] rb_mapping = gpu::get_span_mapping(&device, rb_span)!!;
    char* dst = rb_mapping.ptr;
    for (uint i = 0; i < 256; i++) assert(dst[i] == (char)i, "readback byte matches uploaded byte");
}

fn void empty_command_list_submits() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void command_list_owns_its_device_token_value() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Device temporary = device;
    gpu::Queue queue = gpu::get_queue(&temporary, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&temporary, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    temporary = {};
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;

    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit_desc)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void stale_command_owner_faults_invalid_handle() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList live_commands = commands_executable;
    commands_executable.device = {};

    gpu::ExecutableCommandList[1] stale_lists = { commands_executable };
    gpu::SubmitDesc stale_submit = { .command_lists = stale_lists[..] };
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &stale_submit));
    assert(rejected == gpu::INVALID_HANDLE, "stale command owner faults INVALID_HANDLE");

    gpu::ExecutableCommandList[1] live_lists = { live_commands };
    gpu::SubmitDesc live_submit = { .command_lists = live_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &live_submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void copied_alias_cannot_resubmit_after_success() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList original = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&original)!!;
    gpu::ExecutableCommandList original_executable = gpu::end_commands(&original)!!;
    defer discard_test_executable(&original_executable)!!;
    gpu::ExecutableCommandList[1] copied = { original_executable };
    gpu::SubmitDesc first = { .command_lists = copied[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &first)!!;
    gpu::CommandList replacement = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&replacement)!!;
    gpu::ExecutableCommandList replacement_executable = gpu::end_commands(&replacement)!!;
    defer discard_test_executable(&replacement_executable)!!;
    assert(replacement_executable.handle.index == original_executable.handle.index,
        "the freed command slot is reused");
    assert(replacement_executable.handle.generation != original_executable.handle.generation,
        "slot reuse advances the command generation");


    ulong before = command_submit_count(&device);
    gpu::ExecutableCommandList[1] original_alias = { original_executable };
    gpu::SubmitDesc second = { .command_lists = original_alias[..] };
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &second));
    assert(rejected == gpu::INVALID_HANDLE, "a consumed alias faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "a consumed alias reaches no Vulkan queue submit");
    gpu::ExecutableCommandList[1] replacement_lists = { replacement_executable };
    gpu::SubmitDesc replacement_submit = { .command_lists = replacement_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &replacement_submit)!!;

    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void command_generation_exhaustion_retires_slot() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList final_generation = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&final_generation)!!;
    gpu::ExecutableCommandList final_generation_executable = gpu::end_commands(&final_generation)!!;
    defer discard_test_executable(&final_generation_executable)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    state.commands.slots[final_generation_executable.handle.index].generation =
        vk::HANDLE_GENERATION_MAX - 1;
    final_generation_executable.handle.generation = vk::HANDLE_GENERATION_MAX - 1;
    ((gpu::internal::CommandEncoderState*)final_generation_executable.encoder).handle =
        final_generation_executable.handle;
    gpu::internal::device_slots[device.index].command_leases[
        final_generation_executable.handle.index
    ].store(
        final_generation_executable.handle.generation,
        AtomicOrdering.RELEASE,
    );

    gpu::ExecutableCommandList[1] first_lists = { final_generation_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &first_submit)!!;

    gpu::CommandList next = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&next)!!;
    assert(next.handle.index != final_generation_executable.handle.index,
        "exhausted command slot is retired");
    assert(next.handle.generation == 1,
        "replacement uses a different slot's first generation");
    gpu::ExecutableCommandList next_executable = gpu::end_commands(&next)!!;
    defer discard_test_executable(&next_executable)!!;
    gpu::ExecutableCommandList[1] next_lists = { next_executable };
    gpu::SubmitDesc next_submit = { .command_lists = next_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &next_submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void duplicate_token_batch_rejects_without_consuming() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[2] duplicates = { cmd_executable, cmd_executable };
    gpu::SubmitDesc bad = { .command_lists = duplicates[..] };
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::reset_submit_duplicate_work_counters(state);
    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &bad));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR, "duplicate tokens fault before submission");
    assert(command_submit_count(&device) == before, "duplicate-token rejection records no queue submit");
    vk::SubmitDuplicateWorkCounters duplicate_work =
        vk::submit_duplicate_work_counters(state);
    assert(duplicate_work.token_visits == 2
            && duplicate_work.epoch_reset_cells == 0,
        "a two-token duplicate faults after two linear visits");

    gpu::ExecutableCommandList[1] one = { cmd_executable };
    gpu::SubmitDesc good = { .command_lists = one[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &good)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void submit_duplicate_positions_stop_at_exact_linear_visit() @test {
    const uint BATCH_SIZE = 128;
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = BATCH_SIZE,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::ExecutableCommandList[BATCH_SIZE] unique;
    record_empty_executables(&allocator, unique[..]);
    bool consumed;
    defer {
        if (!consumed) {
            foreach (&list : unique) discard_test_executable(list)!!;
        }
    }
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    ulong submits_before = command_submit_count(&device);

    gpu::ExecutableCommandList[BATCH_SIZE] beginning = unique;
    beginning[1] = unique[0];
    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc beginning_desc = { .command_lists = beginning[..] };
    assert(@catch(gpu::submit(queue, &beginning_desc))
            == gpu::COMMAND_RECORDING_ERROR,
        "a duplicate at the beginning rejects the batch");
    assert(vk::submit_duplicate_work_counters(state).token_visits == 2,
        "a beginning duplicate stops after two visits");

    gpu::ExecutableCommandList[BATCH_SIZE] middle = unique;
    middle[BATCH_SIZE / 2] = unique[0];
    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc middle_desc = { .command_lists = middle[..] };
    assert(@catch(gpu::submit(queue, &middle_desc))
            == gpu::COMMAND_RECORDING_ERROR,
        "a duplicate in the middle rejects the batch");
    assert(vk::submit_duplicate_work_counters(state).token_visits
            == BATCH_SIZE / 2 + 1,
        "a middle duplicate stops at its exact position");

    gpu::ExecutableCommandList[BATCH_SIZE] final = unique;
    final[BATCH_SIZE - 1] = unique[0];
    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc final_desc = { .command_lists = final[..] };
    assert(@catch(gpu::submit(queue, &final_desc))
            == gpu::COMMAND_RECORDING_ERROR,
        "a duplicate at the end rejects the batch");
    assert(vk::submit_duplicate_work_counters(state).token_visits
            == BATCH_SIZE,
        "an end duplicate remains one visit per submitted item");
    assert(command_submit_count(&device) == submits_before,
        "all duplicate positions reject before native submit");
    foreach (list : unique) {
        assert(state.commands.get(list.handle)!!.state
                == vk::CommandState.EXECUTABLE,
            "duplicate faults preserve every command record as executable");
    }

    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc retry = { .command_lists = unique[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &retry)!!;
    consumed = true;
    assert(vk::submit_duplicate_work_counters(state).token_visits
            == BATCH_SIZE,
        "a distinct retry uses a fresh epoch and visits every record once");
    gpu::wait_completion(completion)!!;
}

fn void duplicate_fault_allows_subset_retry_and_remaining_discards() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::ExecutableCommandList[3] lists;
    record_empty_executables(&allocator, lists[..]);
    gpu::ExecutableCommandList[4] duplicate = {
        lists[0],
        lists[1],
        lists[0],
        lists[2],
    };
    gpu::SubmitDesc duplicate_submit = {
        .command_lists = duplicate[..],
    };
    assert(@catch(gpu::submit(queue, &duplicate_submit))
            == gpu::COMMAND_RECORDING_ERROR,
        "a duplicate batch faults before consuming any distinct record");

    gpu::ExecutableCommandList[1] retry = { lists[0] };
    gpu::SubmitDesc retry_submit = { .command_lists = retry[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &retry_submit)!!;
    gpu::discard_executable_commands(&lists[1])!!;
    gpu::discard_executable_commands(&lists[2])!!;
    gpu::wait_completion(completion)!!;
}

fn void submit_valid_batch_sizes_visit_each_record_once() @test {
    const uint MAX_BATCH_SIZE = 1_024;
    uint[5] batch_sizes = { 1, 8, 32, 128, 1_024 };
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = MAX_BATCH_SIZE,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);

    foreach (batch_size : batch_sizes) {
        gpu::ExecutableCommandList[MAX_BATCH_SIZE] lists;
        record_empty_executables(&allocator, lists[:batch_size]);
        vk::reset_submit_duplicate_work_counters(state);
        ulong submits_before = command_submit_count(&device);
        gpu::SubmitDesc submit = {
            .command_lists = lists[:batch_size],
        };

        gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;

        vk::SubmitDuplicateWorkCounters work =
            vk::submit_duplicate_work_counters(state);
        assert(work.token_visits == batch_size
                && work.epoch_reset_cells == 0,
            "a valid batch performs one duplicate visit per command record");
        assert(command_submit_count(&device) == submits_before + 1,
            "a valid batch performs one native queue submit");
        gpu::wait_completion(completion)!!;
    }
}

fn void submit_epoch_rollover_clears_allocated_cells_once() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::ExecutableCommandList[3] lists;
    record_empty_executables(&allocator, lists[..]);
    gpu::CommandListHandle discarded_handle = lists[2].handle;
    gpu::discard_executable_commands(&lists[2])!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::CommandCell* unused =
        &state.commands.slots[discarded_handle.index];
    uint unused_generation = unused.generation;
    vk::CommandRecord* unvisited_live = state.commands.get(lists[1].handle)!!;
    unvisited_live.submit_visit_epoch = ulong::max;
    gpu::ExecutableCommandList[2] duplicate = { lists[0], lists[0] };
    gpu::SubmitDesc submit = { .command_lists = duplicate[..] };

    state.next_submit_visit_epoch = ulong::max - 1;
    vk::reset_submit_duplicate_work_counters(state);
    assert(@catch(gpu::submit(queue, &submit))
            == gpu::COMMAND_RECORDING_ERROR,
        "the maximum epoch remains usable for duplicate detection");
    vk::SubmitDuplicateWorkCounters maximum =
        vk::submit_duplicate_work_counters(state);
    assert(state.next_submit_visit_epoch == ulong::max
            && maximum.token_visits == 2
            && maximum.epoch_reset_cells == 0,
        "the maximum epoch is used once without a rollover sweep");

    vk::reset_submit_duplicate_work_counters(state);
    assert(@catch(gpu::submit(queue, &submit))
            == gpu::COMMAND_RECORDING_ERROR,
        "the rollover epoch still detects an in-attempt duplicate");
    vk::SubmitDuplicateWorkCounters rollover =
        vk::submit_duplicate_work_counters(state);
    assert(state.next_submit_visit_epoch == 1
            && rollover.token_visits == 2
            && rollover.epoch_reset_cells == state.commands.next_new,
        "rollover clears each allocated cell then uses epoch one");
    assert(!unused.used && unused.generation == unused_generation,
        "rollover preserves unused command-cell lifecycle state");
    assert(unvisited_live.submit_visit_epoch == 0,
        "rollover clears old stamps on live records outside the submitted batch");

    state.next_submit_visit_epoch = ulong::max - 1;
    gpu::ExecutableCommandList[1] unvisited_retry = { lists[1] };
    gpu::SubmitDesc unvisited_submit = {
        .command_lists = unvisited_retry[..],
    };
    gpu::CompletionPoint unvisited_completion =
        gpu::submit(queue, &unvisited_submit)!!;
    assert(state.next_submit_visit_epoch == ulong::max,
        "the cleared live record remains valid when its old epoch is reused");
    gpu::wait_completion(unvisited_completion)!!;

    gpu::ExecutableCommandList[1] retry = { lists[0] };
    gpu::SubmitDesc retry_submit = { .command_lists = retry[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &retry_submit)!!;
    gpu::wait_completion(completion)!!;
}

fn void empty_submit_preserves_duplicate_epoch_and_work() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    state.next_submit_visit_epoch = 73;
    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc submit = {};

    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;

    vk::SubmitDuplicateWorkCounters work =
        vk::submit_duplicate_work_counters(state);
    assert(state.next_submit_visit_epoch == 73
            && work.token_visits == 0
            && work.epoch_reset_cells == 0,
        "empty submit signals without allocating a duplicate epoch");
    gpu::wait_completion(completion)!!;
}

fn void readiness_only_submit_validates_stale_swapchain() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    gpu::SwapchainHandle stale = {
        .owner      = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.SWAPCHAIN),
        .generation = 1,
    };
    gpu::SwapchainReadiness readiness = gpu::vk::make_swapchain_readiness(
        device,
        stale,
        1,
    )!!;
    gpu::SubmitDesc submit = {
        .readiness        = readiness,
        .readiness_before = { .color_output },
    };

    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit));
    assert(rejected == gpu::INVALID_HANDLE, "stale readiness faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "stale readiness records no queue submit");
}

fn void readiness_only_submit_rejects_malformed_identity() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::SwapchainReadiness malformed = {
        .device  = device,
        .payload = 1,
    };
    gpu::SubmitDesc submit = {
        .readiness        = malformed,
        .readiness_before = { .color_output },
    };

    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit));
    assert(rejected == gpu::INVALID_HANDLE, "malformed readiness faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "malformed readiness records no queue submit");
}

fn void simultaneous_devices_submit_independently() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device first = create_test_device(&desc)!!;
    gpu::Device second = create_test_device(&desc)!!;
    defer destroy_test_device(&second)!!;

    gpu::Queue first_queue = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator first_allocator = gpu::create_command_allocator(&first, first_queue)!!;
    gpu::CommandList first_cmd = gpu::begin_commands(&first_allocator)!!;
    defer discard_test_commands(&first_cmd)!!;
    gpu::ExecutableCommandList first_cmd_executable = gpu::end_commands(&first_cmd)!!;
    defer discard_test_executable(&first_cmd_executable)!!;
    gpu::ExecutableCommandList[1] first_lists = { first_cmd_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    submit_test_queue(&first, gpu::QueueKind.GRAPHICS, &first_submit)!!;
    wait_test_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::destroy_command_allocator(&first_allocator)!!;
    destroy_test_device(&first)!!;

    assert(gpu::get_device_caps(&second)!!.strict_enabled, "destroying one device leaves the other live");
    gpu::Queue second_queue = gpu::get_queue(&second, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator second_allocator = gpu::create_command_allocator(&second, second_queue)!!;
    defer gpu::destroy_command_allocator(&second_allocator)!!;
    gpu::CommandList second_cmd = gpu::begin_commands(&second_allocator)!!;
    defer discard_test_commands(&second_cmd)!!;
    gpu::ExecutableCommandList second_cmd_executable = gpu::end_commands(&second_cmd)!!;
    defer discard_test_executable(&second_cmd_executable)!!;
    gpu::ExecutableCommandList[1] second_lists = { second_cmd_executable };
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    submit_test_queue(&second, gpu::QueueKind.GRAPHICS, &second_submit)!!;
    wait_test_queue(&second, gpu::QueueKind.GRAPHICS)!!;
}

fn void discarded_command_buffer_is_reused_by_recording_owner() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::CommandBufferStats before = vk::command_buffer_stats(state);
    gpu::CommandList abandoned = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&abandoned)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&abandoned)!!;
    defer discard_test_executable(&executable)!!;
    gpu::discard_executable_commands(&executable)!!;

    vk::CommandBufferStats discarded = vk::command_buffer_stats(state);
    assert(discarded.allocated == before.allocated,
        "recording uses a command buffer preallocated by its allocator");
    assert(discarded.freed == before.freed, "discard leaves reclamation to the recording owner");
    assert(discarded.pending == before.pending, "discard does not create completion-tracked work");
    fault stale = @catch(gpu::discard_executable_commands(&executable));
    assert(stale == gpu::INVALID_HANDLE, "discard invalidates the executable token");

    gpu::CommandList next = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&next)!!;
    vk::CommandBufferStats reused = vk::command_buffer_stats(state);
    assert(reused.allocated == discarded.allocated,
        "the recording owner reuses the discarded command buffer");
    assert(reused.freed == before.freed,
        "warm begin does not free the reusable command buffer");
}

fn void bad_command_state_rejected() @test {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::CommandList stale_recording = commands;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;

    gpu::Barrier barrier = {
        .before = { .host = true },
        .after = { .transfer = true },
    };
    fault rejected = @catch(gpu::cmd_barrier(
        &stale_recording,
        &barrier,
    ));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR,
        "recording through an ended alias faults COMMAND_RECORDING_ERROR");
}
fn void texture_barrier_faults() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier bar = {
        .texture = gpu::TEXTURE_HANDLE_INVALID,
        .before  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after   = gpu::sampled_at({ .fragment_shader }),
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &bar));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on an invalid texture handle faults INVALID_HANDLE");
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void submitted_command_buffer_recycles_at_completion() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::CommandBufferStats before = vk::command_buffer_stats(state);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;

    vk::CommandBufferStats submitted = vk::command_buffer_stats(state);
    assert(submitted.allocated == before.allocated,
        "submit uses a command buffer preallocated by its allocator");
    assert(submitted.freed == before.freed, "submit retains the native command buffer");
    assert(submitted.pending == before.pending + 1, "submit tracks one pending native command buffer");

    gpu::wait_completion(completion, ulong::max)!!;
    vk::CommandBufferStats completed = vk::command_buffer_stats(state);
    assert(completed.freed == before.freed, "wait leaves native reclamation to the recording owner");
    assert(completed.pending == before.pending, "wait retires the completion-tracked command buffer");

    gpu::CommandList next = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&next)!!;
    vk::CommandBufferStats reused = vk::command_buffer_stats(state);
    assert(reused.allocated == completed.allocated,
        "the recording owner reuses the completed command buffer");
    assert(reused.freed == before.freed,
        "warm begin does not free the completed command buffer");
}

fn void command_recording_requires_no_lifecycle_boundary() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    for (uint i = 0; i < 4; i++) {
        gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
        gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
        gpu::ExecutableCommandList[1] lists = { executable };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        gpu::CompletionPoint completion = submit_test_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
            &submit,
        )!!;
        gpu::wait_completion(completion, ulong::max)!!;
    }
}


fn void warm_command_begin_reuses_allocator_owned_storage() @test {
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
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );

    gpu::CommandList first = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::discard_executable_commands(&first_executable)!!;
    vk::reset_command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters before = vk::command_recording_work_counters(
        state,
    );

    gpu::CommandList second = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::discard_executable_commands(&second_executable)!!;
    gpu::internal::RecordingWorkCounters after = vk::command_recording_work_counters(
        state,
    );

    assert(after.command_buffer_allocations
        == before.command_buffer_allocations,
        "warm begin reuses the allocator-owned native command buffer");
    assert(after.command_buffer_frees == before.command_buffer_frees,
        "normal discard never frees a reusable command buffer");
    assert(after.command_buffer_resets == before.command_buffer_resets + 1,
        "warm begin individually resets the completed command buffer");
    assert(after.host_allocations == before.host_allocations,
        "warm begin/discard retains reusable host bookkeeping");
}

fn void recording_work_reset_snapshot_and_delta_are_exact() @test {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "recording_work_stats_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );

    gpu::CommandList warm = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList warm_executable = gpu::end_commands(&warm)!!;
    gpu::discard_executable_commands(&warm_executable)!!;
    vk::reset_command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters zero = vk::command_recording_work_counters(state);
    assert(zero.host_allocations == 0
            && zero.command_pool_creations == 0
            && zero.command_buffer_allocations == 0
            && zero.command_buffer_frees == 0
            && zero.command_buffer_resets == 0
            && zero.image_view_creations == 0
            && zero.vma_allocations == 0
            && zero.registry_lock_acquisitions == 0
            && zero.generated_scratch_misses == 0
            && zero.generated_dispatch_emissions == 0
            && zero.generated_draw_emissions == 0
            && zero.generated_indexed_draw_emissions == 0,
        "reset clears every structural recording-work counter");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::RecordingWorkCounters after = vk::command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters work = vk::command_recording_work_delta(after, zero);
    assert(work.command_buffer_resets == 1,
        "one warmed command-list lifecycle records one reset");
    assert(work.host_allocations == 0
            && work.command_pool_creations == 0
            && work.command_buffer_allocations == 0
            && work.command_buffer_frees == 0
            && work.image_view_creations == 0
            && work.vma_allocations == 0
            && work.registry_lock_acquisitions == 0
            && work.generated_scratch_misses == 0
            && work.generated_dispatch_emissions == 0
            && work.generated_draw_emissions == 0
            && work.generated_indexed_draw_emissions == 0,
        "the warmed lifecycle delta contains no unrelated structural work");

    assert(gpu::internal::device_slot_available(),
        "the registry probe finds an available device slot");
    gpu::internal::RecordingWorkCounters registry_after =
        vk::command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters registry_work =
        vk::command_recording_work_delta(registry_after, after);
    assert(registry_work.registry_lock_acquisitions == 1,
        "the structural delta includes one exact registry-lock acquisition");
    assert(registry_work.host_allocations == 0
            && registry_work.command_pool_creations == 0
            && registry_work.command_buffer_allocations == 0
            && registry_work.command_buffer_frees == 0
            && registry_work.command_buffer_resets == 0
            && registry_work.image_view_creations == 0
            && registry_work.vma_allocations == 0,
        "the registry probe changes no other structural work counter");
}

fn void mixed_allocators_submit_and_retire_to_exact_owners() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator first = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    gpu::CommandAllocator second = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);

    gpu::CommandList first_commands = gpu::begin_commands(&first)!!;
    gpu::CommandList second_commands = gpu::begin_commands(&second)!!;
    uint first_buffer =
        vk::resolve_command(&first_commands)!!.allocator_buffer;
    uint second_buffer =
        vk::resolve_command(&second_commands)!!.allocator_buffer;
    gpu::ExecutableCommandList[2] lists = {
        gpu::end_commands(&first_commands)!!,
        gpu::end_commands(&second_commands)!!,
    };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    vk::CommandAllocatorSlot* first_slot =
        state.command_allocators.get(first.handle)!!;
    vk::CommandAllocatorSlot* second_slot =
        state.command_allocators.get(second.handle)!!;

    assert(@catch(gpu::destroy_command_allocator(&first))
            == gpu::RESOURCE_IN_USE
        && @catch(gpu::destroy_command_allocator(&second))
            == gpu::RESOURCE_IN_USE,
        "each submitted unit keeps its originating allocator live");
    assert(first_slot.available_count == 0
            && second_slot.available_count == 0
            && first_slot.live_executables == 0
            && second_slot.live_executables == 0
            && first_slot.in_flight_buffers == 1
            && second_slot.in_flight_buffers == 1
            && first_slot.scratch[first_buffer].in_use
            && second_slot.scratch[second_buffer].in_use,
        "successful publication transfers both units without an ownership gap");
    gpu::wait_completion(completion, ulong::max)!!;
    assert(gpu::poll_completion(completion)!!,
        "polling an already-retired mixed completion is idempotent");
    gpu::wait_completion(completion, ulong::max)!!;
    assert(first_slot.available_count == 1
            && second_slot.available_count == 1
            && first_slot.in_flight_buffers == 0
            && second_slot.in_flight_buffers == 0
            && !first_slot.scratch[first_buffer].in_use
            && !second_slot.scratch[second_buffer].in_use,
        "repeated completion observation returns each exact allocator unit once");
    gpu::CommandList first_reuse = gpu::begin_commands(&first)!!;
    gpu::CommandList second_reuse = gpu::begin_commands(&second)!!;
    assert(vk::resolve_command(&first_reuse)!!.allocator_buffer == first_buffer
            && vk::resolve_command(&second_reuse)!!.allocator_buffer
                == second_buffer,
        "mixed retirement makes each exact preallocated buffer reusable");
    gpu::discard_commands(&first_reuse)!!;
    gpu::discard_commands(&second_reuse)!!;
    gpu::destroy_command_allocator(&first)!!;
    gpu::destroy_command_allocator(&second)!!;
    destroy_test_device(&device)!!;
}

fn void later_completion_observation_retires_multi_allocator_prefix_once() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator first = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    gpu::CommandAllocator second = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);

    gpu::CommandList first_commands = gpu::begin_commands(&first)!!;
    uint first_buffer =
        vk::resolve_command(&first_commands)!!.allocator_buffer;
    gpu::ExecutableCommandList[1] first_lists = {
        gpu::end_commands(&first_commands)!!,
    };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::CompletionPoint first_completion = gpu::submit(
        queue,
        &first_submit,
    )!!;

    gpu::CommandList second_commands = gpu::begin_commands(&second)!!;
    uint second_buffer =
        vk::resolve_command(&second_commands)!!.allocator_buffer;
    gpu::ExecutableCommandList[1] second_lists = {
        gpu::end_commands(&second_commands)!!,
    };
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    gpu::CompletionPoint second_completion = gpu::submit(
        queue,
        &second_submit,
    )!!;
    vk::CommandAllocatorSlot* first_slot =
        state.command_allocators.get(first.handle)!!;
    vk::CommandAllocatorSlot* second_slot =
        state.command_allocators.get(second.handle)!!;
    assert(first_slot.in_flight_buffers == 1
            && second_slot.in_flight_buffers == 1,
        "separate submissions keep both allocator units in flight");

    gpu::wait_completion(second_completion, ulong::max)!!;
    assert(first_slot.in_flight_buffers == 0
            && second_slot.in_flight_buffers == 0
            && first_slot.available_count == 1
            && second_slot.available_count == 1
            && !first_slot.scratch[first_buffer].in_use
            && !second_slot.scratch[second_buffer].in_use,
        "observing the later point retires the multi-allocator queue prefix");
    assert(gpu::poll_completion(first_completion)!!,
        "the earlier point stays observable after later-prefix retirement");
    gpu::wait_completion(first_completion, ulong::max)!!;
    assert(first_slot.available_count == 1
            && second_slot.available_count == 1,
        "out-of-order and repeated observation cannot return either unit twice");

    gpu::destroy_command_allocator(&first)!!;
    gpu::destroy_command_allocator(&second)!!;
    destroy_test_device(&device)!!;
}

fn void submit_rejects_stale_allocator_generation_before_claim() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    vk::CommandRecord* record = state.commands.get(executable.handle)!!;
    gpu::CommandAllocatorHandle original = record.allocator;
    record.allocator.generation++;

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    assert(@catch(gpu::submit(queue, &submit)) == gpu::INVALID_HANDLE,
        "submit rejects a stale allocator generation before claiming tokens");
    assert(lists[0].handle.is_valid()
        && record.state == vk::CommandState.EXECUTABLE,
        "allocator validation failure preserves executable state");

    record.allocator = original;
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion, ulong::max)!!;
}
