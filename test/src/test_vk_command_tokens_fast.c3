module gpu_test;

import gpu::internal @public;
import gpu @public;
import std::thread;
import std::atomic;
import std::core::mem::allocators;

struct DirectRecordWorkerArgs {
    gpu::CommandAllocator*         allocator;
    Atomic{bool}*                 start;
    Atomic{bool}                  ready;
    Atomic{bool}                  release;
    gpu::internal::CommandRecord* record;
    gpu::ExecutableCommandList    executable;
}

fn gpu::internal::CommandRecord* direct_record(
    gpu::CommandList* commands,
) {
    return (gpu::internal::CommandRecord*)commands.record;
}

fn gpu::internal::CommandRecord* direct_executable_record(
    gpu::ExecutableCommandList* commands,
) {
    return (gpu::internal::CommandRecord*)commands.record;
}

fn int direct_record_worker(void* raw_args) {
    DirectRecordWorkerArgs* args = (DirectRecordWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        while (!args.start.load(AtomicOrdering.ACQUIRE)) thread::yield();
        gpu::CommandList commands = gpu::begin_commands(args.allocator)!!;
        args.record = direct_record(&commands);
        args.ready.store(true, AtomicOrdering.RELEASE);
        while (!args.release.load(AtomicOrdering.ACQUIRE)) thread::yield();
        gpu::Barrier barrier = {
            .before = { .transfer = true },
            .after  = { .host = true },
        };
        gpu::cmd_barrier(&commands, &barrier);
        args.executable = gpu::end_commands(&commands)!!;
    };
    return 0;
}

fn void direct_command_tokens_are_one_pointer() @test {
    assert(gpu::DIRECT_COMMAND_TOKENS);
    assert(gpu::CommandList::size == void*::size,
        "a direct recording token is exactly one pointer");
    assert(gpu::ExecutableCommandList::size == void*::size,
        "a direct executable token is exactly one pointer");
}

fn void direct_command_tokens_reject_full_validation() @test {
    gpu::RuntimeDesc desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .contract_validation = gpu::ContractValidation.FULL,
    };
    assert(@catch(gpu::create_runtime(&desc)) == gpu::INVALID_ARGUMENT,
        "the FAST profile rejects full validation before Vulkan initialization");
}

fn void direct_command_record_lifecycle_reuses_only_after_retirement() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    gpu::CommandList discarded = gpu::begin_commands(&allocator)!!;
    gpu::internal::CommandRecord* discarded_record = direct_record(&discarded);
    assert(discarded_record.state == gpu::internal::CommandState.RECORDING);
    gpu::discard_commands(&discarded)!!;
    assert(discarded.record == null
        && discarded_record.state == gpu::internal::CommandState.INACTIVE,
        "recording discard consumes the token before releasing the record");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::internal::CommandRecord* record = direct_record(&commands);
    assert(record == discarded_record,
        "begin reinitializes the same fixed record");
    gpu::Barrier barrier = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::reset_command_resolution_stats();
    gpu::cmd_barrier(&commands, &barrier);
    assert_warm_command_resolution(gpu::command_resolution_stats(), 1);

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    assert(commands.record == null
        && direct_executable_record(&executable) == record
        && record.state == gpu::internal::CommandState.EXECUTABLE,
        "end consumes the recording value and preserves the record");
    gpu::ExecutableCommandList[1] submitted = { executable };
    gpu::SubmitDesc submit = { .command_lists = submitted[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    assert(submitted[0].record == null
        && record.state == gpu::internal::CommandState.SUBMITTED,
        "submit consumes only the public value");
    gpu::CommandList concurrent = gpu::begin_commands(&allocator)!!;
    assert(direct_record(&concurrent) != record,
        "submitted storage is not reused before retirement");

    gpu::wait_completion(completion)!!;
    assert(record.state == gpu::internal::CommandState.INACTIVE,
        "ordered retirement invalidates the authoritative record");
    gpu::discard_commands(&concurrent)!!;
    gpu::CommandList reused = gpu::begin_commands(&allocator)!!;
    gpu::internal::CommandRecord* reused_record = direct_record(&reused);
    assert(reused_record.state == gpu::internal::CommandState.RECORDING,
        "returned fixed storage is fully reinitialized");
    gpu::ExecutableCommandList reused_executable =
        gpu::end_commands(&reused)!!;
    gpu::discard_executable_commands(&reused_executable)!!;
    assert(reused_record.state == gpu::internal::CommandState.INACTIVE,
        "executable discard invalidates the authoritative record");
}

fn void distinct_allocators_record_in_parallel_with_direct_tokens() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator[2] allocators;
    DirectRecordWorkerArgs[2] args;
    Thread[2] workers;
    Atomic{bool} start;
    for (uint i = 0; i < workers.len; i++) {
        allocators[i] = gpu::create_command_allocator(
            &device,
            queue,
        )!!;
        args[i].allocator = &allocators[i];
        args[i].start = &start;
        workers[i].create(&direct_record_worker, &args[i])!!;
    }

    start.store(true, AtomicOrdering.RELEASE);
    while (!args[0].ready.load(AtomicOrdering.ACQUIRE)
        || !args[1].ready.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    bool simultaneous_recording =
        args[0].record != null
        && args[1].record != null
        && args[0].record != args[1].record
        && args[0].record.state == gpu::internal::CommandState.RECORDING
        && args[1].record.state == gpu::internal::CommandState.RECORDING;
    foreach (&worker_args : args) {
        worker_args.release.store(true, AtomicOrdering.RELEASE);
    }
    foreach (&worker : workers) assert(worker.join()!! == 0);

    assert(simultaneous_recording,
        "two worker threads hold distinct recording records concurrently");
    foreach (i, &worker_args : args) {
        assert(direct_executable_record(&worker_args.executable)
                == worker_args.record
            && worker_args.record.state
                == gpu::internal::CommandState.EXECUTABLE,
            "each worker ends through its own authoritative record");
        gpu::discard_executable_commands(&worker_args.executable)!!;
        assert(worker_args.record.state
                == gpu::internal::CommandState.INACTIVE,
            "each worker record invalidates on executable discard");
        gpu::destroy_command_allocator(&allocators[i])!!;
    }
}

fn void direct_lost_device_discards_recording_and_executable_records() @test {
    gpu::RuntimeDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;

    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[device.index];
    ulong children_before =
        slot.live_children.load(AtomicOrdering.ACQUIRE);
    gpu::CommandList recording = gpu::begin_commands(&allocator)!!;
    gpu::internal::CommandRecord* recording_record =
        direct_record(&recording);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::CommandRecord* executable_record =
        direct_executable_record(&executable);

    gpu::internal::mark_device_lost(&device);
    gpu::discard_commands(&recording)!!;
    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );

    assert(!recording.is_valid() && !executable.is_valid()
        && recording_record.state == gpu::internal::CommandState.INACTIVE
        && executable_record.state == gpu::internal::CommandState.INACTIVE,
        "lost-device discard consumes both direct token phases");
    assert(control.pins == 0
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == children_before,
        "lost-device discard releases both retained pins and command children");
    gpu::destroy_command_allocator(&allocator)!!;
}
