module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;
import std::atomic;
import std::thread;

const uint DEVICE_REGISTRY_THREAD_COUNT = 8;

struct DeviceRegistryWorkerArgs {
    Atomic{uint}* ready;
    Atomic{bool}* release;
    DeviceFixture fixture;
    gpu::Device   device;
    bool          failed;
}

struct BlockingDestroyFixture {
    gpu::internal::vk::VkDeviceState state;
}

struct BlockingOperationFixture {
    gpu::internal::vk::VkDeviceState state;
}

struct CommandPinFixture {
    gpu::internal::vk::VkDeviceState state;
    gpu::Device  device;
}

struct DiscardCommandWorkerArgs {
    gpu::CommandList commands;
    fault            result;
    bool             failed;
}

struct DestroyWorkerArgs {
    gpu::Device device;
    bool        failed;
}

struct DeviceOperationWorkerArgs {
    gpu::Device device;
}

struct RuntimeRetainWorkerArgs {
    gpu::Runtime  runtime;
    Atomic{uint}* ready;
    Atomic{bool}* release;
    bool          failed;
}

fn int discard_command_worker(void* raw_args) {
    DiscardCommandWorkerArgs* args = (DiscardCommandWorkerArgs*)raw_args;
    if (catch result = gpu::discard_commands(&args.commands)) {
        args.result = result;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn gpu::CommandList? begin_command_pin(gpu::Device* device) {
    if (device == null || !device.is_valid()) {
        abort("gpu: zero Device");
    }
    gpu::internal::DeviceData* data = gpu::internal::resolve_pinned_device(device);
    if (data == null) abort("gpu: stale Device");
    gpu::CommandAllocator allocator = {
        .device = *device,
        .queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS),
        .handle = {
            .owner = gpu::internal::resource_owner_tag(
                data.resource_owner,
                gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
            ),
            .generation = 1,
        },
    };
    return gpu::begin_commands(&allocator);
}

fn int destroy_device_worker(void* raw_args) {
    DestroyWorkerArgs* args = (DestroyWorkerArgs*)raw_args;
    if (catch destroy_fault = gpu::destroy_device(&args.device)) {
        (void)destroy_fault;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int device_operation_worker(void* raw_args) {
    DeviceOperationWorkerArgs* args = (DeviceOperationWorkerArgs*)raw_args;
    (void)gpu::get_memory_stats(&args.device);
    return 0;
}

fn int runtime_retain_worker(void* raw_args) {
    RuntimeRetainWorkerArgs* args = (RuntimeRetainWorkerArgs*)raw_args;
    bool retained = true;
    if (catch retain_fault = gpu::internal::runtime_retain_device(&args.runtime)) {
        (void)retain_fault;
        args.failed = true;
        retained = false;
    }

    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (retained) {
        if (catch release_fault = gpu::internal::runtime_release_device(&args.runtime)) {
            (void)release_fault;
            args.failed = true;
        }
    }
    return args.failed ? 1 : 0;
}

fn int device_registry_worker(void* raw_args) {
    DeviceRegistryWorkerArgs* args = (DeviceRegistryWorkerArgs*)raw_args;
    gpu::Device? registered = gpu::internal::register_device(
        state:  &args.fixture.state,
        caps:   {},
    );
    if (catch registration_fault = registered) {
        (void)registration_fault;
        args.failed = true;
    } else {
        args.device = registered;
    }

    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (!args.failed) {
        gpu::Device device = args.device;
        gpu::destroy_device(&device)!!;
    }
    return args.failed ? 1 : 0;
}

fn void command_allocator_public_lifecycle_is_transactional() @test {
    CommandPinFixture fixture;
    fixture.state.resource_owner =
        gpu::internal::allocate_device_resource_owner()!!;
    gpu::Device device = gpu::internal::register_device(
        state:          &fixture.state,
        caps:           {},
        resource_owner: fixture.state.resource_owner,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    );
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];

    assert(allocator.device == device && allocator.queue == queue,
        "allocator carries its exact device and queue identity");
    assert(fixture.state.allocator_desc.command_buffer_capacity
            == gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY
        && fixture.state.allocator_desc
            .max_acceleration_structure_geometries_per_build == 0,
        "zero descriptor fields normalize before backend dispatch");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 1,
        "a published allocator retains one device child");

    fixture.state.reject_allocator_destroy = true;
    gpu::CommandAllocator original = allocator;
    assert(@catch(gpu::destroy_command_allocator(&allocator))
        == gpu::RESOURCE_IN_USE,
        "backend destroy failure remains retryable");
    assert(allocator.handle == original.handle
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == 1,
        "failed destroy preserves the token and child retain");

    fixture.state.reject_allocator_destroy = false;
    gpu::destroy_command_allocator(&allocator)!!;
    assert(!allocator.handle.is_valid()
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "successful destroy consumes the token and releases its child");
    gpu::destroy_device(&device)!!;
}

fn void command_allocator_creation_rejects_before_publication() @test {
    CommandPinFixture first_fixture;
    CommandPinFixture second_fixture;
    first_fixture.state.resource_owner =
        gpu::internal::allocate_device_resource_owner()!!;
    gpu::Device first = gpu::internal::register_device(
        state:          &first_fixture.state,
        caps:           {},
        resource_owner: first_fixture.state.resource_owner,
    )!!;
    gpu::Device second = gpu::internal::register_device(
        state:  &second_fixture.state,
        caps:   {},
    )!!;
    gpu::Queue foreign = gpu::get_queue(
        &second,
        gpu::QueueKind.GRAPHICS,
    );
    gpu::CommandAllocatorDesc oversized = {
        .command_buffer_capacity = gpu::MAX_COMMAND_ALLOCATOR_CAPACITY + 1,
    };

    gpu::Queue queue = gpu::get_queue(
        &first,
        gpu::QueueKind.GRAPHICS,
    );
    assert(@catch(gpu::create_command_allocator(
        &first,
        queue,
        &oversized,
    )) == gpu::INVALID_ARGUMENT,
        "an oversized descriptor is rejected before backend creation");
    assert(first_fixture.state.allocator_creates == 0
        && gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "validation failures publish no backend allocator or child");

    first_fixture.state.reject_allocator_create = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::DEVICE_BUSY,
        "backend create faults propagate exactly");
    assert(gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "backend create failure retains no child");

    first_fixture.state.reject_allocator_create = false;
    first_fixture.state.return_invalid_allocator = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::BACKEND_ERROR,
        "an invalid backend handle fails publication");
    first_fixture.state.return_invalid_allocator = false;
    first_fixture.state.return_foreign_allocator = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::BACKEND_ERROR,
        "a foreign backend handle fails publication");
    assert(first_fixture.state.allocator_destroy_calls == 2
        && gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "invalid backend publication rolls back exactly once without a child");
    gpu::destroy_device(&second)!!;
    gpu::destroy_device(&first)!!;
}

fn void command_calls_use_the_retained_lifetime_pin() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::internal::vk::CommandRecord* record = commands.record;
    uint record_index = record.identity.index;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_begin =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::DeviceSlotControl after_end =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    bool record_transferred =
        record.state == gpu::internal::CommandState.EXECUTABLE
        && executable.record == record
        && executable.generation == record.identity.generation;
    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(after_begin.pins == 1,
        "begin transfers its operation pin to the recording token");
    assert(record_transferred,
        "end transfers the same stable record to its executable phase");
    assert(fixture.state.pins_during_command == 1,
        "hot command recording uses the retained lifetime without another pin");
    assert(after_end.pins == 1,
        "the executable token retains the same lifetime pin");
    assert(after_discard.pins == 0,
        "discard releases the command lifetime pin exactly once");
    assert(record.state == gpu::internal::CommandState.INACTIVE
        && !fixture.state.command_used[record_index],
        "successful discard invalidates the record before lifetime release");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases the command child exactly once");
    gpu::destroy_device(&device)!!;
}

fn void failed_command_lifecycle_preserves_record_state() @test {
    CommandPinFixture fixture = {
        .state = {
            .reject_discard = true,
            .reject_end     = true,
        },
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::CommandList original = commands;
    gpu::internal::vk::CommandRecord* record = commands.record;

    fault end_fault = @catch(gpu::end_commands(&commands));
    fault discard_fault = @catch(gpu::discard_commands(&commands));

    assert(end_fault == gpu::DEVICE_BUSY
        && discard_fault == gpu::DEVICE_BUSY,
        "failed end and discard preserve their backend faults");
    assert(commands.record == original.record
        && commands.generation == original.generation,
        "failed lifecycle calls leave the recording token unchanged");
    assert(record.state == gpu::internal::CommandState.RECORDING,
        "failed lifecycle calls leave the record recording");

    fixture.state.reject_end = false;
    fixture.state.reject_discard = false;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
    gpu::destroy_device(&device)!!;
}

fn void executable_alias_of_recording_token_faults_command_state() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList forged = {
        .record          = commands.record,
        .generation      = commands.generation,
        .device_identity = commands.device_identity,
    };

    assert(@catch(gpu::discard_executable_commands(&forged))
        == gpu::COMMAND_RECORDING_ERROR,
        "an executable alias of a recording token faults command state");
    gpu::discard_commands(&commands)!!;
    gpu::destroy_device(&device)!!;
}

fn void failed_submit_preserves_command_lifetime_pins() @test {
    CommandPinFixture fixture = {
        .state = { .submit_busy = true },
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    gpu::CommandList second = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::ExecutableCommandList[2] lists = {
        first_executable,
        second_executable,
    };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    fault submit_fault = @catch(gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS), &desc));
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_failure =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    ulong children_after_failure =
        slot.live_children.load(AtomicOrdering.ACQUIRE);

    fixture.state.submit_busy = false;
    gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS), &desc)!!;
    gpu::internal::DeviceSlotControl after_success =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(submit_fault == gpu::DEVICE_BUSY,
        "retryable submit failure preserves executable tokens");
    assert(fixture.state.pins_during_submit == 3,
        "submit observes two command pins and its short operation pin");
    assert(after_failure.pins == 2,
        "failed submit preserves both command lifetime pins");
    assert(children_after_failure == 2,
        "failed submit preserves both command children");
    assert(after_success.pins == 0,
        "successful submit releases both command lifetime pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "successful submit releases both command children");
    gpu::destroy_device(&device)!!;
}

fn void device_loss_after_backend_submit_still_consumes_every_token() @test {
    CommandPinFixture fixture = {
        .state = { .lose_during_submit = true },
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    gpu::CommandList second = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::ExecutableCommandList[2] lists = {
        first_executable,
        second_executable,
    };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS),
        &desc,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(!lists[0].is_valid() && !lists[1].is_valid(),
        "successful submit consumes every token after concurrent device loss");
    assert(control.pins == 0,
        "post-submit device loss cannot strand command lifetime pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "post-submit device loss cannot strand command children");
    gpu::destroy_device(&device)!!;
}

fn void command_record_is_not_reused_before_discard_returns() @test {
    CommandPinFixture fixture = {
        .state = {
            .block_discard       = true,
            .reuse_command_index = true,
        },
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    );
    gpu::CommandList first = begin_command_pin(&device)!!;
    DiscardCommandWorkerArgs worker_args = { .commands = first };
    Thread worker;
    worker.create(&discard_command_worker, &worker_args)!!;
    while (!fixture.state.discard_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault busy = @catch(begin_command_pin(&device));
    fixture.state.discard_release.store(true, AtomicOrdering.RELEASE);
    assert(worker.join()!! == 0 && !worker_args.failed,
        "the prior discard completes before record reuse");
    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_failed_begin =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );
    assert(after_failed_begin.pins == 0,
        "the failed begin cannot strand a device lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "the failed begin cannot strand a command child");
    gpu::CommandList next = begin_command_pin(&device)!!;
    assert(busy == gpu::DEVICE_BUSY,
        "a discarding record remains unavailable");
    assert(next.record == first.record
        && next.generation != first.generation,
        "the backend generation-advances a record only after discard");

    gpu::discard_commands(&next)!!;
    gpu::destroy_device(&device)!!;
}

fn void lost_device_commands_record_end_and_discard() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::internal::mark_device_lost(&device);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    fault destroy_fault = @catch(gpu::destroy_device(&device));
    uint destroy_calls_before_discard =
        fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE);
    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(fixture.state.command_entered.load(AtomicOrdering.ACQUIRE),
        "recording reaches the backend after device loss");
    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "a live command child blocks destroying a lost device");
    assert(destroy_calls_before_discard == 0,
        "device teardown does not begin before the lost command is discarded");
    assert(after_discard.pins == 0,
        "discard releases a lost command lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases a lost command child");
    gpu::destroy_device(&device)!!;
}

fn void lost_executable_commands_can_be_discarded() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::mark_device_lost(&device);

    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );

    assert(after_discard.pins == 0,
        "executable discard releases a lost command lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "executable discard releases a lost command child");
    gpu::destroy_device(&device)!!;
}

fn void device_handles_are_compact_structured_tokens() @test {
    assert(gpu::Device::size == ulong::size, "Device stays one machine word");

    gpu::Device device = {
        .index      = 7,
        .generation = 11,
    };
    assert(device.is_valid(), "a nonzero generation is valid");
    assert(device.index == 7, "the device index round-trips");
    assert(device.generation == 11, "the device generation round-trips");
    assert(device.reserved == 0, "valid devices keep reserved bits clear");
}

fn void device_loss_isolated_to_marked_slot() @test {
    DeviceFixture first_fixture;
    DeviceFixture second_fixture;
    gpu::Device first = fixture_device(&first_fixture);
    gpu::Device second = fixture_device(&second_fixture);

    gpu::internal::mark_device_lost(&first);

    assert(gpu::is_device_lost(&first), "the affected device stays lost");
    assert(!gpu::is_device_lost(&second), "a peer device is unaffected");
    (void)gpu::get_device_caps(&first);
    (void)gpu::get_device_caps(&second);

    gpu::Device stale = first;
    gpu::destroy_device(&first)!!;
    DeviceFixture replacement_fixture;
    gpu::Device replacement = fixture_device(&replacement_fixture);
    gpu::internal::mark_device_lost(&stale);
    (void)gpu::get_device_caps(&replacement);

    gpu::destroy_device(&second)!!;
    gpu::destroy_device(&replacement)!!;
}

fn void exhausted_device_slots_are_retired() @test {
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[0];
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "the exhaustion fixture starts from a free slot");

    control.generation = gpu::internal::MAX_DEVICE_GENERATION - 1;
    slot.control.store((ulong)control, AtomicOrdering.RELAXED);

    DeviceFixture penultimate_fixture;
    gpu::Device penultimate = fixture_device(&penultimate_fixture);
    assert(penultimate.index == 0, "the seeded slot publishes its current generation");
    assert(penultimate.generation == gpu::internal::MAX_DEVICE_GENERATION - 1, "registration does not advance generation");
    gpu::destroy_device(&penultimate)!!;

    control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "the final generation remains publishable");
    assert(control.generation == gpu::internal::MAX_DEVICE_GENERATION, "successful destruction advances to the final generation");

    DeviceFixture final_fixture;
    gpu::Device final = fixture_device(&final_fixture);
    assert(final.index == 0, "the final generation publishes from the seeded slot");
    assert(final.generation == gpu::internal::MAX_DEVICE_GENERATION, "registration preserves the final generation");
    gpu::destroy_device(&final)!!;

    control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_RETIRED, "destruction retires the exhausted slot");
}

fn void concurrent_registry_mutation_keeps_devices_distinct() @test {
    Atomic{uint} ready;
    Atomic{bool} release;
    DeviceRegistryWorkerArgs[DEVICE_REGISTRY_THREAD_COUNT] args;
    Thread[DEVICE_REGISTRY_THREAD_COUNT] threads;

    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        args[i].ready = &ready;
        args[i].release = &release;
        threads[i].create(&device_registry_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < DEVICE_REGISTRY_THREAD_COUNT) {
        thread::yield();
    }

    gpu::Device[DEVICE_REGISTRY_THREAD_COUNT] published;
    bool all_published = true;
    bool all_distinct = true;
    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        published[i] = args[i].device;
        if (args[i].failed || !published[i].is_valid()) all_published = false;
        for (uint j = 0; j < i; j++) {
            if ((ulong)published[i] == (ulong)published[j]) all_distinct = false;
        }
    }

    release.store(true, AtomicOrdering.RELEASE);
    bool all_joined = true;
    foreach (&worker : threads) {
        if (worker.join()!! != 0) all_joined = false;
    }

    assert(all_published, "every concurrent registration succeeds");
    assert(all_distinct, "concurrent registrations publish distinct devices");
    assert(all_joined, "every concurrent destruction succeeds");
}

fn void registration_proceeds_while_another_device_is_closing() @test {
    BlockingDestroyFixture blocking_fixture;
    blocking_fixture.state.block_destroy = true;
    gpu::Device closing = gpu::internal::register_device(
        state:  &blocking_fixture.state,
        caps:   {},
        owner:  &blocking_fixture.state.owner,
    )!!;

    DestroyWorkerArgs destroy_args = { .device = closing };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!blocking_fixture.state.destroy_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    DeviceFixture survivor_fixture;
    gpu::Device survivor;
    bool survivor_created = true;
    gpu::Device? registered = gpu::internal::register_device(
        state:  &survivor_fixture.state,
        caps:   {},
    );
    if (catch registration_fault = registered) {
        (void)registration_fault;
        survivor_created = false;
    } else {
        survivor = registered;
    }

    blocking_fixture.state.destroy_release.store(true, AtomicOrdering.RELEASE);
    int destroy_result = destroyer.join()!!;

    assert(survivor_created, "registration succeeds while another slot is closing");
    assert(destroy_result == 0 && !destroy_args.failed, "the closing device finishes destruction");
    if (survivor_created) {
        (void)gpu::get_device_caps(&survivor);
        gpu::destroy_device(&survivor)!!;
    }
}

fn void live_children_take_precedence_over_active_operations() @test {
    BlockingOperationFixture fixture;
    fixture.state.block_memory_stats = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    slot.live_children.store(1, AtomicOrdering.RELEASE);

    DeviceOperationWorkerArgs operation_args = { .device = device };
    Thread operation_thread;
    operation_thread.create(&device_operation_worker, &operation_args)!!;
    while (!fixture.state.memory_stats_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    fixture.state.memory_stats_release.store(true, AtomicOrdering.RELEASE);
    int operation_result = operation_thread.join()!!;
    slot.live_children.store(0, AtomicOrdering.RELEASE);
    gpu::destroy_device(&device)!!;

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "live children take precedence over active operation pins");
    assert(operation_result == 0,
        "the active operation completes after child rejection");
}

fn void backend_busy_device_destruction_is_retryable() @test {
    BlockingOperationFixture fixture;
    fixture.state.destroy_busy = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[device.index].control.load(AtomicOrdering.ACQUIRE);

    assert(destroy_fault == gpu::DEVICE_BUSY, "incomplete backend work makes destruction retryable");
    assert(device == original, "backend busy preserves the device token");
    assert(control.state == gpu::internal::DEVICE_SLOT_LIVE, "backend busy never publishes closing state");
    assert(control.generation == original.generation, "backend busy preserves generation");
    assert(fixture.state.check_calls.load(AtomicOrdering.ACQUIRE) == 1, "the readiness check runs once");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 0, "backend busy prevents teardown");

    fixture.state.destroy_busy = false;
    gpu::destroy_device(&device)!!;
    assert(fixture.state.check_calls.load(AtomicOrdering.ACQUIRE) == 2, "the retry checks readiness again");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1, "only the ready retry tears down the backend");
}

fn void child_published_while_closing_blocks_destruction() @test {
    BlockingOperationFixture fixture;
    fixture.state.publish_child_on_check = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        owner:  &fixture.state.owner,
    )!!;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[original.index];
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)slot.control.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong live_children = slot.live_children.load(AtomicOrdering.ACQUIRE);

    if (destroy_fault == gpu::RESOURCE_IN_USE) {
        (void)slot.live_children.sub(1, AtomicOrdering.RELEASE);
        fixture.state.publish_child_on_check = false;
        gpu::destroy_device(&device)!!;
    }

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "a child published before closing completes blocks destruction");
    assert(device == gpu::DEVICE_INVALID,
        "the retry succeeds after the raced child is released");
    assert(control.state == gpu::internal::DEVICE_SLOT_LIVE,
        "the failed attempt restores the live state");
    assert(control.generation == original.generation,
        "the failed attempt preserves generation");
    assert(live_children == 1,
        "the late child publication remains owned by the live device");
}

fn void successful_device_destruction_advances_generation() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    uint index = device.index;
    uint generation = device.generation;

    gpu::destroy_device(&device)!!;

    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[index].control.load(AtomicOrdering.ACQUIRE);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "successful destruction releases the slot");
    assert(control.generation == generation + 1, "successful destruction advances generation exactly once");
}

fn void runtime_device_retains_are_atomic() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_device_registry_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;

    Atomic{uint} ready;
    Atomic{bool} release;
    RuntimeRetainWorkerArgs[DEVICE_REGISTRY_THREAD_COUNT] args;
    Thread[DEVICE_REGISTRY_THREAD_COUNT] threads;
    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        args[i].runtime = runtime;
        args[i].ready = &ready;
        args[i].release = &release;
        threads[i].create(&runtime_retain_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < DEVICE_REGISTRY_THREAD_COUNT) {
        thread::yield();
    }

    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(&runtime);
    uint retained_count = runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE);
    release.store(true, AtomicOrdering.RELEASE);

    bool all_joined = true;
    foreach (&worker : threads) {
        if (worker.join()!! != 0) all_joined = false;
    }
    uint final_count = runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE);

    assert(retained_count == DEVICE_REGISTRY_THREAD_COUNT, "concurrent retains lose no updates");
    assert(all_joined, "concurrent releases return without faults");
    assert(final_count == 0, "concurrent releases restore the runtime count");
    gpu::destroy_runtime(&runtime)!!;
}

struct CapsWorkerArgs {
    gpu::Device device;
    uint        iterations;
}

fn int caps_worker(void* raw_args) {
    CapsWorkerArgs* args = (CapsWorkerArgs*)raw_args;
    for (uint i = 0; i < args.iterations; i++) {
        (void)gpu::get_device_caps(&args.device);
    }
    return 0;
}

fn void concurrent_pins_never_terminate() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    defer gpu::destroy_device(&device)!!;
    CapsWorkerArgs[DEVICE_REGISTRY_THREAD_COUNT] args;
    Thread[DEVICE_REGISTRY_THREAD_COUNT] threads;
    foreach (i, &arg : args) {
        *arg = { .device = device, .iterations = 100_000 };
        threads[i].create(&caps_worker, arg)!!;
    }
    foreach (&worker : threads) {
        assert(worker.join()!! == 0, "concurrent device operations complete");
    }
}
