module gpu_test;

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
    Atomic{bool} entered;
    Atomic{bool} release;
}

struct BlockingOperationFixture {
    Atomic{bool} entered;
    Atomic{bool} release;
    Atomic{bool} check_entered;
    Atomic{bool} check_release;
    Atomic{uint} check_calls;
    Atomic{uint} destroy_calls;
    gpu::Device  device;
    bool         block_check;
    bool         destroy_busy;
    bool         publish_child_on_check;
}

struct CommandPinFixture {
    Atomic{bool} command_entered;
    Atomic{bool} command_release;
    Atomic{uint} destroy_calls;
    uint         pins_during_command;
    uint         pins_during_submit;
    uint         next_command;
    bool         block_command;
    bool         preserve_discarded_token;
    bool         reject_command;
    bool         reuse_command_index;
    bool         submit_busy;
}

struct StaleCommandWorkerArgs {
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
    bool        failed;
}

struct RuntimeRetainWorkerArgs {
    gpu::Runtime  runtime;
    Atomic{uint}* ready;
    Atomic{bool}* release;
    bool          failed;
}

fn void blocking_device_destroy(void* backend_state) {
    BlockingDestroyFixture* fixture = (BlockingDestroyFixture*)backend_state;
    fixture.entered.store(true, AtomicOrdering.RELEASE);
    while (!fixture.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn void? check_counted_device_destroy(void* backend_state) {
    BlockingOperationFixture* fixture = (BlockingOperationFixture*)backend_state;
    (void)fixture.check_calls.add(1, AtomicOrdering.RELEASE);
    fixture.check_entered.store(true, AtomicOrdering.RELEASE);
    while (fixture.block_check
        && !fixture.check_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (fixture.publish_child_on_check) {
        (void)gpu::device_slots[fixture.device.index].live_children.add(
            1,
            AtomicOrdering.RELEASE,
        );
    }
    if (fixture.destroy_busy) return gpu::DEVICE_BUSY~;
}

fn void counted_device_destroy(void* backend_state) {
    BlockingOperationFixture* fixture = (BlockingOperationFixture*)backend_state;
    (void)fixture.destroy_calls.add(1, AtomicOrdering.RELEASE);
}

fn void command_pin_device_destroy(void* backend_state) {
    CommandPinFixture* fixture = (CommandPinFixture*)backend_state;
    (void)fixture.destroy_calls.add(1, AtomicOrdering.RELEASE);
}

fn void? command_pin_begin(
    gpu::Queue queue,
    gpu::CommandList* out_commands,
) {
    gpu::Device device = queue.device;
    gpu::DeviceData* data = gpu::resolve_pinned_device(&device);
    if (data == null || out_commands == null) return gpu::INVALID_HANDLE~;
    CommandPinFixture* fixture = (CommandPinFixture*)data.backend_state;
    fixture.next_command++;
    *out_commands = {
        .device = device,
        .handle = {
            .owner      = gpu::resource_owner_tag(
                data.resource_owner,
                gpu::ResourceOwnerKind.COMMAND,
            ),
            .index      = fixture.reuse_command_index ? 1 : fixture.next_command,
            .generation = fixture.reuse_command_index ? fixture.next_command : 1,
        },
    };
}

fn void? command_pin_end(gpu::CommandList* commands) {}

fn void? command_pin_discard(
    gpu::Device* device,
    gpu::CommandListHandle handle,
    bool executable,
) {
    if (device == null || !handle.is_valid()) return gpu::INVALID_HANDLE~;
    CommandPinFixture* fixture =
        (CommandPinFixture*)gpu::device_backend_state_ptr(device);
    if (fixture == null) return gpu::INVALID_HANDLE~;
}

fn void? command_pin_barrier(
    gpu::CommandList* commands,
    gpu::Barrier* barrier,
) {
    CommandPinFixture* fixture =
        (CommandPinFixture*)gpu::device_backend_state_ptr(&commands.device);
    if (fixture == null) return gpu::INVALID_HANDLE~;
    gpu::DeviceSlotControl control =
        (gpu::DeviceSlotControl)gpu::device_slots[commands.device.index].control.load(
            AtomicOrdering.ACQUIRE,
        );
    fixture.pins_during_command = control.pins;
    fixture.command_entered.store(true, AtomicOrdering.RELEASE);
    while (fixture.block_command
        && !fixture.command_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (fixture.reject_command) return gpu::INVALID_HANDLE~;
}

fn gpu::CompletionPoint? command_pin_submit(
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
) {
    gpu::Device device = queue.device;
    CommandPinFixture* fixture =
        (CommandPinFixture*)gpu::device_backend_state_ptr(&device);
    if (fixture == null) return gpu::INVALID_HANDLE~;
    gpu::DeviceSlotControl control =
        (gpu::DeviceSlotControl)gpu::device_slots[device.index].control.load(
            AtomicOrdering.ACQUIRE,
        );
    fixture.pins_during_submit = control.pins;
    if (fixture.submit_busy) return gpu::DEVICE_BUSY~;
    return {
        .device  = device,
        .payload = 1ull << 10,
    };
}

fn gpu::MemoryStats? blocking_get_memory_stats(gpu::Device* device) {
    BlockingOperationFixture* fixture = (BlockingOperationFixture*)gpu::device_backend_state_ptr(device);
    if (fixture == null) return gpu::INVALID_HANDLE~;
    fixture.entered.store(true, AtomicOrdering.RELEASE);
    while (!fixture.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    return {};
}

gpu::BackendVTable blocking_device_vtable = {
    .destroy_device = &blocking_device_destroy,
};

gpu::BackendVTable blocking_operation_vtable = {
    .check_device_destroy = &check_counted_device_destroy,
    .destroy_device       = &counted_device_destroy,
    .get_memory_stats     = &blocking_get_memory_stats,
};

gpu::BackendVTable command_pin_vtable = {
    .destroy_device     = &command_pin_device_destroy,
    .begin_commands     = &command_pin_begin,
    .end_commands       = &command_pin_end,
    .discard_commands   = &command_pin_discard,
    .submit             = &command_pin_submit,
    .cmd_barrier        = &command_pin_barrier,
};

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
    if (catch operation_fault = gpu::get_memory_stats(&args.device)) {
        (void)operation_fault;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int stale_command_worker(void* raw_args) {
    StaleCommandWorkerArgs* args = (StaleCommandWorkerArgs*)raw_args;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    if (catch command_fault = gpu::cmd_barrier(
        &args.commands,
        &barrier,
    )) {
        args.result = command_fault;
    } else {
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int runtime_retain_worker(void* raw_args) {
    RuntimeRetainWorkerArgs* args = (RuntimeRetainWorkerArgs*)raw_args;
    bool retained = true;
    if (catch retain_fault = gpu::runtime_retain_device(&args.runtime)) {
        (void)retain_fault;
        args.failed = true;
        retained = false;
    }

    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (retained) {
        if (catch release_fault = gpu::runtime_release_device(&args.runtime)) {
            (void)release_fault;
            args.failed = true;
        }
    }
    return args.failed ? 1 : 0;
}

fn int device_registry_worker(void* raw_args) {
    DeviceRegistryWorkerArgs* args = (DeviceRegistryWorkerArgs*)raw_args;
    gpu::Device? registered = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &device_fixture_vtable,
        backend_state: &args.fixture,
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

fn void command_calls_borrow_the_lifetime_pin() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];
    gpu::DeviceSlotControl after_begin =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::DeviceSlotControl after_end =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    gpu::discard_executable_commands(&executable)!!;
    gpu::DeviceSlotControl after_discard =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(after_begin.pins == 1,
        "begin transfers its operation pin to the recording token");
    assert(fixture.pins_during_command == 1,
        "hot command recording borrows without acquiring another pin");
    assert(after_end.pins == 1,
        "the executable token retains the same lifetime pin");
    assert(after_discard.pins == 0,
        "discard releases the command lifetime pin exactly once");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases the command child exactly once");
    gpu::destroy_device(&device)!!;
}

fn void failed_submit_preserves_command_lifetime_pins() @test {
    CommandPinFixture fixture = { .submit_busy = true };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList first = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::CommandList second = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::ExecutableCommandList[2] lists = {
        first_executable,
        second_executable,
    };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    fault submit_fault = @catch(gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!, &desc));
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];
    gpu::DeviceSlotControl after_failure =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    ulong children_after_failure =
        slot.live_children.load(AtomicOrdering.ACQUIRE);

    fixture.submit_busy = false;
    gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!, &desc)!!;
    gpu::DeviceSlotControl after_success =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(submit_fault == gpu::DEVICE_BUSY,
        "retryable submit failure preserves executable tokens");
    assert(fixture.pins_during_submit == 3,
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

fn void duplicate_backend_discard_cannot_underflow_command_accounting() @test {
    CommandPinFixture fixture = { .preserve_discarded_token = true };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::CommandList duplicate = commands;
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];

    gpu::discard_commands(&commands)!!;
    gpu::discard_commands(&duplicate)!!;
    gpu::DeviceSlotControl after_duplicate =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(after_duplicate.pins == 0,
        "a duplicate backend discard cannot underflow retained pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "a duplicate backend discard cannot underflow command children");
    gpu::destroy_device(&device)!!;
}

fn void lost_commands_can_be_discarded() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::mark_device_lost(&device);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    fault record_fault = @catch(gpu::cmd_barrier(&commands, &barrier));
    fault destroy_fault = @catch(gpu::destroy_device(&device));
    uint destroy_calls_before_discard =
        fixture.destroy_calls.load(AtomicOrdering.ACQUIRE);
    gpu::discard_commands(&commands)!!;
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];
    gpu::DeviceSlotControl after_discard =
        (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(record_fault == gpu::DEVICE_LOST,
        "lost command recording faults without consuming its token");
    assert(destroy_fault == gpu::DEVICE_BUSY,
        "a lost device remains busy while a command retains its lifetime pin");
    assert(destroy_calls_before_discard == 0,
        "device teardown does not begin before the lost command is discarded");
    assert(after_discard.pins == 0,
        "discard releases a lost command lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases a lost command child");
    gpu::destroy_device(&device)!!;
}

fn void stale_commands_cannot_borrow_unrelated_lifetime_pins() @test {
    CommandPinFixture fixture = {
        .block_command  = true,
        .reject_command = true,
    };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList discarded = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::CommandList stale = discarded;
    gpu::discard_commands(&discarded)!!;
    gpu::CommandList live = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;

    StaleCommandWorkerArgs worker_args = { .commands = stale };
    Thread worker;
    worker.create(&stale_command_worker, &worker_args)!!;
    while (!fixture.command_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    gpu::discard_commands(&live)!!;
    fault destroy_fault = @catch(gpu::destroy_device(&device));
    uint destroy_calls_before_release =
        fixture.destroy_calls.load(AtomicOrdering.ACQUIRE);

    fixture.command_release.store(true, AtomicOrdering.RELEASE);
    int worker_result = worker.join()!!;
    gpu::destroy_device(&device)!!;

    assert(fixture.pins_during_command == 2,
        "a stale command takes a short pin instead of borrowing a live command pin");
    assert(destroy_fault == gpu::DEVICE_BUSY,
        "the stale-command validation pin prevents concurrent device reclamation");
    assert(destroy_calls_before_release == 0,
        "device state remains alive while stale-command validation is active");
    assert(worker_result == 0 && !worker_args.failed,
        "the stale command reports its backend validation fault");
    assert(worker_args.result == gpu::INVALID_HANDLE,
        "the stale command remains invalid");
    assert(fixture.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1,
        "destruction succeeds once the stale-command validation pin retires");
}

fn void forged_command_owners_cannot_borrow_lifetime_pins() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::CommandList forged = commands;
    forged.handle.owner++;

    bool real_borrows = gpu::has_retained_command_pin(&commands);
    bool forged_borrows = gpu::has_retained_command_pin(&forged);
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    fault forged_fault = @catch(gpu::cmd_barrier(&forged, &barrier));

    assert(real_borrows, "the exact command owner borrows its lifetime pin");
    assert(!forged_borrows,
        "a forged owner cannot borrow another command's lifetime pin");
    assert(forged_fault == gpu::INVALID_HANDLE,
        "a forged command owner faults before backend access");
    assert(!fixture.command_entered.load(AtomicOrdering.ACQUIRE),
        "a forged command owner never reaches the backend");
    gpu::discard_commands(&commands)!!;
    gpu::destroy_device(&device)!!;
}

fn void stale_command_generation_cannot_borrow_reused_slot_pin() @test {
    CommandPinFixture fixture = {
        .reject_command      = true,
        .reuse_command_index = true,
    };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList first = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::CommandList stale = first;
    gpu::discard_commands(&first)!!;
    gpu::CommandList live = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;

    bool stale_borrows = gpu::has_retained_command_pin(&stale);
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    fault stale_fault = @catch(gpu::cmd_barrier(&stale, &barrier));

    assert(stale.handle.index == live.handle.index,
        "the command slot is reused");
    assert(stale.handle.generation != live.handle.generation,
        "the reused command slot advances its generation");
    assert(!stale_borrows,
        "a stale generation cannot borrow the reused slot's lifetime pin");
    assert(fixture.pins_during_command == 2,
        "stale-generation validation takes a short pin beside the live lease");
    assert(stale_fault == gpu::INVALID_HANDLE,
        "the stale generation remains invalid");
    gpu::discard_commands(&live)!!;
    gpu::destroy_device(&device)!!;
}

fn void malformed_command_lease_index_is_rejected() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList malformed = {
        .device = device,
        .handle = {
            .index      = gpu::MAX_DEVICE_COMMANDS,
            .generation = 1,
        },
    };
    gpu::DeviceOperation operation = gpu::begin_device_operation(&device)!!;
    bool retained = gpu::retain_command_pin_lease(&operation, &malformed);

    assert(!retained, "an out-of-range command index cannot publish a lease");
    gpu::end_device_operation(&operation);
    gpu::destroy_device(&device)!!;
}

fn void stale_lease_release_preserves_a_reused_command_slot() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &command_pin_vtable,
        backend_state: &fixture,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];
    uint next_generation = commands.handle.generation + 1;
    slot.command_leases[commands.handle.index].store(
        next_generation,
        AtomicOrdering.RELEASE,
    );

    gpu::DeviceOperation operation = gpu::begin_device_operation(&device)!!;
    gpu::release_command_pin_lease(&operation, &commands);
    uint retained_generation = slot.command_leases[commands.handle.index].load(
        AtomicOrdering.ACQUIRE,
    );
    gpu::end_device_operation(&operation);

    assert(retained_generation == next_generation,
        "an old release cannot clear a reused command slot's lease");
    slot.command_leases[commands.handle.index].store(
        commands.handle.generation,
        AtomicOrdering.RELEASE,
    );
    gpu::discard_commands(&commands)!!;
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

fn void forged_device_tokens_are_rejected() @test {
    gpu::Device zero_generation = {
        .index      = 7,
        .generation = 0,
    };
    assert(
        @catch(gpu::get_device_caps(&zero_generation)) == gpu::INVALID_HANDLE,
        "zero-generation device tokens are rejected"
    );

    gpu::Device reserved = {
        .index      = 0,
        .generation = 1,
        .reserved   = 1,
    };
    assert(
        @catch(gpu::get_device_caps(&reserved)) == gpu::INVALID_HANDLE,
        "reserved device bits are rejected"
    );

    gpu::Device out_of_range = {
        .index      = gpu::MAX_DEVICES,
        .generation = 1,
    };
    assert(
        @catch(gpu::get_device_caps(&out_of_range)) == gpu::INVALID_HANDLE,
        "out-of-range device slots are rejected"
    );
}

fn void device_loss_isolated_to_marked_slot() @test {
    DeviceFixture first_fixture;
    DeviceFixture second_fixture;
    gpu::Device first = fixture_device(&first_fixture);
    gpu::Device second = fixture_device(&second_fixture);

    gpu::mark_device_lost(&first);

    assert(
        @catch(gpu::get_device_caps(&first)) == gpu::DEVICE_LOST,
        "the affected device stays lost"
    );
    assert(
        gpu::get_device_backend(&second)!! == gpu::BackendKind.VULKAN,
        "a peer device remains usable"
    );

    gpu::Device stale = first;
    gpu::device_slots[first.index].live_children.store(1, AtomicOrdering.RELAXED);
    gpu::destroy_device(&first)!!;
    DeviceFixture replacement_fixture;
    gpu::Device replacement = fixture_device(&replacement_fixture);
    gpu::mark_device_lost(&stale);
    assert(
        gpu::get_device_backend(&replacement)!! == gpu::BackendKind.VULKAN,
        "slot reuse clears lost state"
    );

    gpu::destroy_device(&second)!!;
    gpu::destroy_device(&replacement)!!;
}

fn void exhausted_device_slots_are_retired() @test {
    gpu::DeviceSlot* slot = &gpu::device_slots[0];
    gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::DEVICE_SLOT_FREE, "the exhaustion fixture starts from a free slot");

    control.generation = gpu::MAX_DEVICE_GENERATION - 1;
    slot.control.store((ulong)control, AtomicOrdering.RELAXED);

    DeviceFixture penultimate_fixture;
    gpu::Device penultimate = fixture_device(&penultimate_fixture);
    assert(penultimate.index == 0, "the seeded slot publishes its current generation");
    assert(penultimate.generation == gpu::MAX_DEVICE_GENERATION - 1, "registration does not advance generation");
    gpu::destroy_device(&penultimate)!!;

    control = (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::DEVICE_SLOT_FREE, "the final generation remains publishable");
    assert(control.generation == gpu::MAX_DEVICE_GENERATION, "successful destruction advances to the final generation");

    DeviceFixture final_fixture;
    gpu::Device final = fixture_device(&final_fixture);
    assert(final.index == 0, "the final generation publishes from the seeded slot");
    assert(final.generation == gpu::MAX_DEVICE_GENERATION, "registration preserves the final generation");
    gpu::destroy_device(&final)!!;

    control = (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::DEVICE_SLOT_RETIRED, "destruction retires the exhausted slot");
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
    foreach (&stale : published) {
        assert(
            @catch(gpu::get_device_caps(stale)) == gpu::INVALID_HANDLE,
            "destroyed concurrent devices become stale"
        );
    }
}

fn void registration_proceeds_while_another_device_is_closing() @test {
    BlockingDestroyFixture blocking_fixture;
    gpu::Device closing = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_device_vtable,
        backend_state: &blocking_fixture,
    )!!;
    gpu::Device stale = closing;
    gpu::Queue closing_queue = {
        .device = stale,
        .id     = 0,
        .roles  = { .graphics },
    };
    gpu::CompletionPoint closing_point = gpu::make_completion_point(
        closing_queue,
        1,
    )!!;

    DestroyWorkerArgs destroy_args = { .device = closing };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!blocking_fixture.entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    fault closing_fault = @catch(gpu::get_device_caps(&stale));
    fault poll_fault = @catch(gpu::poll_completion(closing_point));

    gpu::Device concurrent_destroy = stale;
    fault concurrent_destroy_fault = @catch(gpu::destroy_device(&concurrent_destroy));
    bool concurrent_destroy_preserved = concurrent_destroy.is_valid();

    DeviceFixture survivor_fixture;
    gpu::Device survivor;
    bool survivor_created = true;
    gpu::Device? registered = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &device_fixture_vtable,
        backend_state: &survivor_fixture,
    );
    if (catch registration_fault = registered) {
        (void)registration_fault;
        survivor_created = false;
    } else {
        survivor = registered;
    }

    blocking_fixture.release.store(true, AtomicOrdering.RELEASE);
    int destroy_result = destroyer.join()!!;

    assert(survivor_created, "registration succeeds while another slot is closing");
    assert(destroy_result == 0 && !destroy_args.failed, "the closing device finishes destruction");
    assert(closing_fault == gpu::DEVICE_BUSY, "operations reject a closing device with DEVICE_BUSY");
    assert(poll_fault == gpu::DEVICE_BUSY,
        "completion polling rejects a closing device with DEVICE_BUSY");

    assert(concurrent_destroy_fault == gpu::DEVICE_BUSY, "a second destruction attempt is retryable");
    assert(concurrent_destroy_preserved, "a busy destruction attempt preserves its token");
    assert(
        @catch(gpu::get_device_caps(&stale)) == gpu::INVALID_HANDLE,
        "the destroyed device remains stale"
    );
    if (survivor_created) {
        assert(gpu::get_device_backend(&survivor)!! == gpu::BackendKind.VULKAN, "the survivor remains resolvable");
        gpu::destroy_device(&survivor)!!;
    }
}

fn void active_operations_make_device_destruction_retryable() @test {
    BlockingOperationFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_operation_vtable,
        backend_state: &fixture,
    )!!;

    DeviceOperationWorkerArgs operation_args = { .device = device };
    Thread operation_thread;
    operation_thread.create(&device_operation_worker, &operation_args)!!;
    while (!fixture.entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    bool handle_preserved = device.is_valid();
    uint calls_before_retry = fixture.destroy_calls.load(AtomicOrdering.ACQUIRE);

    fixture.release.store(true, AtomicOrdering.RELEASE);
    int operation_result = operation_thread.join()!!;

    bool retry_succeeded = false;
    if (device.is_valid()) {
        if (catch retry_fault = gpu::destroy_device(&device)) {
            (void)retry_fault;
        } else {
            retry_succeeded = true;
        }
    }

    assert(destroy_fault == gpu::DEVICE_BUSY, "active operations make destruction retryable");
    assert(handle_preserved, "a busy destruction attempt preserves the device token");
    assert(calls_before_retry == 0, "a busy destruction attempt does not tear down the backend");
    assert(operation_result == 0 && !operation_args.failed, "the pinned operation completes normally");
    assert(retry_succeeded, "destruction succeeds after the active operation releases its pin");
    assert(fixture.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1, "only the successful retry tears down the backend");
}

fn void live_children_take_precedence_over_active_operations() @test {
    BlockingOperationFixture fixture;
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_operation_vtable,
        backend_state: &fixture,
    )!!;
    gpu::DeviceSlot* slot = &gpu::device_slots[device.index];
    slot.live_children.store(1, AtomicOrdering.RELEASE);

    DeviceOperationWorkerArgs operation_args = { .device = device };
    Thread operation_thread;
    operation_thread.create(&device_operation_worker, &operation_args)!!;
    while (!fixture.entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    fixture.release.store(true, AtomicOrdering.RELEASE);
    int operation_result = operation_thread.join()!!;
    slot.live_children.store(0, AtomicOrdering.RELEASE);
    gpu::destroy_device(&device)!!;

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "live children take precedence over active operation pins");
    assert(operation_result == 0 && !operation_args.failed,
        "the active operation completes after child rejection");
}

fn void backend_busy_device_destruction_is_retryable() @test {
    BlockingOperationFixture fixture = { .destroy_busy = true };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_operation_vtable,
        backend_state: &fixture,
    )!!;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)gpu::device_slots[device.index].control.load(AtomicOrdering.ACQUIRE);

    assert(destroy_fault == gpu::DEVICE_BUSY, "incomplete backend work makes destruction retryable");
    assert(device == original, "backend busy preserves the device token");
    assert(control.state == gpu::DEVICE_SLOT_LIVE, "backend busy never publishes closing state");
    assert(control.generation == original.generation, "backend busy preserves generation");
    assert(fixture.check_calls.load(AtomicOrdering.ACQUIRE) == 1, "the readiness check runs once");
    assert(fixture.destroy_calls.load(AtomicOrdering.ACQUIRE) == 0, "backend busy prevents teardown");

    fixture.destroy_busy = false;
    gpu::destroy_device(&device)!!;
    assert(fixture.check_calls.load(AtomicOrdering.ACQUIRE) == 2, "the retry checks readiness again");
    assert(fixture.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1, "only the ready retry tears down the backend");
}

fn void readiness_check_runs_after_new_operations_are_closed() @test {
    BlockingOperationFixture fixture = { .block_check = true };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_operation_vtable,
        backend_state: &fixture,
    )!!;
    gpu::Device stale = device;

    DestroyWorkerArgs destroy_args = { .device = device };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!fixture.check_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault operation_fault = @catch(gpu::get_device_caps(&stale));
    fixture.check_release.store(true, AtomicOrdering.RELEASE);
    int destroy_result = destroyer.join()!!;

    assert(operation_fault == gpu::DEVICE_BUSY,
        "the readiness check runs only after new operations are closed");
    assert(destroy_result == 0 && !destroy_args.failed,
        "destruction completes after the readiness check");
    assert(fixture.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1,
        "the backend tears down once after readiness succeeds");
}

fn void child_published_while_closing_blocks_destruction() @test {
    BlockingOperationFixture fixture = { .publish_child_on_check = true };
    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &blocking_operation_vtable,
        backend_state: &fixture,
    )!!;
    fixture.device = device;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::DeviceSlot* slot = &gpu::device_slots[original.index];
    gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)slot.control.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong live_children = slot.live_children.load(AtomicOrdering.ACQUIRE);

    if (destroy_fault == gpu::RESOURCE_IN_USE) {
        (void)slot.live_children.sub(1, AtomicOrdering.RELEASE);
        fixture.publish_child_on_check = false;
        gpu::destroy_device(&device)!!;
    }

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "a child published before closing completes blocks destruction");
    assert(device == gpu::DEVICE_INVALID,
        "the retry succeeds after the raced child is released");
    assert(control.state == gpu::DEVICE_SLOT_LIVE,
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

    gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)gpu::device_slots[index].control.load(AtomicOrdering.ACQUIRE);
    assert(control.state == gpu::DEVICE_SLOT_FREE, "successful destruction releases the slot");
    assert(control.generation == generation + 1, "successful destruction advances generation exactly once");
}

fn void runtime_device_retains_are_atomic() @test {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
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

    gpu::RuntimeData* runtime_data = gpu::resolve_runtime(&runtime);
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
