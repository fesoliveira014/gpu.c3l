module gpu_test;

import gpu @public;
import std::atomic;
import std::thread;

const uint DEVICE_REGISTRY_THREAD_COUNT = 8;

struct DeviceRegistryWorkerArgs {
    Atomic{uint}* ready;
    Atomic{bool}* release;
    FrameFixture  fixture;
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
    Atomic{uint} destroy_calls;
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

fn void counted_device_destroy(void* backend_state) {
    BlockingOperationFixture* fixture = (BlockingOperationFixture*)backend_state;
    (void)fixture.destroy_calls.add(1, AtomicOrdering.RELEASE);
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
    .destroy_device   = &counted_device_destroy,
    .get_memory_stats = &blocking_get_memory_stats,
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
        vtable:        &frame_fixture_vtable,
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

fn void exhausted_device_slots_are_retired() @test {
    gpu::DeviceSlot* slot = &gpu::device_slots[0];
    gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::DEVICE_SLOT_FREE, "the exhaustion fixture starts from a free slot");

    control.generation = gpu::MAX_DEVICE_GENERATION - 1;
    slot.control.store((ulong)control, AtomicOrdering.RELAXED);

    FrameFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    assert(device.index == 0, "the final live generation publishes from the seeded slot");
    assert(device.generation == gpu::MAX_DEVICE_GENERATION, "registration reaches the final generation naturally");

    gpu::destroy_device(&device)!!;

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

    DestroyWorkerArgs destroy_args = { .device = closing };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!blocking_fixture.entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    fault closing_fault = @catch(gpu::get_device_caps(&stale));
    gpu::ReadbackTicket ticket;
    fault poll_fault = @catch(gpu::poll_readback(&stale, &ticket));
    gpu::Device concurrent_destroy = stale;
    fault concurrent_destroy_fault = @catch(gpu::destroy_device(&concurrent_destroy));
    bool concurrent_destroy_preserved = concurrent_destroy.is_valid();

    FrameFixture survivor_fixture;
    gpu::Device survivor;
    bool survivor_created = true;
    gpu::Device? registered = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &frame_fixture_vtable,
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
    assert(poll_fault == gpu::DEVICE_BUSY, "non-blocking queries report a closing device");
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
