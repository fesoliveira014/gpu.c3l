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

    control.generation = gpu::MAX_DEVICE_GENERATION;
    slot.control.store((ulong)control, AtomicOrdering.RELAXED);

    FrameFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    defer gpu::destroy_device(&device)!!;

    control = (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(device.index != 0, "an exhausted slot is not reused");
    assert(control.state == gpu::DEVICE_SLOT_RETIRED, "an exhausted slot is retired");
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
