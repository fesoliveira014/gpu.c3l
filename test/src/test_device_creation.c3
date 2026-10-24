module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc device_creation_runtime_desc() {
    return {
        .application_name = "gpu_adapter_device_test",
    };
}

fn Adapter device_creation_adapter(Runtime* runtime, uint index = 0) {
    return gpu::enumerate_adapters(runtime).get(index)!!;
}

fn void public_device_desc_defaults_are_equivalent_and_reusable() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    DeviceDesc zero_desc;
    DeviceDesc original = zero_desc;

    assert(gpu::supports_device_desc(&adapter)!!.supported);
    assert(gpu::supports_device_desc(&adapter, null)!!.supported);
    assert(gpu::supports_device_desc(&adapter, &zero_desc)!!.supported);
    assert(device_desc_equal(zero_desc, original));

    Device omitted = gpu::create_device(&adapter)!!;
    Device explicit_null = gpu::create_device(&adapter, null)!!;
    Device first_zero = gpu::create_device(&adapter, &zero_desc)!!;
    Device second_zero = gpu::create_device(&adapter, &zero_desc)!!;
    assert(device_desc_equal(zero_desc, original));

    Device[4] devices = {
        omitted,
        explicit_null,
        first_zero,
        second_zero,
    };
    foreach (device : devices) {
        QueueRoles queues = gpu::get_device_caps(&device).queues;
        assert(queues.graphics);
        assert(queues.compute);
        assert(queues.transfer);
        gpu::destroy_device(&device)!!;
    }
    gpu::destroy_runtime(&runtime)!!;
}

fn void adapter_device_preserves_identity_and_runtime_lifetime() @test {
    gpu::internal::vk::stub_set_adapter_count(2);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 1);
    Device device = gpu::create_device(&adapter)!!;
    assert(
        (ulong)gpu::internal::vk::stub_last_device_adapter() == (ulong)adapter,
        "creation routes the exact borrowed adapter"
    );
    assert(
        @catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE,
        "a live adapter device retains its runtime"
    );

    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
    assert(
        gpu::internal::vk::stub_device_temporary_create_count()
            == gpu::internal::vk::stub_device_temporary_destroy_count(),
        "destruction releases backend state"
    );
}

fn void distinct_adapter_devices_preserve_exact_owners() @test {
    gpu::internal::vk::stub_set_adapter_count(2);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter first_adapter = device_creation_adapter(&runtime, 0);
    Adapter second_adapter = device_creation_adapter(&runtime, 1);
    Device first = gpu::create_device(&first_adapter)!!;
    Device second = gpu::create_device(&second_adapter)!!;

    assert(
        (ulong)gpu::internal::vk::stub_device_adapter(&first)!! == (ulong)first_adapter,
        "the first device keeps the first exact adapter"
    );
    assert(
        (ulong)gpu::internal::vk::stub_device_adapter(&second)!! == (ulong)second_adapter,
        "the second device keeps the second exact adapter"
    );

    gpu::destroy_device(&first)!!;
    assert(
        (ulong)gpu::internal::vk::stub_device_adapter(&second)!! == (ulong)second_adapter,
        "destroying one adapter device preserves its peer ownership"
    );
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
    assert(
        gpu::internal::vk::stub_device_temporary_create_count()
            == gpu::internal::vk::stub_device_temporary_destroy_count(),
        "both adapter-owned backend states are released"
    );
}

fn AllocationDesc device_creation_allocation_desc() {
    return {
        .size         = 256,
        .alignment    = 16,
        .memory_class = MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
    };
}

fn void foreign_allocation_owner_rejects_before_backend_mutation() @test {
    gpu::internal::vk::stub_reset_allocation_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 0);
    Device first = gpu::create_device(&adapter)!!;
    Device second = gpu::create_device(&adapter)!!;
    AllocationDesc allocation_desc = device_creation_allocation_desc();
    GpuAllocation first_allocation = gpu::allocate_memory(&first, &allocation_desc)!!;
    GpuAllocation second_allocation = gpu::allocate_memory(&second, &allocation_desc)!!;

    assert(first_allocation.index == second_allocation.index,
        "stub devices use coincident local slots");
    assert(first_allocation.generation == second_allocation.generation,
        "stub devices use coincident local generations");
    assert(first_allocation.owner != second_allocation.owner,
        "coincident allocations retain distinct owners");
    GpuAllocation foreign = first_allocation;
    uint before = gpu::internal::vk::stub_allocation_free_attempt_count();
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == before,
        "foreign ownership rejects before backend mutation");

    gpu::free_allocation(&first, &first_allocation)!!;
    gpu::free_allocation(&second, &second_allocation)!!;
    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void stale_child_allocation_rejects_after_device_slot_reuse() @test {
    gpu::internal::vk::stub_reset_allocation_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 0);
    AllocationDesc allocation_desc = device_creation_allocation_desc();

    Device old_device = gpu::create_device(&adapter)!!;
    GpuAllocation old_allocation = gpu::allocate_memory(&old_device, &allocation_desc)!!;
    GpuAllocation stale_allocation = old_allocation;
    uint old_index = old_device.index;
    uint old_generation = old_device.generation;
    gpu::free_allocation(&old_device, &old_allocation)!!;
    gpu::destroy_device(&old_device)!!;

    Device replacement = gpu::create_device(&adapter)!!;
    GpuAllocation replacement_allocation =
        gpu::allocate_memory(&replacement, &allocation_desc)!!;
    assert(replacement.index == old_index, "freed registry slot is reused");
    assert(replacement.generation != old_generation,
        "replacement has a distinct device generation");
    assert(replacement_allocation.index == stale_allocation.index,
        "replacement reuses the same child slot");
    assert(replacement_allocation.generation == stale_allocation.generation,
        "replacement reuses the same child generation");
    assert(replacement_allocation.owner != stale_allocation.owner,
        "replacement has a distinct child owner");
    uint before = gpu::internal::vk::stub_allocation_free_attempt_count();
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == before,
        "stale child owner rejects before the new tenant backend mutates");
    gpu::free_allocation(&replacement, &replacement_allocation)!!;
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == before + 1,
        "rejection preserves the replacement child");

    gpu::destroy_device(&replacement)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void unsupported_and_stale_adapter_creation_faults_before_allocation() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    assert(
        @catch(gpu::create_device(&adapter)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported device baseline faults without fallback"
    );
    assert(gpu::internal::vk::stub_device_temporary_create_count() == 0, "unsupported requests allocate nothing");

    gpu::destroy_runtime(&runtime)!!;
    assert(gpu::internal::vk::stub_device_temporary_create_count() == 0, "stale adapters allocate nothing");
}

fn void device_creation_failure_rolls_back_runtime_and_backend_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    gpu::internal::vk::stub_fail_device_after_temporary_allocation(true);
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    assert(
        @catch(gpu::create_device(&adapter)) == gpu::BACKEND_ERROR,
        "failure after temporary allocation publishes no device"
    );
    assert(gpu::internal::device_slot_available(), "failed creation leaves the device slot available");
    assert(
        gpu::internal::vk::stub_device_temporary_create_count()
            == gpu::internal::vk::stub_device_temporary_destroy_count(),
        "failed creation releases temporary backend state"
    );

    gpu::internal::vk::stub_fail_device_after_temporary_allocation(false);
    gpu::destroy_runtime(&runtime)!!;
}
