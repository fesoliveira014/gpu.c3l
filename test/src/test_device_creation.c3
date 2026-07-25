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
    return gpu::enumerate_adapters(runtime)!!.get(index)!!;
}

fn void adapter_device_preserves_identity_and_runtime_lifetime() @test {
    gpu::internal::vk::stub_set_adapter_count(2);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 1);
    DeviceRequest request = gpu::strict_device_request();

    Device device = gpu::create_device(&adapter, &request)!!;
    assert(
        (ulong)gpu::internal::vk::stub_last_device_adapter() == (ulong)adapter,
        "creation routes the exact borrowed adapter"
    );
    assert(gpu::get_device_caps(&device)!!.strict_enabled, "strict state is enabled");
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
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter first_adapter = device_creation_adapter(&runtime, 0);
    Adapter second_adapter = device_creation_adapter(&runtime, 1);
    DeviceRequest request = gpu::strict_device_request();

    Device first = gpu::create_device(&first_adapter, &request)!!;
    Device second = gpu::create_device(&second_adapter, &request)!!;

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
    DeviceRequest request = gpu::strict_device_request();
    Device first = gpu::create_device(&adapter, &request)!!;
    Device second = gpu::create_device(&adapter, &request)!!;
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
    assert(@catch(gpu::free_allocation(&second, &foreign)) == gpu::INVALID_HANDLE);
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
    DeviceRequest request = gpu::strict_device_request();
    AllocationDesc allocation_desc = device_creation_allocation_desc();

    Device old_device = gpu::create_device(&adapter, &request)!!;
    GpuAllocation old_allocation = gpu::allocate_memory(&old_device, &allocation_desc)!!;
    GpuAllocation stale_allocation = old_allocation;
    uint old_index = old_device.index;
    uint old_generation = old_device.generation;
    gpu::free_allocation(&old_device, &old_allocation)!!;
    gpu::destroy_device(&old_device)!!;

    Device replacement = gpu::create_device(&adapter, &request)!!;
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
    assert(@catch(gpu::free_allocation(&replacement, &stale_allocation))
        == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == before,
        "stale child owner rejects before the new tenant backend mutates");
    gpu::free_allocation(&replacement, &replacement_allocation)!!;
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == before + 1,
        "rejection preserves the replacement child");

    gpu::destroy_device(&replacement)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void explicit_cross_kind_token_rejects_before_backend_mutation() @test {
    gpu::internal::vk::stub_reset_allocation_counts();
    gpu::internal::vk::stub_reset_texture_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 0);
    DeviceRequest request = gpu::strict_device_request();
    Device device = gpu::create_device(&adapter, &request)!!;
    AllocationDesc allocation_desc = device_creation_allocation_desc();
    TextureDesc texture_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
    GpuAllocation allocation = gpu::allocate_memory(&device, &allocation_desc)!!;
    TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    TextureHandle forged_texture = {
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
    };
    GpuAllocation forged_allocation = {
        .owner      = texture.owner,
        .index      = texture.index,
        .generation = texture.generation,
    };

    uint texture_before = gpu::internal::vk::stub_texture_destroy_attempt_count();
    assert(@catch(gpu::destroy_texture(&device, forged_texture)) == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_texture_destroy_attempt_count() == texture_before,
        "allocation owner bits cannot mutate the texture table");

    uint allocation_before = gpu::internal::vk::stub_allocation_free_attempt_count();
    assert(@catch(gpu::free_allocation(&device, &forged_allocation))
        == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_allocation_free_attempt_count() == allocation_before,
        "texture owner bits cannot mutate the allocation table");

    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void unsupported_and_stale_adapter_creation_faults_before_allocation() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported strict creation faults without fallback"
    );
    assert(gpu::internal::vk::stub_device_temporary_create_count() == 0, "unsupported requests allocate nothing");

    gpu::destroy_runtime(&runtime)!!;
    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::INVALID_HANDLE,
        "a stale adapter faults before device work"
    );
    assert(gpu::internal::vk::stub_device_temporary_create_count() == 0, "stale adapters allocate nothing");
}

fn void device_creation_failure_rolls_back_runtime_and_backend_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    gpu::internal::vk::stub_fail_device_after_temporary_allocation(true);
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::BACKEND_ERROR,
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

fn void invalid_retained_runtime_metadata_is_rejected_before_publication() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    Adapter forged_adapter = adapter;
    forged_adapter.runtime_generation++;
    DeviceFixture fixture;
    Device owner;

    gpu::internal::runtime_retain_device(&runtime)!!;
    Device? result = gpu::internal::register_device(
        state:            &fixture.state,
        caps:             {},
        vtable:           &device_fixture_vtable,
        retained_runtime: &runtime,
        adapter:          &forged_adapter,
        owner:            &owner,
    );

    assert(@catch(result) == gpu::INVALID_HANDLE);
    assert(gpu::internal::device_slot_available(), "invalid ownership metadata publishes no device");
    assert((ulong)owner == 0, "invalid ownership metadata writes no backend owner");
    gpu::internal::runtime_release_device(&runtime)!!;
    gpu::destroy_runtime(&runtime)!!;
}
