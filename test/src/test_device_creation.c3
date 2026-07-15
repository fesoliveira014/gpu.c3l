module gpu_test;

import gpu @public;
import gpu::vk @public;

fn RuntimeDesc device_creation_runtime_desc() {
    return {
        .backend          = BackendKind.VULKAN,
        .application_name = "gpu_adapter_device_test",
    };
}

fn Adapter device_creation_adapter(Runtime* runtime, uint index = 0) {
    return gpu::enumerate_adapters(runtime)!!.get(index)!!;
}

fn void adapter_device_preserves_identity_and_runtime_lifetime() @test {
    gpu::vk::stub_set_adapter_count(2);
    gpu::vk::stub_set_strict_support(true);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 1);
    DeviceRequest request = gpu::strict_device_request();

    Device device = gpu::create_device(&adapter, &request)!!;
    assert(
        (ulong)gpu::vk::stub_last_device_adapter() == (ulong)adapter,
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
        gpu::vk::stub_device_temporary_create_count()
            == gpu::vk::stub_device_temporary_destroy_count(),
        "destruction releases backend state"
    );
}

fn void distinct_adapter_devices_preserve_exact_owners() @test {
    gpu::vk::stub_set_adapter_count(2);
    gpu::vk::stub_set_strict_support(true);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter first_adapter = device_creation_adapter(&runtime, 0);
    Adapter second_adapter = device_creation_adapter(&runtime, 1);
    DeviceRequest request = gpu::strict_device_request();

    Device first = gpu::create_device(&first_adapter, &request)!!;
    Device second = gpu::create_device(&second_adapter, &request)!!;

    assert(
        (ulong)gpu::vk::stub_device_adapter(&first)!! == (ulong)first_adapter,
        "the first device keeps the first exact adapter"
    );
    assert(
        (ulong)gpu::vk::stub_device_adapter(&second)!! == (ulong)second_adapter,
        "the second device keeps the second exact adapter"
    );

    gpu::destroy_device(&first)!!;
    assert(
        (ulong)gpu::vk::stub_device_adapter(&second)!! == (ulong)second_adapter,
        "destroying one adapter device preserves its peer ownership"
    );
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
    assert(
        gpu::vk::stub_device_temporary_create_count()
            == gpu::vk::stub_device_temporary_destroy_count(),
        "both adapter-owned backend states are released"
    );
}

fn void foreign_buffer_owner_rejects_before_backend_mutation() @test {
    gpu::vk::stub_reset_buffer_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 0);
    DeviceRequest request = gpu::strict_device_request();
    Device first = gpu::create_device(&adapter, &request)!!;
    Device second = gpu::create_device(&adapter, &request)!!;
    BufferDesc buffer_desc = {
        .size        = 256,
        .usage       = { .transfer_dst },
        .memory_kind = MemoryKind.DEVICE,
    };
    BufferHandle first_buffer = gpu::create_buffer(&first, &buffer_desc)!!;
    BufferHandle second_buffer = gpu::create_buffer(&second, &buffer_desc)!!;

    assert(first_buffer.index == second_buffer.index, "stub devices use coincident local slots");
    assert(first_buffer.generation == second_buffer.generation, "stub devices use coincident local generations");
    assert(first_buffer.owner != second_buffer.owner, "coincident buffers retain distinct owners");
    uint before = gpu::vk::stub_buffer_destroy_attempt_count();
    assert(@catch(gpu::destroy_buffer(&second, first_buffer)) == gpu::INVALID_HANDLE);
    assert(
        gpu::vk::stub_buffer_destroy_attempt_count() == before,
        "foreign ownership rejects before backend mutation"
    );

    gpu::destroy_buffer(&first, first_buffer)!!;
    gpu::destroy_buffer(&second, second_buffer)!!;
    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}
fn void explicit_cross_kind_token_rejects_before_backend_mutation() @test {
    gpu::vk::stub_reset_texture_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime, 0);
    DeviceRequest request = gpu::strict_device_request();
    Device device = gpu::create_device(&adapter, &request)!!;
    BufferDesc buffer_desc = {
        .size        = 256,
        .usage       = { .transfer_dst },
        .memory_kind = MemoryKind.DEVICE,
    };
    BufferHandle buffer = gpu::create_buffer(&device, &buffer_desc)!!;
    TextureHandle forged = {
        .owner      = buffer.owner,
        .index      = buffer.index,
        .generation = buffer.generation,
    };

    uint before = gpu::vk::stub_texture_destroy_attempt_count();
    assert(@catch(gpu::destroy_texture(&device, forged)) == gpu::INVALID_HANDLE);
    assert(
        gpu::vk::stub_texture_destroy_attempt_count() == before,
        "resource-kind ownership rejects before backend mutation"
    );

    gpu::destroy_buffer(&device, buffer)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void unsupported_and_stale_adapter_creation_faults_before_allocation() @test {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(false);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported strict creation faults without fallback"
    );
    assert(gpu::vk::stub_device_temporary_create_count() == 0, "unsupported requests allocate nothing");

    gpu::destroy_runtime(&runtime)!!;
    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::INVALID_HANDLE,
        "a stale adapter faults before device work"
    );
    assert(gpu::vk::stub_device_temporary_create_count() == 0, "stale adapters allocate nothing");
}

fn void device_creation_failure_rolls_back_runtime_and_backend_state() @test {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(true);
    gpu::vk::stub_reset_device_counts();
    gpu::vk::stub_fail_device_after_temporary_allocation(true);
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    assert(
        @catch(gpu::create_device(&adapter, &request)) == gpu::BACKEND_ERROR,
        "failure after temporary allocation publishes no device"
    );
    assert(gpu::device_slot_available(), "failed creation leaves the device slot available");
    assert(
        gpu::vk::stub_device_temporary_create_count()
            == gpu::vk::stub_device_temporary_destroy_count(),
        "failed creation releases temporary backend state"
    );

    gpu::vk::stub_fail_device_after_temporary_allocation(false);
    gpu::destroy_runtime(&runtime)!!;
}

fn void invalid_retained_runtime_metadata_is_rejected_before_publication() @test {
    gpu::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = device_creation_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = device_creation_adapter(&runtime);
    Adapter forged_adapter = adapter;
    forged_adapter.runtime_generation++;
    FrameFixture fixture;
    Device owner;

    gpu::runtime_retain_device(&runtime)!!;
    Device? result = gpu::register_device(
        backend:          BackendKind.VULKAN,
        caps:             {},
        vtable:           &frame_fixture_vtable,
        backend_state:    &fixture,
        retained_runtime: &runtime,
        adapter:          &forged_adapter,
        owner:            &owner,
    );

    assert(@catch(result) == gpu::INVALID_HANDLE);
    assert(gpu::device_slot_available(), "invalid ownership metadata publishes no device");
    assert((ulong)owner == 0, "invalid ownership metadata writes no backend owner");
    gpu::runtime_release_device(&runtime)!!;
    gpu::destroy_runtime(&runtime)!!;
}
