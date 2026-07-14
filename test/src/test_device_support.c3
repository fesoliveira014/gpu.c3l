module gpu_test;

import gpu @public;
import gpu::vk @public;

fn RuntimeDesc device_support_runtime_desc() {
    return {
        .backend          = BackendKind.VULKAN,
        .application_name = "gpu_device_request_support_test",
    };
}

fn Adapter first_support_adapter(Runtime* runtime) {
    return gpu::enumerate_adapters(runtime)!!.get(0)!!;
}

fn void request_support_is_semantic_and_does_not_enable_state() @test {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(true);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    DeviceRequestSupport support = gpu::supports_device_request(&adapter, &request)!!;
    assert(support.supported, "strict semantics are supported");
    assert(support.unmet_requirement.len == 0, "a supported request has no unmet label");
    assert(gpu::device_slot_available(), "support detection enables no device state");
    assert(gpu::vk::stub_device_support_query_count() == 1, "support is queried once");
}

fn void unsupported_request_names_the_first_semantic_requirement() @test {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(false);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    DeviceRequestSupport support = gpu::supports_device_request(&adapter, &request)!!;
    assert(!support.supported, "unsupported strict semantics report false");
    assert(
        support.unmet_requirement == "strict semantics",
        "the unmet label is GPU-semantic"
    );
}

fn void invalid_requests_and_stale_adapters_fault_before_backend_work() @test {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(true);
    gpu::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest empty = gpu::empty_device_request();
    DeviceRequest malformed;
    DeviceRequest reserved = (DeviceRequest)((ulong)gpu::strict_device_request() | (1ull << 63));

    assert(@catch(gpu::supports_device_request(&adapter, &empty)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::supports_device_request(&adapter, &malformed)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::supports_device_request(&adapter, &reserved)) == gpu::INVALID_ARGUMENT);
    assert(gpu::vk::stub_device_support_query_count() == 0, "invalid requests bypass backend work");

    DeviceRequest valid = gpu::strict_device_request();
    gpu::destroy_runtime(&runtime)!!;
    assert(@catch(gpu::supports_device_request(&adapter, &valid)) == gpu::INVALID_HANDLE);
    assert(gpu::vk::stub_device_support_query_count() == 0, "stale adapters bypass backend work");
}
