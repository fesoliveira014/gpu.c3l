module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc device_support_runtime_desc() {
    return {
        .application_name = "gpu_device_request_support_test",
    };
}

fn Adapter first_support_adapter(Runtime* runtime) {
    return gpu::enumerate_adapters(runtime)!!.get(0)!!;
}

fn void request_support_is_semantic_and_does_not_enable_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();

    DeviceRequestSupport support = gpu::supports_device_request(&adapter, &request)!!;
    assert(support.supported, "strict semantics are supported");
    assert(support.unmet_requirement.len == 0, "a supported request has no unmet label");
    assert(gpu::internal::device_slot_available(), "support detection enables no device state");
    assert(gpu::internal::vk::stub_device_support_query_count() == 1, "support is queried once");
}

fn void unsupported_request_names_the_first_semantic_requirement() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(false);
    gpu::internal::vk::stub_reset_device_counts();
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
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest empty = gpu::internal::empty_device_request();
    DeviceRequest malformed;
    DeviceRequest reserved = (DeviceRequest)((uint128)gpu::strict_device_request() | ((uint128)1 << 63));

    assert(@catch(gpu::supports_device_request(&adapter, &empty)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::supports_device_request(&adapter, &malformed)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::supports_device_request(&adapter, &reserved)) == gpu::INVALID_ARGUMENT);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0, "invalid requests bypass backend work");

    DeviceRequest valid = gpu::strict_device_request();
    gpu::destroy_runtime(&runtime)!!;
    assert(@catch(gpu::supports_device_request(&adapter, &valid)) == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0, "stale adapters bypass backend work");
}

fn void unavailable_queue_count_is_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    QueueRequirements requirements = { .counts = { .graphics = 2 } };
    DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    DeviceRequestSupport support = gpu::supports_device_request(&adapter, &request)!!;
    assert(!support.supported, "an unavailable queue count is rejected");
    assert(support.unmet_requirement == "graphics queues", "the unmet label stays semantic");
}

fn void unavailable_queue_topology_is_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };
    DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    DeviceRequestSupport support =
        gpu::supports_device_request(&adapter, &request)!!;
    assert(!support.supported, "an unavailable queue topology is rejected");
    assert(
        support.unmet_requirement == "queue topology",
        "the unmet label stays semantic"
    );
}
