module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

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

fn void resource_agnostic_texture_sync_support_uses_the_backend_query() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_resource_agnostic_texture_sync_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest classic = gpu::strict_device_request();

    assert(gpu::supports_device_request(&adapter, &classic)!!.supported);
    assert(
        gpu::internal::vk::stub_resource_agnostic_texture_sync_support_query_count() == 0,
        "an unrequested capability does not query backend support"
    );

    DeviceRequest requested =
        gpu::request_resource_agnostic_texture_sync(classic)!!;
    DeviceRequestSupport unsupported =
        gpu::supports_device_request(&adapter, &requested)!!;
    assert(!unsupported.supported);
    assert(
        unsupported.unmet_requirement
            == "resource-agnostic texture synchronization",
        "the unmet label remains backend-neutral"
    );
    assert(
        gpu::internal::vk::stub_resource_agnostic_texture_sync_support_query_count() == 1,
        "a requested capability queries backend support once"
    );

    gpu::internal::vk::stub_set_resource_agnostic_texture_sync_support(true);
    assert(gpu::supports_device_request(&adapter, &requested)!!.supported);
    assert(
        gpu::internal::vk::stub_resource_agnostic_texture_sync_support_query_count() == 2,
        "a supported request follows the same backend query path"
    );
}

fn void dynamic_color_state_support_uses_the_backend_query() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_dynamic_color_state_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest classic = gpu::strict_device_request();

    assert(gpu::supports_device_request(&adapter, &classic)!!.supported);
    assert(
        gpu::internal::vk::stub_dynamic_color_state_support_query_count() == 0,
        "an unrequested capability does not query backend support"
    );

    DeviceRequest requested = gpu::request_dynamic_color_state(classic)!!;
    DeviceRequestSupport unsupported =
        gpu::supports_device_request(&adapter, &requested)!!;
    assert(!unsupported.supported);
    assert(
        unsupported.unmet_requirement == "dynamic color state",
        "the unmet label remains backend-neutral"
    );
    assert(
        gpu::internal::vk::stub_dynamic_color_state_support_query_count() == 1,
        "a requested capability queries backend support once"
    );

    gpu::internal::vk::stub_set_dynamic_color_state_support(true);
    assert(gpu::supports_device_request(&adapter, &requested)!!.supported);
    assert(
        gpu::internal::vk::stub_dynamic_color_state_support_query_count() == 2,
        "a supported request follows the same backend query path"
    );
}

fn void supported_but_unrequested_device_remains_classic() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_resource_agnostic_texture_sync_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();
    Device device = gpu::create_device(&adapter, &request)!!;
    defer gpu::destroy_device(&device)!!;

    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(
        !caps.resource_agnostic_texture_sync,
        "hardware support alone does not select the capability"
    );
}

fn void supported_but_unrequested_device_has_immutable_color_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_dynamic_color_state_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request = gpu::strict_device_request();
    Device device = gpu::create_device(&adapter, &request)!!;
    defer gpu::destroy_device(&device)!!;

    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(
        !caps.dynamic_color_state,
        "hardware support alone does not enable dynamic color state"
    );
}

fn void supported_requested_device_publishes_dynamic_color_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_dynamic_color_state_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceRequest request =
        gpu::request_dynamic_color_state(gpu::strict_device_request())!!;
    Device device = gpu::create_device(&adapter, &request)!!;
    defer gpu::destroy_device(&device)!!;

    assert(
        gpu::get_device_caps(&device)!!.dynamic_color_state,
        "a supported requested capability is published only by the device"
    );
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
