module gpu_test;

import gpu @public;
import gpu::vk @public;

fn void adapter_request_creation_reuses_runtime_discovery() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_vk_device_request_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::DeviceRequest request = gpu::strict_device_request();
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!!;
        if (!support.supported) continue;

        gpu::RuntimeData* runtime_data = gpu::resolve_adapter(&adapter);
        gpu::vk::VkRuntimeState* runtime_state =
            (gpu::vk::VkRuntimeState*)runtime_data.backend_state;
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        gpu::vk::VkDeviceState* device_state =
            (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

        assert(device_state.instance == runtime_state.instance, "the device reuses the runtime instance");
        assert(
            device_state.physical_device
                == runtime_state.adapters[adapter.adapter_index].physical_device,
            "the device uses the exact requested physical adapter"
        );
        assert(!device_state.owns_instance, "the adapter device does not own the runtime instance");
        assert(gpu::get_device_caps(&device)!!.strict_enabled, "strict state is enabled");
        assert(
            @catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE,
            "the live device retains its runtime"
        );

        gpu::destroy_device(&device)!!;
        assert(
            gpu::get_adapter_info(&adapter)!!.name.len > 0,
            "device destruction leaves runtime discovery live"
        );
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter unsupported_adapter = adapters.get(0)!!;
    gpu::DeviceRequestSupport unsupported =
        gpu::supports_device_request(&unsupported_adapter, &request)!!;
    assert(!unsupported.supported, "no adapter silently falls back from strict semantics");
    assert(
        @catch(gpu::create_device(&unsupported_adapter, &request)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported native creation publishes no device"
    );
    gpu::destroy_runtime(&runtime)!!;
}

fn void malformed_native_request_preserves_runtime_state() @test {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_vk_device_request_failure_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceRequest empty = gpu::empty_device_request();
    assert(
        @catch(gpu::create_device(&adapter, &empty)) == gpu::INVALID_ARGUMENT,
        "an empty request is rejected before native creation"
    );
    assert(
        gpu::get_adapter_info(&adapter)!!.name.len > 0,
        "failed creation leaves runtime-owned adapter state intact"
    );
    gpu::destroy_runtime(&runtime)!!;
}
