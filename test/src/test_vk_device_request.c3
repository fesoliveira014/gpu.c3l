module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::atomic;

fn uint live_device_slot_count() @private {
    uint live_count;
    for (uint i = 0; i < gpu::MAX_DEVICES; i++) {
        gpu::DeviceSlot* slot = &gpu::device_slots[i];
        gpu::DeviceSlotControl control =
            (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state == gpu::DEVICE_SLOT_LIVE) live_count++;
    }
    return live_count;
}

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
        assert(!device_state.presentation.enabled, "headless requests create no presentation state");
        assert(device_state.presentation.queue == null, "headless requests retrieve no presentation queue");
        assert(!device_state.presentation.queue_mutex.is_initialized(), "headless requests initialize no presentation mutex");
        assert(device_state.presentation.swapchains.slots.ptr == null, "headless requests allocate no swapchain table");
        assert(
            !gpu::vk::instance_dispatch_supports_presentation(&device_state.instance_dispatch),
            "headless requests retain no surface procedures"
        );
        assert(
            !gpu::vk::device_dispatch_supports_presentation(&device_state.device_dispatch),
            "headless requests load no presentation procedures"
        );
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

fn void failed_native_device_creation_rolls_back_before_publication() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .application_name  = "gpu_vk_device_request_rollback_test",
        .enable_validation = true,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request = gpu::strict_device_request();

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;

        uint initialized_before = gpu::vk::device_creation_initialized_count();
        uint rolled_back_before = gpu::vk::device_creation_rollback_count();
        uint live_slots_before = live_device_slot_count();
        gpu::vk::inject_device_creation_failure_before_publish(true);
        defer gpu::vk::inject_device_creation_failure_before_publish(false);

        assert(
            @catch(gpu::create_device(&adapter, &request)) == gpu::BACKEND_ERROR,
            "injected native creation failure returns the backend fault"
        );
        assert(
            live_device_slot_count() == live_slots_before,
            "failed native creation publishes no device"
        );
        assert(
            gpu::vk::device_creation_initialized_count() == initialized_before + 1,
            "failure occurs after native initialization"
        );
        assert(
            gpu::vk::device_creation_rollback_count() == rolled_back_before + 1,
            "failure tears down the initialized native state"
        );
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::destroy_runtime(&runtime)!!;
}

fn void runtime_devices_destroy_independently() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_vk_multi_device_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request = gpu::strict_device_request();

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;

        gpu::Adapter second_adapter = adapter;
        for (uint j = 0; j < adapters.count; j++) {
            if (j == i) continue;
            gpu::Adapter candidate = adapters.get(j)!!;
            if (!gpu::supports_device_request(&candidate, &request)!!.supported) continue;
            second_adapter = candidate;
            break;
        }

        gpu::Device first = gpu::create_device(&adapter, &request)!!;
        gpu::Device second = gpu::create_device(&second_adapter, &request)!!;
        assert((ulong)first != (ulong)second, "same-runtime devices have distinct identities");

        gpu::BufferDesc buffer_desc = {
            .size        = 256,
            .usage       = { .storage },
            .memory_kind = gpu::MemoryKind.DEVICE,
        };
        gpu::BufferHandle first_buffer = gpu::create_buffer(&first, &buffer_desc)!!;
        gpu::BufferHandle second_buffer = gpu::create_buffer(&second, &buffer_desc)!!;
        gpu::destroy_buffer(&first, first_buffer)!!;
        gpu::destroy_buffer(&second, second_buffer)!!;

        gpu::destroy_device(&first)!!;
        assert(gpu::get_device_caps(&second)!!.strict_enabled, "destroying one device preserves its peer");

        gpu::BufferHandle buffer = gpu::create_buffer(&second, &buffer_desc)!!;
        gpu::destroy_buffer(&second, buffer)!!;
        gpu::destroy_device(&second)!!;
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::destroy_runtime(&runtime)!!;
}
