module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

struct BarrierEmitCapture {
    uint calls;
}

fn void capture_barrier_emit(
    void* user_data,
    vk::MemoryBarrier2* native,
) {
    (void)native;
    BarrierEmitCapture* capture = (BarrierEmitCapture*)user_data;
    capture.calls++;
}

fn vk::Result create_selected_native_device(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* requested_info,
    vk::Device* device,
) {
    (void)user_data;
    vk::PhysicalDeviceFeatures2* features =
        (vk::PhysicalDeviceFeatures2*)requested_info.p_next;
    vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR* unified =
        (vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR*)features.p_next;
    features.p_next = unified.p_next;
    defer features.p_next = unified;

    ZString[8] extensions;
    uint extension_count;
    for (uint i = 0; i < requested_info.enabled_extension_count; i++) {
        ZString extension =
            (ZString)requested_info.pp_enabled_extension_names[i];
        if (extension.str_view()
            == gpu::internal::vk::RESOURCE_AGNOSTIC_TEXTURE_SYNC_EXTENSION
                .str_view()) {
            continue;
        }
        extensions[extension_count++] = extension;
    }
    vk::DeviceCreateInfo native_info = *requested_info;
    native_info = native_info.set_enabled_extension_names(
        extensions[:extension_count],
    );
    return vk::create_device(
        physical_device: physical_device,
        p_create_info:   &native_info,
        p_allocator:     null,
        p_device:        device,
    );
}

fn gpu::Device? create_selected_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name         = "resource_agnostic_texture_sync_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceRequest request = gpu::request_resource_agnostic_texture_sync(
        gpu::strict_device_request(),
    )!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::internal::RuntimeData* runtime_data =
            gpu::internal::resolve_adapter(&adapter);
        vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        adapter_state.resource_agnostic_texture_sync = {
            .extension             = true,
            .unified_image_layouts = true,
        };
        if (!gpu::supports_device_request(&adapter, &request)!.supported) {
            continue;
        }
        vk::DeviceCreationProbe probe = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_extension_available = true,
            .resource_agnostic_texture_sync_feature_available = true,
            .native_device_create_hook = &create_selected_native_device,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!;
        gpu::Device? device = vk::create_vk_device_with_probe(
            adapter:          &adapter,
            request:          &request,
            retained_runtime: &retained_runtime,
            probe:            &probe,
        );
        if (catch err = device) {
            gpu::internal::runtime_release_device(&retained_runtime)!;
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        return device;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void selected_texture_dependency_is_resource_free() @test {
    gpu::Device device = create_selected_device()!!;
    defer destroy_test_device(&device)!!;
    assert(
        gpu::get_device_caps(&device)!!.resource_agnostic_texture_sync,
        "the selected capability publishes after native creation"
    );

    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    BarrierEmitCapture capture;
    vk::set_barrier_emit_hook(state, &capture_barrier_emit, &capture);
    defer vk::set_barrier_emit_hook(state, null, null);
    vk::reset_barrier_work_counters(state);
    gpu::Barrier dependency = {
        .before = { .compute },
        .after  = { .compute },
    };
    gpu::cmd_barrier(&commands, &dependency)!!;
    vk::BarrierWorkCounters barriers = vk::barrier_work_counters(state);

    assert(
        capture.calls == 1
            && barriers.native_memory_emissions == 1
            && barriers.native_image_emissions == 0,
        "one selected dependency emits one memory barrier and no image barrier"
    );
    assert(
        barriers.texture_helper_calls == 0
            && barriers.texture_handle_resolutions == 0,
        "the selected dependency performs no texture inference"
    );
}
