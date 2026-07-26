module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;

// Headless device create/destroy. Needs a real Vulkan ICD at runtime; the test
// target pins lavapipe via VK_DRIVER_FILES for deterministic CPU runs.

vk::Result destroy_counter_result;
ulong destroy_counter_value;
uint destroy_counter_calls;

fn vk::Result query_destroy_counter(vk::Device device, vk::Semaphore semaphore, ulong* value) {
    (void)device;
    (void)semaphore;
    destroy_counter_calls++;
    *value = destroy_counter_value;
    return destroy_counter_result;
}

fn void device_destroy_readiness_never_waits() @test {
    vk::QueueCompletionState[1] completions = {{
        .timeline = (vk::Semaphore)(uptr)0x10,
        .bound    = true,
    }};
    completions[0].published_sequence.store(3, AtomicOrdering.RELAXED);
    vk::VkDeviceState state = {
        .queue_completions = completions[..],
    };
    destroy_counter_calls = 0;
    destroy_counter_result = vk::Result.SUCCESS;
    destroy_counter_value = 2;

    assert(
        !vk::device_destroy_ready_with_query(&state, &query_destroy_counter)!!,
        "an incomplete queue completion is busy"
    );
    assert(destroy_counter_calls == 1,
        "destruction probes queue progress without waiting");

    destroy_counter_value = 3;
    assert(
        vk::device_destroy_ready_with_query(&state, &query_destroy_counter)!!,
        "completed queue work makes destruction ready"
    );

    destroy_counter_result = vk::Result.ERROR_DEVICE_LOST;
    destroy_counter_value = 0;
    assert(
        vk::device_destroy_ready_with_query(&state, &query_destroy_counter)!!,
        "device loss permits teardown without waiting"
    );
}

fn void device_create_destroy() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .enable_debug_names = true,
        .application_name   = "gpu_vk_bootstrap_test",
    };

    gpu::Device device = create_test_device(&desc)!!;

    assert(gpu::internal::device_slots[device.index].data.state != null, "backend state allocated");
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(
        caps.generated_work == (caps.max_generated_work_count > 0),
        "generated-work support and its device limit are coherent"
    );
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::PhysicalDeviceFeatures features;
    vk::get_physical_device_features(state.physical_device, &features);
    assert(features.independent_blend != 0,
        "selected device supports independent blend");
    assert(features.depth_bias_clamp != 0,
        "selected device supports depth-bias clamp");
    assert(gpu::get_device_caps(&device)!!.line_polygon_mode == (features.fill_mode_non_solid != 0),
        "line polygon cap matches the selected physical device");
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(state.physical_device, &props);
    vk::PhysicalDeviceTimelineSemaphoreProperties timeline_props =
        vk::physical_device_timeline_semaphore_properties();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        vk::physical_device_extended_dynamic_state3properties_ext();
    timeline_props = timeline_props.set_next(&dynamic_properties);
    vk::PhysicalDeviceProperties2 props2 =
        vk::physical_device_properties2().set_next(&timeline_props);
    vk::get_physical_device_properties2(state.physical_device, &props2);
    assert(dynamic_properties.dynamic_primitive_topology_unrestricted != 0,
        "selected device supports unrestricted dynamic topology");
    ulong timeline_difference = state.max_timeline_semaphore_value_difference;
    assert(
        timeline_difference == timeline_props.max_timeline_semaphore_value_difference,
        "reported timeline-value difference matches the selected device"
    );
    uint expected_color_attachments = props.limits.max_color_attachments < gpu::MAX_COLOR_ATTACHMENTS
        ? props.limits.max_color_attachments : gpu::MAX_COLOR_ATTACHMENTS;
    assert(gpu::get_device_caps(&device)!!.max_color_attachments == expected_color_attachments, "effective color attachment limit matches the selected device");
    assert(
        gpu::get_device_caps(&device)!!.max_push_constant_size == props.limits.max_push_constants_size,
        "reported push-constant limit matches the selected physical device"
    );
    gpu::Vec3u compute_count_limit = gpu::get_device_caps(&device)!!.max_compute_work_group_count;
    assert(
        compute_count_limit.x == props.limits.max_compute_work_group_count[0]
            && compute_count_limit.y == props.limits.max_compute_work_group_count[1]
            && compute_count_limit.z == props.limits.max_compute_work_group_count[2],
        "reported dispatch-count limits match the selected physical device"
    );
    assert(
        gpu::get_device_caps(&device)!!.max_draw_indirect_count
            == props.limits.max_draw_indirect_count,
        "reported indirect draw-count limit matches the selected physical device"
    );
    assert(gpu::get_device_caps(&device)!!.max_push_constant_size >= gpu::RootPush::size, "push-constant limit contains RootPush");
    uint device_index = device.index;
    destroy_test_device(&device)!!;
    assert(gpu::internal::device_slots[device_index].data.state == null,
        "backend state cleared on destroy");
    assert((ulong)device == 0, "destroy consumes the device token");

    fault destroyed_fault = @catch(gpu::get_memory_stats(&device));
    assert(destroyed_fault == gpu::INVALID_HANDLE, "destroyed device use faults INVALID_HANDLE");
}

fn void device_heap_capacities() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .texture_heap_capacity = 64,
        .sampler_heap_capacity = 8,
        .application_name      = "gpu_vk_heap_capacity_test",
    };

    gpu::Device device = create_test_device(&desc)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.texture_heap_capacity == 64, "texture capacity reflects the desc");
    assert(caps.sampler_heap_capacity == 8, "sampler capacity reflects the desc");
    destroy_test_device(&device)!!;
}

fn void device_capacity_over_ceiling_rejected() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_capacity_test",
    };
    uint over_ceiling = gpu::MAX_SHADER_HEAP_CAPACITY + 1;

    desc.texture_heap_capacity = over_ceiling;
    assert(
        @catch(create_test_device(&desc)) == gpu::INVALID_ARGUMENT,
        "texture heap capacity above the limit fails with INVALID_ARGUMENT"
    );
    desc.texture_heap_capacity = 0;
    desc.sampler_heap_capacity = over_ceiling;
    assert(
        @catch(create_test_device(&desc)) == gpu::INVALID_ARGUMENT,
        "sampler heap capacity above the limit fails with INVALID_ARGUMENT"
    );
    desc.sampler_heap_capacity = 0;
    desc.texture_capacity = over_ceiling;
    assert(
        @catch(create_test_device(&desc)) == gpu::INVALID_ARGUMENT,
        "texture capacity above the limit fails with INVALID_ARGUMENT"
    );
    desc.texture_capacity = 0;
    desc.pipeline_capacity = over_ceiling;
    assert(
        @catch(create_test_device(&desc)) == gpu::INVALID_ARGUMENT,
        "pipeline capacity above the limit fails with INVALID_ARGUMENT"
    );
}
