module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;

// Headless device create/destroy. Needs a real Vulkan ICD at runtime; the test
// target pins lavapipe via VK_DRIVER_FILES for deterministic CPU runs.

fn void device_create_destroy() @test {
    gpu::compat::DeviceDesc desc = {
        .backend              = gpu::compat::BackendKind.VULKAN,
        .enable_validation    = true,
        .enable_debug_names   = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::compat::DescriptorHeapMode.AUTO,
        .application_name     = "gpu_vk_bootstrap_test",
    };

    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;

    assert(gpu::compat::get_device_backend(&device)!! == gpu::compat::BackendKind.VULKAN, "backend recorded");
    assert(gpu::compat::device_backend_state(device) != null, "backend state allocated");
    assert(gpu::compat::get_device_caps(&device)!!.buffer_device_address, "buffer_device_address cap true");
    assert(gpu::compat::get_device_caps(&device)!!.synchronization2, "synchronization2 cap true");
    assert(gpu::compat::get_device_caps(&device)!!.dynamic_rendering, "dynamic_rendering cap true");
    assert(gpu::compat::get_device_caps(&device)!!.timeline_semaphore, "timeline_semaphore cap true");
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    vk::PhysicalDeviceFeatures features;
    vk::get_physical_device_features(state.physical_device, &features);
    assert(gpu::compat::get_device_caps(&device)!!.line_polygon_mode == (features.fill_mode_non_solid != 0),
        "line polygon cap matches the selected physical device");
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(state.physical_device, &props);
    vk::PhysicalDeviceTimelineSemaphoreProperties timeline_props =
        vk::physical_device_timeline_semaphore_properties();
    vk::PhysicalDeviceProperties2 props2 =
        vk::physical_device_properties2().set_next(&timeline_props);
    vk::get_physical_device_properties2(state.physical_device, &props2);
    ulong timeline_difference = gpu::compat::get_device_caps(&device)!!.max_timeline_semaphore_value_difference;
    assert(
        timeline_difference == timeline_props.max_timeline_semaphore_value_difference,
        "reported timeline-value difference matches the selected device"
    );
    uint expected_color_attachments = props.limits.max_color_attachments < gpu::compat::MAX_COLOR_ATTACHMENTS
        ? props.limits.max_color_attachments : gpu::compat::MAX_COLOR_ATTACHMENTS;
    assert(gpu::compat::get_device_caps(&device)!!.max_color_attachments == expected_color_attachments, "effective color attachment limit matches the selected device");
    assert(
        gpu::compat::get_device_caps(&device)!!.max_push_constant_size == props.limits.max_push_constants_size,
        "reported push-constant limit matches the selected physical device"
    );
    gpu::compat::Vec3u compute_count_limit = gpu::compat::get_device_caps(&device)!!.max_compute_work_group_count;
    assert(
        compute_count_limit.x == props.limits.max_compute_work_group_count[0]
            && compute_count_limit.y == props.limits.max_compute_work_group_count[1]
            && compute_count_limit.z == props.limits.max_compute_work_group_count[2],
        "reported dispatch-count limits match the selected physical device"
    );
    assert(
        gpu::compat::get_device_caps(&device)!!.max_draw_indirect_count
            == props.limits.max_draw_indirect_count,
        "reported indirect draw-count limit matches the selected physical device"
    );
    assert(gpu::compat::get_device_caps(&device)!!.max_push_constant_size >= gpu::compat::RootPush::size, "push-constant limit contains RootPush");
    gpu::compat::destroy_device(&device)!!;
    assert(gpu::compat::device_backend_state(device) == null, "backend state cleared on destroy");
    assert((ulong)device == 0, "destroy consumes the device token");

    fault destroyed_fault = @catch(gpu::compat::get_memory_stats(&device));
    assert(destroyed_fault == gpu::compat::INVALID_HANDLE, "destroyed device use faults INVALID_HANDLE");
}

fn void device_heap_mode_and_capacities() @test {
    gpu::compat::DeviceDesc desc = {
        .backend                     = gpu::compat::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .texture_descriptor_capacity = 64,
        .sampler_descriptor_capacity = 8,
        .application_name            = "gpu_vk_heap_mode_test",
    };

    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    assert(gpu::compat::get_device_caps(&device)!!.descriptor_indexing, "indexing support reported");
    assert(gpu::compat::get_device_caps(&device)!!.max_texture_descriptors == 64, "texture capacity reflects the desc");
    assert(gpu::compat::get_device_caps(&device)!!.max_sampler_descriptors == 8, "sampler capacity reflects the desc");
    gpu::compat::destroy_device(&device)!!;
}

fn void device_capacity_over_ceiling_rejected() @test {
    gpu::compat::DeviceDesc desc = {
        .backend                     = gpu::compat::BackendKind.VULKAN,
        .texture_descriptor_capacity = gpu::compat::MAX_DESCRIPTOR_SLOTS + 1,
        .application_name            = "gpu_vk_capacity_test",
    };

    fault capacity_fault = @catch(gpu::compat::create_device(&desc));
    assert(capacity_fault == gpu::compat::INVALID_ARGUMENT, "capacity above the packed-slot ceiling fails with INVALID_ARGUMENT");
}

// Note: the UNSUPPORTED_BACKEND default arm is unreachable while BackendKind has
// only one value — C3 safe-mode panics on any invalid-ordinal cast — so it is
// not unit-tested here. It gains coverage when a second backend is added.
