module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Headless device create/destroy. Needs a real Vulkan ICD at runtime; the test
// target pins lavapipe via VK_DRIVER_FILES for deterministic CPU runs.

fn void device_create_destroy() @test {
    gpu::DeviceDesc desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .enable_debug_names   = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.AUTO,
        .application_name     = "gpu_vk_bootstrap_test",
    };

    gpu::Device device = gpu::create_device(&desc)!!;

    assert(device.backend == gpu::BackendKind.VULKAN, "backend recorded");
    assert(device.backend_state != null, "backend state allocated");
    assert(device.caps.buffer_device_address, "buffer_device_address cap true");
    assert(device.caps.synchronization2, "synchronization2 cap true");
    assert(device.caps.dynamic_rendering, "dynamic_rendering cap true");
    assert(device.caps.timeline_semaphore, "timeline_semaphore cap true");
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::PhysicalDeviceFeatures features;
    vk::get_physical_device_features(state.physical_device, &features);
    assert(device.caps.line_polygon_mode == (features.fill_mode_non_solid != 0),
        "line polygon cap matches the selected physical device");
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(state.physical_device, &props);
    uint expected_color_attachments = props.limits.max_color_attachments < gpu::MAX_COLOR_ATTACHMENTS
        ? props.limits.max_color_attachments : gpu::MAX_COLOR_ATTACHMENTS;
    assert(device.caps.max_color_attachments == expected_color_attachments, "effective color attachment limit matches the selected device");
    assert(
        device.caps.max_push_constant_size == props.limits.max_push_constants_size,
        "reported push-constant limit matches the selected physical device"
    );
    assert(device.caps.max_push_constant_size >= gpu::RootPush::size, "push-constant limit contains RootPush");
    gpu::destroy_device(&device)!!;
    assert(device.backend_state == null, "backend state cleared on destroy");
    assert(device.vtable == null, "destroyed device dispatch is invalidated");

    fault destroyed_fault = @catch(gpu::get_memory_stats(&device));
    assert(destroyed_fault == gpu::INVALID_ARGUMENT, "destroyed device use faults INVALID_ARGUMENT");
}

fn void device_heap_mode_and_capacities() @test {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .texture_descriptor_capacity = 64,
        .sampler_descriptor_capacity = 8,
        .application_name            = "gpu_vk_heap_mode_test",
    };

    gpu::Device device = gpu::create_device(&desc)!!;
    assert(device.caps.descriptor_indexing, "indexing support reported");
    assert(device.caps.max_texture_descriptors == 64, "texture capacity reflects the desc");
    assert(device.caps.max_sampler_descriptors == 8, "sampler capacity reflects the desc");
    gpu::destroy_device(&device)!!;
}

fn void device_capacity_over_ceiling_rejected() @test {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .texture_descriptor_capacity = gpu::MAX_DESCRIPTOR_SLOTS + 1,
        .application_name            = "gpu_vk_capacity_test",
    };

    fault capacity_fault = @catch(gpu::create_device(&desc));
    assert(capacity_fault == gpu::INVALID_ARGUMENT, "capacity above the packed-slot ceiling fails with INVALID_ARGUMENT");
}

// Note: the UNSUPPORTED_BACKEND default arm is unreachable while BackendKind has
// only one value — C3 safe-mode panics on any invalid-ordinal cast — so it is
// not unit-tested here. It gains coverage when a second backend is added.
