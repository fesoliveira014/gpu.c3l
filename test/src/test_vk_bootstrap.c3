module gpu_test;

import gpu;

// Headless device create/destroy. Needs a real Vulkan ICD at runtime; the test
// target pins lavapipe via VK_DRIVER_FILES for deterministic CPU runs.

fn void device_create_destroy() @test {
    gpu::DeviceDesc desc = {
        .backend                            = gpu::BackendKind.VULKAN,
        .enable_validation                  = true,
        .enable_debug_names                 = true,
        .frames_in_flight                   = 2,
        .allow_descriptor_indexing_fallback = true,
        .application_name                   = "gpu_vk_bootstrap_test",
    };

    gpu::Device device = gpu::create_device(&desc)!!;

    assert(device.backend == gpu::BackendKind.VULKAN, "backend recorded");
    assert(device.backend_state != null, "backend state allocated");
    assert(device.caps.buffer_device_address, "buffer_device_address cap true");
    assert(device.caps.synchronization2, "synchronization2 cap true");
    assert(device.caps.dynamic_rendering, "dynamic_rendering cap true");
    assert(device.caps.timeline_semaphore, "timeline_semaphore cap true");

    gpu::destroy_device(&device)!!;
    assert(device.backend_state == null, "backend state cleared on destroy");
}

// Note: the UNSUPPORTED_BACKEND default arm is unreachable while BackendKind has
// only one value — C3 safe-mode panics on any invalid-ordinal cast — so it is
// not unit-tested here. It gains coverage when a second backend is added.
