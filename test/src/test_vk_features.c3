module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Pure-CPU: feeds the feature gate synthetic structs, no ICD required.

fn vk::PhysicalDeviceFeatures base_with_int64() {
    vk::PhysicalDeviceFeatures base;
    base.shader_int64 = vk::TRUE;
    return base;
}

fn void required_features_present_ok() @test {
    vk::PhysicalDeviceFeatures base = base_with_int64();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);

    bool faulted = false;
    if (catch gpu::vk::check_required_features(&base, &v12, &v13)) faulted = true;
    assert(!faulted, "all required features present must pass the gate");
}

fn void missing_feature_rejected() @test {
    // All-zero structs: no required feature set.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features();
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features();

    bool faulted = false;
    if (catch gpu::vk::check_required_features(&base, &v12, &v13)) faulted = true;
    assert(faulted, "a missing required feature must raise UNSUPPORTED_FEATURE");
}

fn void one_missing_feature_rejected() @test {
    // Everything but dynamic_rendering.
    vk::PhysicalDeviceFeatures base = base_with_int64();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE);

    bool faulted = false;
    if (catch gpu::vk::check_required_features(&base, &v12, &v13)) faulted = true;
    assert(faulted, "a single missing feature must still fault");
}

fn void missing_shader_int64_rejected() @test {
    // Everything except shaderInt64.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);

    bool faulted = false;
    if (catch gpu::vk::check_required_features(&base, &v12, &v13)) faulted = true;
    assert(faulted, "missing shaderInt64 must raise UNSUPPORTED_FEATURE");
}
