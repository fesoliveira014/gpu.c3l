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

    assert(!@catch(gpu::vk::check_required_features(&base, &v12, &v13)), "all required features present must pass the gate");
}

fn void missing_feature_rejected() @test {
    // All-zero structs: no required feature set.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features();
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features();

    fault gate_fault = @catch(gpu::vk::check_required_features(&base, &v12, &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "a missing required feature raises UNSUPPORTED_FEATURE");
}

fn void one_missing_feature_rejected() @test {
    // Everything but dynamic_rendering.
    vk::PhysicalDeviceFeatures base = base_with_int64();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE);

    fault gate_fault = @catch(gpu::vk::check_required_features(&base, &v12, &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "a single missing feature still raises UNSUPPORTED_FEATURE");
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

    fault gate_fault = @catch(gpu::vk::check_required_features(&base, &v12, &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "missing shaderInt64 raises UNSUPPORTED_FEATURE");
}

fn vk::PhysicalDeviceVulkan12Features v12_with_heap_bits() {
    return vk::physical_device_vulkan12features()
        .set_runtime_descriptor_array(vk::TRUE)
        .set_shader_sampled_image_array_non_uniform_indexing(vk::TRUE)
        .set_shader_storage_image_array_non_uniform_indexing(vk::TRUE)
        .set_descriptor_binding_partially_bound(vk::TRUE)
        .set_descriptor_binding_sampled_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_storage_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_update_unused_while_pending(vk::TRUE);
}

fn vk::PhysicalDeviceFeatures base_with_storage_access() {
    vk::PhysicalDeviceFeatures base;
    base.shader_storage_image_read_without_format  = vk::TRUE;
    base.shader_storage_image_write_without_format = vk::TRUE;
    return base;
}

fn void heap_indexing_gate() @test {
    vk::PhysicalDeviceFeatures base = base_with_storage_access();
    vk::PhysicalDeviceVulkan12Features full = v12_with_heap_bits();
    assert(gpu::vk::heap_shared_features_supported(&base, &full), "full bits pass the shared gate");
    assert(gpu::vk::heap_indexing_supported(&base, &full), "full bits pass the indexing gate");

    vk::PhysicalDeviceVulkan12Features no_partial = v12_with_heap_bits()
        .set_descriptor_binding_partially_bound(vk::FALSE);
    assert(gpu::vk::heap_shared_features_supported(&base, &no_partial), "shared gate ignores set-path bits");
    assert(!gpu::vk::heap_indexing_supported(&base, &no_partial), "missing partially-bound fails the indexing gate");

    vk::PhysicalDeviceFeatures no_storage_access;
    assert(!gpu::vk::heap_shared_features_supported(&no_storage_access, &full), "missing format-less storage access fails the shared gate");

    vk::PhysicalDeviceVulkan12Features empty = vk::physical_device_vulkan12features();
    assert(!gpu::vk::heap_shared_features_supported(&base, &empty), "empty 1.2 features fail the shared gate");
}

fn void heap_mode_auto_resolution() @test {
    gpu::DescriptorHeapMode auto_mode = gpu::DescriptorHeapMode.AUTO;

    gpu::DescriptorHeapMode both = gpu::vk::resolve_heap_mode(auto_mode, true, true)!!;
    assert(both == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING, "AUTO prefers descriptor indexing");

    gpu::DescriptorHeapMode buffer_only = gpu::vk::resolve_heap_mode(auto_mode, false, true)!!;
    assert(buffer_only == gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER, "AUTO falls back to descriptor buffer");

    fault auto_fault = @catch(gpu::vk::resolve_heap_mode(auto_mode, false, false));
    assert(auto_fault == gpu::UNSUPPORTED_FEATURE, "AUTO with no supported path raises UNSUPPORTED_FEATURE");
}

fn void heap_mode_forced_resolution() @test {
    gpu::DescriptorHeapMode forced = gpu::vk::resolve_heap_mode(gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING, true, true)!!;
    assert(forced == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING, "forced indexing wins over available buffer");

    fault buffer_fault = @catch(gpu::vk::resolve_heap_mode(gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER, true, false));
    assert(buffer_fault == gpu::UNSUPPORTED_FEATURE, "forced buffer without support raises UNSUPPORTED_FEATURE");

    fault indexing_fault = @catch(gpu::vk::resolve_heap_mode(gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING, false, true));
    assert(indexing_fault == gpu::UNSUPPORTED_FEATURE, "forced indexing without support raises UNSUPPORTED_FEATURE");
}
