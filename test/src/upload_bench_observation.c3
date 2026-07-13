module gpu_bench;

import gpu::compat;
import vk;

fn ZString upload_device_type_name(vk::PhysicalDeviceType device_type) {
    switch (device_type) {
        case vk::PhysicalDeviceType.INTEGRATED_GPU: return "integrated";
        case vk::PhysicalDeviceType.DISCRETE_GPU:   return "discrete";
        case vk::PhysicalDeviceType.VIRTUAL_GPU:    return "virtual";
        case vk::PhysicalDeviceType.CPU:            return "cpu";
        case vk::PhysicalDeviceType.OTHER:          return "other";
        default:                                    return "unknown";
    }
}

fn ZString upload_heap_mode_name(gpu::compat::DescriptorHeapMode heap_mode) {
    switch (heap_mode) {
        case gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING: return "descriptor-indexing";
        case gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER:   return "descriptor-buffer";
        case gpu::compat::DescriptorHeapMode.AUTO:                return "auto";
        default:                                          return "unknown";
    }
}

fn bool upload_is_lavapipe_driver(vk::DriverId driver_id) {
    return driver_id == vk::DriverId.MESA_LLVMPIPE;
}

fn double upload_scaling_vs_one(double rate, double baseline) {
    return baseline > 0.0 ? rate / baseline : 0.0;
}

fn bool upload_dedicated_fallback_used(ulong fallback_count) {
    return fallback_count > 0;
}
