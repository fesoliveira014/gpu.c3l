module gpu_bench;

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

fn bool upload_is_lavapipe_driver(vk::DriverId driver_id) {
    return driver_id == vk::DriverId.MESA_LLVMPIPE;
}

fn double upload_scaling_vs_one(double rate, double baseline) {
    return baseline > 0.0 ? rate / baseline : 0.0;
}
