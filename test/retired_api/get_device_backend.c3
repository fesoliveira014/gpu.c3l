module retired_get_device_backend;

import gpu;

fn void use_retired_get_device_backend(gpu::Device* device) {
    (void)gpu::get_device_backend(device);
}
