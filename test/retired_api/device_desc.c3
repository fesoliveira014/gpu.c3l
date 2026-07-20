module retired_device_desc;

import gpu;

fn void use_retired_device_desc() {
    gpu::DeviceDesc desc = {};
    (void)desc;
}
