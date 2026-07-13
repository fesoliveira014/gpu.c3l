module gpu::compat::vk @private;

import gpu::compat;

uint create_device_calls;

fn gpu::compat::Device? create_vk_device(gpu::compat::DeviceDesc* desc) {
    create_device_calls++;
    return gpu::compat::UNSUPPORTED_BACKEND~;
}
