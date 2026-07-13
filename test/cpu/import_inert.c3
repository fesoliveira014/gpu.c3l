module gpu::compat::vk @private;

import gpu;

fn int main() @public {
    gpu::compat::Device device;
    (void)device;

    return create_device_calls == 0 ? 0 : 1;
}
