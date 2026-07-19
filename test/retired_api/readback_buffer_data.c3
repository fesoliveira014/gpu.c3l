module retired_readback_buffer_data;

import gpu;

fn void? use_retired_readback_buffer_data(gpu::Device* device) {
    gpu::readback_buffer_data(device, {}, {}, {}, {})!;
}
