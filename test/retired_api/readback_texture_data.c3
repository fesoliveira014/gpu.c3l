module retired_readback_texture_data;

import gpu;

fn void? use_retired_readback_texture_data(gpu::Device* device) {
    gpu::readback_texture_data(device, {}, 0, {}, {}, {}, {})!;
}
