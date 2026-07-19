module retired_poll_readback;

import gpu;

fn bool? use_retired_poll_readback(gpu::Device* device) {
    return gpu::poll_readback(device, null);
}