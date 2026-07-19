module retired_resolve_readback;

import gpu;

fn void? use_retired_resolve_readback(gpu::Device* device, char[] data) {
    gpu::resolve_readback(device, null, data)!;
}