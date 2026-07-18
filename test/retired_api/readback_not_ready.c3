module retired_readback_not_ready;

import gpu;

fn fault use_retired_readback_not_ready() {
    return gpu::READBACK_NOT_READY;
}