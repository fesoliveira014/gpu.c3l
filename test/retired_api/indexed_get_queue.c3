module retired_api;

import gpu;

fn void use_retired_indexed_queue(gpu::Device* device) {
    (void)gpu::get_queue(device, gpu::QueueKind.GRAPHICS, 0);
}
