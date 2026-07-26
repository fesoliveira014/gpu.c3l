module retired_api;

import gpu;

fn void use_retired_queue_counts(gpu::Device* device) {
    (void)gpu::get_queue_counts(device);
}
