module retired_get_persistent_stats;

import gpu;

fn void? use_retired_get_persistent_stats(gpu::Device* device) {
    gpu::get_persistent_stats(device)!;
}
