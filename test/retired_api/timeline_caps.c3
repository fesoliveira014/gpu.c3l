module retired_timeline_caps;

import gpu;

fn bool use_timeline_caps(gpu::DeviceCaps caps) {
    return caps.timeline_semaphore
        && caps.max_timeline_semaphore_value_difference > 0;
}