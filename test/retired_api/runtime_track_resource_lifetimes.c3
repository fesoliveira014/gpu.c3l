module retired_runtime_tracking;

import gpu;

fn gpu::RuntimeDesc use_retired_tracking_flag() => {
    .track_resource_lifetimes = true,
};
