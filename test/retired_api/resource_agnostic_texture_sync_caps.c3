module retired_texture_sync_caps;

import gpu;

fn bool use_retired_resource_agnostic_texture_sync_caps(
    gpu::DeviceCaps caps,
) {
    return caps.resource_agnostic_texture_sync;
}
