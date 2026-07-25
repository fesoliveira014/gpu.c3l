module retired_strict_enabled_caps;

import gpu;

fn bool use_retired_strict_enabled_caps(gpu::DeviceCaps caps) {
    return caps.strict_enabled;
}
