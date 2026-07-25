module retired_dynamic_color_caps;

import gpu;

fn bool use_retired_dynamic_color_state_caps(gpu::DeviceCaps caps) {
    return caps.dynamic_color_state;
}
