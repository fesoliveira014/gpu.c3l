module retired_color_state_format;

import gpu;

fn gpu::ColorTargetState use_retired_color_target_state_format() => {
    .format = gpu::Format.RGBA8_UNORM,
};
