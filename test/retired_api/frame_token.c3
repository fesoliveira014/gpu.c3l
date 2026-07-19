module retired_frame_token;

import gpu;

fn void use_retired_frame_token() {
    gpu::FrameToken frame = {};
    (void)frame;
}