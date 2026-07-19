module retired_begin_frame;

import gpu;

fn void? use_retired_begin_frame(gpu::Device* device) {
    (void)gpu::begin_frame(device)!;
}