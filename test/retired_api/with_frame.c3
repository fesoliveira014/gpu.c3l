module retired_with_frame;

import gpu;

fn void? frame_work(void* frame) { }

fn void? use_retired_with_frame(gpu::Device* device) {
    gpu::@with_frame(null, device, frame_work)!;
}