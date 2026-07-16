module retired_queue_idle;

import gpu;

fn void? use_queue_idle(gpu::Device* device) {
    gpu::wait_queue_idle(device, gpu::QueueKind.GRAPHICS)!;
}