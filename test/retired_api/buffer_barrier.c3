module retired_buffer_barrier;

import gpu;

fn void use_retired_buffer_barrier() {
    gpu::BufferBarrier barrier = {};
    (void)barrier;
}
