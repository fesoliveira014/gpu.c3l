module retired_global_barrier;

import gpu;

fn void use_retired_global_barrier() {
    gpu::GlobalBarrier barrier = {};
    (void)barrier;
}
