module retired_barrier_hazards;

import gpu;

fn void use_retired_barrier_hazards() {
    gpu::Barrier barrier = { .hazards = {} };
    (void)barrier;
}
