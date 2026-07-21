module retired_compute_push_constant;

import gpu;

fn void use_retired_compute_push_constant() {
    gpu::ComputePipelineDesc desc = { .push_constant_size = 8 };
    (void)desc;
}
