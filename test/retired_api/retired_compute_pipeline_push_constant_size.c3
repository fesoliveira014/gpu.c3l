module retired_compute_push_size;

import gpu;

fn void use_retired_compute_pipeline_push_constant_size() {
    gpu::ComputePipelineDesc desc = { .push_constant_size = 8 };
    (void)desc;
}
