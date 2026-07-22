module retired_prior_use;

import gpu;

fn void use_retired_prior_use() {
    gpu::AcquiredImage image = { .prior_use = {} };
    (void)image;
}
