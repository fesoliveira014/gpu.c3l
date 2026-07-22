module retired_sampler_invalid;

import gpu;

fn void use_retired() {
    (void)gpu::SAMPLER_INVALID;
}
