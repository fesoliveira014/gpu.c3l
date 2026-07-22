module retired_sampler_identity;

import gpu;

fn void use_retired() {
    gpu::Sampler sampler;
    (void)sampler;
}
