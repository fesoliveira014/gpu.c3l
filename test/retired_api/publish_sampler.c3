module retired_publish_sampler;

import gpu;

fn void use_retired(gpu::Device* device) {
    (void)gpu::publish_sampler(device, {});
}
