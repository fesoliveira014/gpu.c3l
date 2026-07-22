module sampler_operations;

import gpu;

fn gpu::SamplerIndex? sampler_operations(
    gpu::Device* device,
    gpu::SamplerDesc* desc,
) {
    return gpu::intern_sampler(device, desc);
}
