module sampler_operations;

import gpu;

fn gpu::SamplerIndex? sampler_operations(
    gpu::Device* device,
    gpu::SamplerDesc* desc,
) {
    gpu::Sampler sampler = gpu::intern_sampler(device, desc)!;
    return gpu::publish_sampler(device, sampler);
}