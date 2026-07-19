module retired_create_sampler;
import gpu;
fn void? use_retired(gpu::Device* device, gpu::SamplerDesc* desc) {
    gpu::create_sampler(device, desc)!;
}
