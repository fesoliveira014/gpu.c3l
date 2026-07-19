module retired_destroy_sampler;
import gpu;
fn void? use_retired(gpu::Device* device, gpu::SamplerIndex index) {
    gpu::destroy_sampler(device, index)!;
}
