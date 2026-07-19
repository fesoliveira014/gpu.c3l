module retired_smp_desc_capacity;

import gpu;

fn gpu::DeviceDesc use_retired_sampler_descriptor_capacity() => { .sampler_descriptor_capacity = 1 };
