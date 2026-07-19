module retired_max_smp_desc_caps;

import gpu;

fn uint use_retired_max_sampler_descriptors_caps(gpu::DeviceCaps caps) => caps.max_sampler_descriptors;
