module retired_desc_buffer_caps;

import gpu;

fn bool use_retired_descriptor_buffer_caps(gpu::DeviceCaps caps) => caps.descriptor_buffer;
