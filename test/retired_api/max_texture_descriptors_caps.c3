module retired_max_tex_desc_caps;

import gpu;

fn uint use_retired_max_texture_descriptors_caps(gpu::DeviceCaps caps) => caps.max_texture_descriptors;
