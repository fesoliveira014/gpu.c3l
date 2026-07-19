module retired_tex_desc_capacity;

import gpu;

fn gpu::DeviceDesc use_retired_texture_descriptor_capacity() => { .texture_descriptor_capacity = 1 };
