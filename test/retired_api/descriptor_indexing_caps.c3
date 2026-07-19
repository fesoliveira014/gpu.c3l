module retired_desc_index_caps;

import gpu;

fn bool use_retired_descriptor_indexing_caps(gpu::DeviceCaps caps) => caps.descriptor_indexing;
