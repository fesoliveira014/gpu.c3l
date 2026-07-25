module retired_hazard_descriptors;

import gpu;

fn bool use_retired_descriptor_hazard(gpu::HazardFlags hazards) => hazards.descriptors;
