module retired_hazard_flags;

import gpu;

fn void use_retired_hazard_flags() {
    gpu::HazardFlags hazards = {};
    (void)hazards;
}
