module retired_hazard;

import gpu;

fn void use_retired_hazard() {
    gpu::Hazard hazard = {};
    (void)hazard;
}
