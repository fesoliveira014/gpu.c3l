module retired_raster_state;

import gpu;

fn void use_retired_raster_state() {
    gpu::RasterState state = {};
    (void)state;
}
