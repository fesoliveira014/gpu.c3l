module retired_gfx_raster;

import gpu;

fn void use_retired_graphics_pipeline_raster() {
    gpu::GraphicsPipelineDesc desc = { .raster = {} };
    (void)desc;
}
