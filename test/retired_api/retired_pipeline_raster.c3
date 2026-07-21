module retired_pipeline_raster;

import gpu;

fn void use_retired_pipeline_raster() {
    gpu::GraphicsPipelineDesc desc = { .raster = {} };
    (void)desc;
}
