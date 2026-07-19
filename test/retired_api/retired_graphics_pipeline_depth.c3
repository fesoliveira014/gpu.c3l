module retired_graphics_pipeline_depth;

import gpu;

fn void use_retired_graphics_pipeline_depth() {
    gpu::GraphicsPipelineDesc desc = { .depth = {} };
    (void)desc;
}
