module retired_graphics_pipeline_blend;

import gpu;

fn void use_retired_graphics_pipeline_blend() {
    gpu::GraphicsPipelineDesc desc = { .blend = {} };
    (void)desc;
}
