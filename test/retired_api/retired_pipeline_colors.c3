module retired_pipeline_colors;

import gpu;

fn void use_retired_pipeline_colors() {
    gpu::GraphicsPipelineDesc desc = { .colors = {} };
    (void)desc;
}
