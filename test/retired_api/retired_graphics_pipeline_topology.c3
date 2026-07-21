module retired_gfx_topology;

import gpu;

fn void use_retired_graphics_pipeline_topology() {
    gpu::GraphicsPipelineDesc desc = { .topology = {} };
    (void)desc;
}
