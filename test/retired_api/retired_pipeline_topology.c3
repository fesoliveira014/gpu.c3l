module retired_pipeline_topology;

import gpu;

fn void use_retired_pipeline_topology() {
    gpu::GraphicsPipelineDesc desc = { .topology = {} };
    (void)desc;
}
