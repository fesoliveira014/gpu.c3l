module retired_create_dynamic_batch;

import gpu;

fn void? use_retired_create_dynamic_graphics_pipelines(
    gpu::Device* device,
    gpu::GraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    gpu::create_dynamic_graphics_pipelines(device, descs, out_pipelines)!;
}
