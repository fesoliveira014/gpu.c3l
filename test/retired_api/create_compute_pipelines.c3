module retired_compute_pipelines;

import gpu;

fn void? use_retired_create_compute_pipelines(
    gpu::Device* device,
    gpu::ComputePipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    gpu::create_compute_pipelines(device, descs, out_pipelines)!;
}
