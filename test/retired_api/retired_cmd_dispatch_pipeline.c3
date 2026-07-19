module retired_cmd_dispatch_pipeline;

import gpu;

fn void? use_retired_cmd_dispatch(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
) {
    gpu::cmd_dispatch(commands, pipeline, root, {})!;
}
