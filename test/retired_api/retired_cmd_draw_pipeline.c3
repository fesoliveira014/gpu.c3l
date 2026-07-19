module retired_cmd_draw_pipeline;

import gpu;

fn void? use_retired_cmd_draw(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
) {
    gpu::cmd_draw(commands, pipeline, vertex_root, fragment_root, 3, 1)!;
}
