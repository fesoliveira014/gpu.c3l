module retired_sig_draw_indexed;

import gpu;

fn void? use_retired_cmd_draw_indexed(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan indices,
) {
    gpu::cmd_draw_indexed(commands, pipeline, vertex_root, fragment_root, indices, 3, 1)!;
}
