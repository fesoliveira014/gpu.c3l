module retired_sig_draw_indirect;

import gpu;

fn void? use_retired_cmd_draw_indirect(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
) {
    gpu::cmd_draw_indirect(commands, pipeline, vertex_root, fragment_root, args, 1)!;
}
