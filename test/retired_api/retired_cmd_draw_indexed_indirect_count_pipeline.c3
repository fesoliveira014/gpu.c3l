module retired_sig_draw_idx_count;

import gpu;

fn void? use_retired_cmd_draw_indexed_indirect_count(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count,
    gpu::GpuSpan indices,
) {
    gpu::cmd_draw_indexed_indirect_count(
        commands,
        pipeline,
        vertex_root,
        fragment_root,
        args,
        count,
        1,
        indices,
    )!;
}
