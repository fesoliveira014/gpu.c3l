module retired_sig_dispatch_indirect;

import gpu;

fn void? use_retired_cmd_dispatch_indirect(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    gpu::cmd_dispatch_indirect(commands, pipeline, root, args)!;
}
