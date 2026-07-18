module retired_cmd_readback_buffer;

import gpu;

fn void? use_retired_cmd_readback_buffer(
    gpu::CommandList* commands,
    gpu::GpuSpan span,
) {
    gpu::cmd_readback_buffer(commands, span)!;
}