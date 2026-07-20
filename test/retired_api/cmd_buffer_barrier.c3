module retired_cmd_buffer_barrier;

import gpu;

fn void? use_retired_cmd_buffer_barrier(gpu::CommandList* commands) {
    gpu::cmd_buffer_barrier(commands, null)!;
}
