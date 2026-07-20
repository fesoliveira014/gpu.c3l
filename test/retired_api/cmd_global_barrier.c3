module retired_cmd_global_barrier;

import gpu;

fn void? use_retired_cmd_global_barrier(gpu::CommandList* commands) {
    gpu::cmd_global_barrier(commands, null)!;
}
