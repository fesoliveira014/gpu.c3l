module gpu_profile_fast_fail;

import gpu;

fn void invalid_fast_call() {
    gpu::CommandList commands;
    gpu::cmd_dispatch(&commands, {}, {})!;
}
