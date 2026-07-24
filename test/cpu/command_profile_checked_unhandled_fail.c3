module gpu_profile_checked_fail;

import gpu;

fn void invalid_checked_call() {
    gpu::CommandList commands;
    gpu::cmd_dispatch(&commands, {}, {});
}
