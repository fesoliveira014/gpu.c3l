module retired_executable_device;

import gpu;

fn void use_retired_executable_command_list_device() {
    gpu::ExecutableCommandList commands;
    (void)commands.device;
}
