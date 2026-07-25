module retired_command_list_device;

import gpu;

fn void use_retired_command_list_device() {
    gpu::CommandList commands;
    (void)commands.device;
}
