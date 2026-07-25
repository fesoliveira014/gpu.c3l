module retired_executable_handle;

import gpu;

fn void use_retired_executable_command_list_handle() {
    gpu::ExecutableCommandList commands;
    (void)commands.handle;
}
