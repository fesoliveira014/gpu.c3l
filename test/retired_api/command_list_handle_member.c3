module retired_command_handle_member;

import gpu;

fn void use_retired_command_list_handle_member() {
    gpu::CommandList commands;
    (void)commands.handle;
}
