module retired_command_invalid;

import gpu;

fn void use_retired_command_list_handle_invalid() {
    (void)gpu::COMMAND_LIST_HANDLE_INVALID;
}
