module retired_cmd_upload_buffer;

import gpu;

fn void? use_retired_cmd_upload_buffer(gpu::CommandList* commands) {
    gpu::cmd_upload_buffer(commands, {}, {})!;
}
