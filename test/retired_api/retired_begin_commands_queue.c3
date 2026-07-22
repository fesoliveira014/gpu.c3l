module retired_begin_commands_queue;

import gpu;

fn void? use_retired_begin_commands(gpu::Queue queue) {
    gpu::CommandList commands = gpu::begin_commands(queue)!;
    (void)commands;
}
