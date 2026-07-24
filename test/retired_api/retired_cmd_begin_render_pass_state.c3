module retired_begin_pass_state;

import gpu;

fn void? use_retired_begin_state(
    gpu::CommandList* commands,
    gpu::RenderPassDesc* pass,
    gpu::GraphicsState* state,
) {
    gpu::cmd_begin_render_pass(commands, pass, state)!;
}
