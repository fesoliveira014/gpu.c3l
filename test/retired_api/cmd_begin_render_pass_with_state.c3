module retired_begin_pass_with_state;

import gpu;

fn void? use_retired_begin_render_pass_with_state(
    gpu::CommandList* commands,
    gpu::RenderPassDesc* pass,
    gpu::GraphicsState* state,
) {
    gpu::cmd_begin_render_pass_with_state(commands, pass, state)!;
}
