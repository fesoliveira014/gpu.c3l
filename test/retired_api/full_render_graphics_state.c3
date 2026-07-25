module retired_render_state;

import gpu;

fn gpu::GraphicsState? use_retired_full_render_graphics_state() {
    return gpu::full_render_graphics_state(1, 1);
}
