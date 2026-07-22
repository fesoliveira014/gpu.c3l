module retired_clear_depth_stencil;

import gpu;

fn void use_retired() {
    gpu::ClearDepthStencil clear;
    (void)clear;
}
