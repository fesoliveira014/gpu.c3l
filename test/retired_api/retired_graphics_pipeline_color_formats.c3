module retired_gfx_color_formats;

import gpu;

fn void use_retired_graphics_pipeline_color_formats() {
    gpu::GraphicsPipelineDesc desc = { .color_formats = {} };
    (void)desc;
}
