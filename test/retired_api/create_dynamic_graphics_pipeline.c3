module retired_create_dynamic_pipeline;

import gpu;

fn void? use_retired_create_dynamic_graphics_pipeline(
    gpu::Device* device,
    gpu::GraphicsPipelineDesc* desc,
) {
    (void)gpu::create_dynamic_graphics_pipeline(device, desc)!;
}
