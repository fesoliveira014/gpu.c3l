module retired_cmd_readback_texture;

import gpu;

fn void? use_retired_cmd_readback_texture(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
) {
    gpu::cmd_readback_texture(commands, texture, 0)!;
}