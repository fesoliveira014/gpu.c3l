module retired_cmd_upload_texture;

import gpu;

fn void? use_retired_cmd_upload_texture(gpu::CommandList* commands) {
    gpu::TextureUploadDesc desc;
    gpu::cmd_upload_texture(commands, &desc, {})!;
}
