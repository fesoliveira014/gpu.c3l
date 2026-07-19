module retired_upload_texture_data;

import gpu;

fn void? use_retired_upload_texture_data(gpu::Device* device) {
    gpu::upload_texture_data(device, {}, 0, {}, {}, {}, {})!;
}
