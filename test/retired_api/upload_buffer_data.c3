module retired_upload_buffer_data;

import gpu;

fn void? use_retired_upload_buffer_data(gpu::Device* device) {
    gpu::upload_buffer_data(device, {}, {}, {}, {})!;
}
