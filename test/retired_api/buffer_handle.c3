module retired_buffer_handle;

import gpu;

fn void use_retired_buffer_handle() {
    gpu::BufferHandle buffer = {};
    (void)buffer;
}
