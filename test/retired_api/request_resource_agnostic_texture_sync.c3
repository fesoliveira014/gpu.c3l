module retired_texture_sync_request;

import gpu;

fn void use_retired_resource_agnostic_texture_sync_request() {
    (void)gpu::request_resource_agnostic_texture_sync({});
}
