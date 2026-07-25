module retired_texture_sync_request;

import gpu;

fn gpu::DeviceRequest? use_retired_resource_agnostic_texture_sync_request(
    gpu::DeviceRequest request,
) {
    return gpu::request_resource_agnostic_texture_sync(request);
}
