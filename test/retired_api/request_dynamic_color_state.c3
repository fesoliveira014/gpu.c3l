module retired_request_dynamic_color;

import gpu;

fn gpu::DeviceRequest? use_retired_request_dynamic_color_state(
    gpu::DeviceRequest request,
) {
    return gpu::request_dynamic_color_state(request);
}
