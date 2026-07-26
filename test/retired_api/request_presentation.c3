module retired_request_presentation;

import gpu;

fn void use_retired_request_presentation() {
    (void)gpu::request_presentation({}, null);
}
