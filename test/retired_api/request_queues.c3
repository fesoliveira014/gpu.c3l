module retired_request_queues;

import gpu;

fn void use_retired_request_queues() {
    (void)gpu::request_queues({}, {});
}
