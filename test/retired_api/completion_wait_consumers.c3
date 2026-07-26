module retired_wait_consumers;

import gpu;

fn void use_retired_wait_consumers() {
    gpu::CompletionWait wait = { .consumers = {} };
    (void)wait;
}
