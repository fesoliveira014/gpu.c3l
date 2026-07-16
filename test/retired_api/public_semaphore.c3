module retired_public_semaphore;

import gpu;

fn void use_public_semaphore() {
    gpu::SemaphoreHandle semaphore = {};
    (void)semaphore;
}