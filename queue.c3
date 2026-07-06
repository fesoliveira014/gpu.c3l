module gpu;

// A non-zero `swapchain` couples this submit to the last acquire: rendering
// waits the acquire semaphore, presentation waits this submit. One coupled
// submit per acquire — the binary semaphores cannot be waited twice.
struct SubmitDesc {
    CommandList[]     command_lists;
    SemaphoreWait[]   waits;
    SemaphoreSignal[] signals;
    SwapchainHandle   swapchain;
}

<* Submit command lists to their queue, waiting on and signaling the requested
   timeline values. All lists must share a queue kind and be executable. *>
fn void? submit(Device* device, SubmitDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.submit(device, desc);
}

<* Block until the queue drains — a heavyweight sync for shutdown and one-shot work, not per-frame pacing. *>
fn void? wait_queue_idle(Device* device, QueueKind queue) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.wait_queue_idle(device, queue);
}
