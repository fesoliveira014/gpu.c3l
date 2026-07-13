module gpu;

// A non-zero `swapchain` couples this submit to the last acquire: rendering
// waits the acquire semaphore, presentation waits this submit. One coupled
// submit per acquire — a second faults INVALID_RESOURCE_STATE, and coupling
// on a non-GRAPHICS queue faults INVALID_ARGUMENT.
struct SubmitDesc {
    CommandList[]     command_lists;
    SemaphoreWait[]   waits;
    SemaphoreSignal[] signals;
    SwapchainHandle   swapchain;
}

<* Submit command lists to their queue, waiting on and signaling the requested
   timeline values. All lists must share a queue kind and be executable. *>
fn void? submit(Device* device, SubmitDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.submit(device, desc);
}

<* Block until the queue drains — a heavyweight sync for shutdown and one-shot work, not per-frame pacing. *>
fn void? wait_queue_idle(Device* device, QueueKind queue) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.wait_queue_idle(device, queue);
}
