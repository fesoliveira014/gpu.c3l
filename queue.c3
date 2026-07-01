module gpu;

struct SubmitDesc {
    CommandList[]     command_lists;
    SemaphoreWait[]   waits;
    SemaphoreSignal[] signals;
}

<* Submit command lists to their queue, waiting on and signaling the requested
   timeline values. All lists must share a queue kind and be executable. *>
fn void? submit(Device* device, SubmitDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.submit(device, desc);
}

fn void? wait_queue_idle(Device* device, QueueKind queue) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.wait_queue_idle(device, queue);
}
