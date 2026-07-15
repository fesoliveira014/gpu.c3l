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
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    foreach (&commands : desc.command_lists) {
        if (commands.device != *device) return INVALID_HANDLE~;
        validate_resource_owner(&operation, commands.handle.owner, ResourceOwnerKind.COMMAND)!;
    }
    foreach (&wait : desc.waits) {
        validate_resource_owner(&operation, wait.semaphore.owner, ResourceOwnerKind.SEMAPHORE)!;
    }
    foreach (&signal : desc.signals) {
        validate_resource_owner(&operation, signal.semaphore.owner, ResourceOwnerKind.SEMAPHORE)!;
    }
    if (desc.swapchain != SWAPCHAIN_HANDLE_INVALID) {
        validate_resource_owner(&operation, desc.swapchain.owner, ResourceOwnerKind.SWAPCHAIN)!;
    }
    return operation.vtable.submit(device, desc);
}

<* Block until the queue drains — a heavyweight sync for shutdown and one-shot work, not per-frame pacing. *>
fn void? wait_queue_idle(Device* device, QueueKind queue) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.wait_queue_idle(device, queue);
}
