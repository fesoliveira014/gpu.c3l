module gpu;

struct Queue {
    ulong     owner;
    QueueKind kind;
    uint      index;
}

struct QueueInfo {
    QueueKind kind;
    uint      index;
}

struct QueueCounts {
    uint graphics;
    uint compute;
    uint transfer;
}

const Queue QUEUE_INVALID = {};

fn bool Queue.is_valid(self) => self.owner != 0;

fn QueueCounts selected_queue_counts(DeviceCaps* caps) @private {
    return caps == null ? {} : caps.queues;
}

fn uint selected_queue_count(DeviceCaps* caps, QueueKind kind) @private {
    QueueCounts counts = selected_queue_counts(caps);
    switch (kind) {
        case QueueKind.GRAPHICS: return counts.graphics;
        case QueueKind.COMPUTE: return counts.compute;
        case QueueKind.TRANSFER: return counts.transfer;
    }
}

fn void? validate_queue(DeviceOperation* operation, Queue queue) @private {
    if (operation == null || !operation.active || !queue.is_valid()) return INVALID_HANDLE~;
    validate_resource_owner(operation, queue.owner, ResourceOwnerKind.QUEUE)!;
    if (queue.index >= selected_queue_count(&operation.data.caps, queue.kind)) {
        return INVALID_HANDLE~;
    }
}

<* Return the selected semantic queue counts for the device. *>
fn QueueCounts? get_queue_counts(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return selected_queue_counts(&operation.data.caps);
}

<* Return a semantic queue token owned by the device. *>
fn Queue? get_queue(Device* device, QueueKind kind, uint index = 0) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (index >= selected_queue_count(&operation.data.caps, kind)) return INVALID_ARGUMENT~;
    return {
        .owner = resource_owner_tag(operation.data.resource_owner, ResourceOwnerKind.QUEUE),
        .kind  = kind,
        .index = index,
    };
}

<* Return the semantic role and device-local index for a live queue token. *>
fn QueueInfo? get_queue_info(Device* device, Queue queue) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, queue)!;
    return {
        .kind  = queue.kind,
        .index = queue.index,
    };
}

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
