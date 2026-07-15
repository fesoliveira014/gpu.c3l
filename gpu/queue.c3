module gpu;

const uint QUEUE_ID_GRAPHICS @private = 0;
const uint QUEUE_ID_COMPUTE @private = 1;
const uint QUEUE_ID_TRANSFER @private = 2;

const Queue QUEUE_INVALID = {};

bitstruct QueueRoles : uint {
    bool graphics : 0;
    bool compute  : 1;
    bool transfer : 2;
}

struct Queue {
    ulong      owner;
    uint       id;
    QueueRoles roles;
}

struct QueueInfo {
    uint       id;
    QueueRoles roles;
}

struct QueueCounts {
    uint graphics;
    uint compute;
    uint transfer;
}

struct QueueSelection @private {
    QueueCounts counts;
    uint graphics_id;
    uint compute_id;
    uint transfer_id;
    QueueRoles graphics_roles;
    QueueRoles compute_roles;
    QueueRoles transfer_roles;
}

<* Submit command lists to their queue. A non-zero `swapchain` couples this
   submit to the last acquired image and must use the graphics queue. *>
struct SubmitDesc {
    CommandList[]     command_lists;
    SemaphoreWait[]   waits;
    SemaphoreSignal[] signals;
    SwapchainHandle   swapchain;
}

fn bool Queue.is_valid(self) => self.owner != 0;

fn bool queue_roles_equal(QueueRoles first, QueueRoles second) @private {
    return (uint)first == (uint)second;
}

fn QueueCounts default_queue_counts() @private {
    return {
        .graphics = 1,
        .compute  = 1,
        .transfer = 1,
    };
}

fn QueueSelection default_queue_selection(DeviceCaps* caps = null) @private {
    QueueCounts counts = caps == null ? default_queue_counts() : caps.queues;
    if (counts.graphics == 0) counts.graphics = 1;
    if (counts.compute == 0) counts.compute = 1;
    if (counts.transfer == 0) counts.transfer = 1;

    QueueSelection selection = {
        .counts         = counts,
        .graphics_id    = QUEUE_ID_GRAPHICS,
        .compute_id     = QUEUE_ID_GRAPHICS,
        .transfer_id    = QUEUE_ID_GRAPHICS,
        .graphics_roles = { .graphics, .compute, .transfer },
        .compute_roles  = { .graphics, .compute, .transfer },
        .transfer_roles = { .graphics, .compute, .transfer },
    };
    if (caps != null && caps.async_compute) {
        selection.graphics_roles.compute = false;
        selection.compute_id = QUEUE_ID_COMPUTE;
        selection.compute_roles = { .compute };
    }
    return selection;
}

fn QueueSelection selected_queues(DeviceOperation* operation) @private {
    if (operation == null || !operation.active) return {};
    return operation.data.queues;
}

fn QueueCounts selected_queue_counts(DeviceOperation* operation) @private {
    return selected_queues(operation).counts;
}

fn uint selected_queue_count(DeviceOperation* operation, QueueKind kind) @private {
    QueueCounts counts = selected_queue_counts(operation);
    switch (kind) {
        case QueueKind.GRAPHICS: return counts.graphics;
        case QueueKind.COMPUTE:  return counts.compute;
        case QueueKind.TRANSFER: return counts.transfer;
    }
}

fn uint selected_queue_id(DeviceOperation* operation, QueueKind kind) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return selected_queues(operation).graphics_id;
        case QueueKind.COMPUTE:  return selected_queues(operation).compute_id;
        case QueueKind.TRANSFER: return selected_queues(operation).transfer_id;
    }
}

fn QueueRoles selected_queue_roles(DeviceOperation* operation, QueueKind kind) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return selected_queues(operation).graphics_roles;
        case QueueKind.COMPUTE:  return selected_queues(operation).compute_roles;
        case QueueKind.TRANSFER: return selected_queues(operation).transfer_roles;
    }
}

fn bool queue_id_is_selected(DeviceOperation* operation, Queue queue) @private {
    QueueSelection selection = selected_queues(operation);
    if (queue.id == selection.graphics_id
        && queue_roles_equal(queue.roles, selection.graphics_roles)) {
        return true;
    }
    if (queue.id == selection.compute_id
        && queue_roles_equal(queue.roles, selection.compute_roles)) {
        return true;
    }
    return queue.id == selection.transfer_id
        && queue_roles_equal(queue.roles, selection.transfer_roles);
}

fn void? validate_queue(DeviceOperation* operation, Queue queue) @private {
    if (operation == null || !operation.active || !queue.is_valid()) return INVALID_HANDLE~;
    validate_resource_owner(operation, queue.owner, ResourceOwnerKind.QUEUE)!;
    if (!queue_id_is_selected(operation, queue)) return INVALID_HANDLE~;
}

<* Return selected queue counts by semantic role. Aliased roles count the same
   device-local queue under each supported role. *>
fn QueueCounts? get_queue_counts(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return selected_queue_counts(&operation);
}

<* Return an owner-bearing queue identity for a semantic role and index. Aliased
   roles return the same token. *>
fn Queue? get_queue(Device* device, QueueKind kind, uint index = 0) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (index >= selected_queue_count(&operation, kind)) return INVALID_ARGUMENT~;
    return {
        .owner = resource_owner_tag(operation.data.resource_owner, ResourceOwnerKind.QUEUE),
        .id    = selected_queue_id(&operation, kind) + index,
        .roles = selected_queue_roles(&operation, kind),
    };
}

<* Return a live queue token's device-local identity and supported roles. *>
fn QueueInfo? get_queue_info(Device* device, Queue queue) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, queue)!;
    return {
        .id    = queue.id,
        .roles = queue.roles,
    };
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
