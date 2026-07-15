module gpu;

const Queue QUEUE_INVALID = {};

<* Semantic roles supported by one selected queue. *>
bitstruct QueueRoles : uint {
    bool graphics : 0;
    bool compute  : 1;
    bool transfer : 2;
}

<* Device-owned selected queue token. Copy or compare it as a whole; use
   get_queue_info to inspect its identity and supported roles. *>
struct Queue {
    Device     device;
    uint       id;
    QueueRoles roles;
}

<* Stable device-local identity and supported roles for a live queue. *>
struct QueueInfo {
    uint       id;
    QueueRoles roles;
}

<* Selected queue count under each semantic role. *>
struct QueueCounts {
    uint graphics;
    uint compute;
    uint transfer;
}

<* Semantic queues required at device creation. A marked distinct role may
   not alias another requested role. *>
struct QueueRequirements {
    QueueCounts counts;
    QueueRoles distinct;
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

fn bool Queue.is_valid(self) => self.device.is_valid();
fn bool Queue.equals(self, Queue other) @operator(==) {
    return self.device == other.device
        && self.id == other.id
        && (uint)self.roles == (uint)other.roles;
}

fn bool queue_roles_equal(QueueRoles first, QueueRoles second) @private {
    return (uint)first == (uint)second;
}

fn QueueSelection? queue_selection_for_requirements(
    QueueRequirements requirements,
    bool compute_aliases_graphics,
    bool transfer_aliases_graphics,
    bool transfer_aliases_compute,
) @private {
    if (!queue_requirements_are_valid(requirements)) return INVALID_ARGUMENT~;
    if (requirements.counts.graphics > 1
        || requirements.counts.compute > 1
        || requirements.counts.transfer > 1) {
        return UNSUPPORTED_FEATURE~;
    }

    QueueSelection selection = { .counts = requirements.counts };
    QueueRoles[3] identity_roles;
    uint next_id;
    if (requirements.counts.graphics > 0) {
        selection.graphics_id = next_id++;
        identity_roles[selection.graphics_id].graphics = true;
    }
    if (requirements.counts.compute > 0) {
        selection.compute_id = requirements.counts.graphics > 0
            && compute_aliases_graphics
            ? selection.graphics_id
            : next_id++;
        identity_roles[selection.compute_id].compute = true;
    }
    if (requirements.counts.transfer > 0) {
        if (requirements.counts.graphics > 0 && transfer_aliases_graphics) {
            selection.transfer_id = selection.graphics_id;
        } else if (requirements.counts.compute > 0 && transfer_aliases_compute) {
            selection.transfer_id = selection.compute_id;
        } else {
            selection.transfer_id = next_id++;
        }
        identity_roles[selection.transfer_id].transfer = true;
    }

    if (requirements.counts.graphics > 0) {
        selection.graphics_roles = identity_roles[selection.graphics_id];
    }
    if (requirements.counts.compute > 0) {
        selection.compute_roles = identity_roles[selection.compute_id];
    }
    if (requirements.counts.transfer > 0) {
        selection.transfer_roles = identity_roles[selection.transfer_id];
    }

    bool graphics_alias = requirements.counts.graphics > 0
        && ((requirements.counts.compute > 0
                && selection.graphics_id == selection.compute_id)
            || (requirements.counts.transfer > 0
                && selection.graphics_id == selection.transfer_id));
    bool compute_alias = requirements.counts.compute > 0
        && ((requirements.counts.graphics > 0
                && selection.compute_id == selection.graphics_id)
            || (requirements.counts.transfer > 0
                && selection.compute_id == selection.transfer_id));
    bool transfer_alias = requirements.counts.transfer > 0
        && ((requirements.counts.graphics > 0
                && selection.transfer_id == selection.graphics_id)
            || (requirements.counts.compute > 0
                && selection.transfer_id == selection.compute_id));
    if ((requirements.distinct.graphics && graphics_alias)
        || (requirements.distinct.compute && compute_alias)
        || (requirements.distinct.transfer && transfer_alias)) {
        return UNSUPPORTED_FEATURE~;
    }
    return selection;
}

fn QueueSelection queue_selection_for_aliases(
    bool compute_aliases_graphics,
    bool transfer_aliases_graphics,
    bool transfer_aliases_compute,
) @private {
    return queue_selection_for_requirements(
        requirements:               default_queue_requirements(),
        compute_aliases_graphics:    compute_aliases_graphics,
        transfer_aliases_graphics:   transfer_aliases_graphics,
        transfer_aliases_compute:    transfer_aliases_compute,
    )!!;
}

fn QueueSelection default_queue_selection(DeviceCaps* caps = null) @private {
    bool async_compute = caps != null && caps.async_compute;
    return queue_selection_for_requirements(
        requirements:               default_queue_requirements(),
        compute_aliases_graphics:    !async_compute,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    !async_compute,
    )!!;
}

fn bool queue_selection_is_well_formed(QueueSelection selection) @private {
    QueueRequirements requirements = { .counts = selection.counts };
    if (!queue_requirements_are_valid(requirements)
        || selection.counts.graphics > 1
        || selection.counts.compute > 1
        || selection.counts.transfer > 1) {
        return false;
    }
    QueueSelection canonical = queue_selection_for_requirements(
        requirements:               requirements,
        compute_aliases_graphics:    selection.counts.graphics > 0
            && selection.counts.compute > 0
            && selection.compute_id == selection.graphics_id,
        transfer_aliases_graphics:   selection.counts.graphics > 0
            && selection.counts.transfer > 0
            && selection.transfer_id == selection.graphics_id,
        transfer_aliases_compute:    selection.counts.compute > 0
            && selection.counts.transfer > 0
            && selection.transfer_id == selection.compute_id,
    )!!;
    return selection.graphics_id == canonical.graphics_id
        && selection.compute_id == canonical.compute_id
        && selection.transfer_id == canonical.transfer_id
        && queue_roles_equal(
            selection.graphics_roles,
            canonical.graphics_roles,
        )
        && queue_roles_equal(
            selection.compute_roles,
            canonical.compute_roles,
        )
        && queue_roles_equal(
            selection.transfer_roles,
            canonical.transfer_roles,
        );
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
    if (selection.counts.graphics > 0
        && queue.id == selection.graphics_id
        && queue_roles_equal(queue.roles, selection.graphics_roles)) {
        return true;
    }
    if (selection.counts.compute > 0
        && queue.id == selection.compute_id
        && queue_roles_equal(queue.roles, selection.compute_roles)) {
        return true;
    }
    return selection.counts.transfer > 0
        && queue.id == selection.transfer_id
        && queue_roles_equal(queue.roles, selection.transfer_roles);
}

fn void? validate_queue(
    DeviceOperation* operation,
    Device* device,
    Queue queue,
) @private {
    if (operation == null || !operation.active || device == null || !queue.is_valid()) {
        return INVALID_HANDLE~;
    }
    if (queue.device != *device || !queue_id_is_selected(operation, queue)) {
        return INVALID_HANDLE~;
    }
}

<* Return selected queue counts by semantic role; aliased roles count the same
   queue. INVALID_HANDLE if the device is not live. *>
fn QueueCounts? get_queue_counts(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return selected_queue_counts(&operation);
}

<* Return a device-owned queue identity for a semantic role and index. Aliased
   roles return the same token. A dead device faults INVALID_HANDLE; an
   unavailable index faults INVALID_ARGUMENT. *>
fn Queue? get_queue(Device* device, QueueKind kind, uint index = 0) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (index >= selected_queue_count(&operation, kind)) return INVALID_ARGUMENT~;
    return {
        .device = *device,
        .id     = selected_queue_id(&operation, kind) + index,
        .roles  = selected_queue_roles(&operation, kind),
    };
}

<* Return a live queue's device-local identity and supported roles.
   INVALID_HANDLE if the token is stale, foreign, or malformed. *>
fn QueueInfo? get_queue_info(Device* device, Queue queue) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, device, queue)!;
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
