module gpu;

const uint QUEUE_ROLE_COUNT @private = 3;
const uint QUEUE_ROLE_MASK @private = 0x7;
const uint COMPLETION_QUEUE_BITS @private = 10;
const ulong COMPLETION_QUEUE_MASK @private =
    (1ull << COMPLETION_QUEUE_BITS) - 1;
const ulong MAX_COMPLETION_SEQUENCE @private =
    ulong::max >> COMPLETION_QUEUE_BITS;
const uint MAX_SELECTED_QUEUE_IDENTITIES @private =
    MAX_REQUESTED_QUEUES_PER_ROLE * QUEUE_ROLE_COUNT;
const uint MAX_SUBMIT_COMPLETION_WAITS @private = 1024;
const uint MAX_SUBMIT_COMMAND_LISTS @private = 1024;
const Queue QUEUE_INVALID = {};
$assert(MAX_SELECTED_QUEUE_IDENTITIES - 1 <= ushort::max);
$assert(MAX_SELECTED_QUEUE_IDENTITIES <= 1 << COMPLETION_QUEUE_BITS);
$assert(CompletionPoint::size <= ulong::size * 2);

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

<* Semantic queues required at device creation. Each count is 0..255 and
   requests distinct identities within that role. A marked distinct role must
   be requested and may not alias another requested role. *>
struct QueueRequirements {
    QueueCounts counts;
    QueueRoles distinct;
}

struct QueueSelection @private {
    QueueCounts counts;
    QueueRoles distinct;
    uint identity_count;
    ushort[MAX_REQUESTED_QUEUES_PER_ROLE] graphics_ids;
    ushort[MAX_REQUESTED_QUEUES_PER_ROLE] compute_ids;
    ushort[MAX_REQUESTED_QUEUES_PER_ROLE] transfer_ids;
    QueueRoles[MAX_SELECTED_QUEUE_IDENTITIES] identity_roles;
}

<* Work submitted to an explicit queue. Every wait names its first destination
   stages. Non-zero readiness requires a graphics-capable queue and an explicit
   supported destination stage, and is consumed only by successful submission. *>
struct SubmitDesc {
    ExecutableCommandList[] command_lists;
    CompletionWait[]        completion_waits;
    SwapchainReadiness      readiness;
    StageMask               readiness_before;
}

fn bool Queue.is_valid(self) => self.device.is_valid();
fn bool Queue.equals(self, Queue other) @operator(==) {
    return self.device == other.device
        && self.id == other.id
        && (uint)self.roles == (uint)other.roles;
}

fn Device completion_point_device(CompletionPoint point) @private {
    return point.device;
}

fn uint completion_point_queue_id(CompletionPoint point) @private {
    return (uint)(point.payload & COMPLETION_QUEUE_MASK);
}

fn ulong completion_point_sequence(CompletionPoint point) @private {
    return point.payload >> COMPLETION_QUEUE_BITS;
}

fn CompletionPoint? make_completion_point(
    Queue queue,
    ulong sequence,
) @private {
    if (!queue.is_valid() || queue.id >= MAX_SELECTED_QUEUE_IDENTITIES) {
        return INVALID_HANDLE~;
    }
    if (sequence == 0 || sequence > MAX_COMPLETION_SEQUENCE) {
        return INVALID_ARGUMENT~;
    }
    return {
        .device  = queue.device,
        .payload = (sequence << COMPLETION_QUEUE_BITS) | (ulong)queue.id,
    };
}

fn CompletionPoint? next_completion_point(
    Queue queue,
    ulong* sequence,
) @private {
    if (sequence == null) return INVALID_ARGUMENT~;
    if (*sequence == MAX_COMPLETION_SEQUENCE) return SLOT_TABLE_FULL~;
    CompletionPoint point = make_completion_point(queue, *sequence + 1)!;
    (*sequence)++;
    return point;
}

fn bool queue_roles_equal(QueueRoles first, QueueRoles second) @private {
    return (uint)first == (uint)second;
}

fn bool queue_roles_violate_distinct(
    QueueRoles roles,
    QueueRoles distinct,
) @private {
    return (distinct.graphics
            && roles.graphics
            && (roles.compute || roles.transfer))
        || (distinct.compute
            && roles.compute
            && (roles.graphics || roles.transfer))
        || (distinct.transfer
            && roles.transfer
            && (roles.graphics || roles.compute));
}

fn uint queue_count_for_kind(QueueCounts counts, QueueKind kind) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return counts.graphics;
        case QueueKind.COMPUTE:  return counts.compute;
        case QueueKind.TRANSFER: return counts.transfer;
    }
}

fn uint queue_selection_id(
    QueueSelection* selection,
    QueueKind kind,
    uint index,
) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return (uint)selection.graphics_ids[index];
        case QueueKind.COMPUTE:  return (uint)selection.compute_ids[index];
        case QueueKind.TRANSFER: return (uint)selection.transfer_ids[index];
    }
}

fn QueueRoles queue_selection_roles(
    QueueSelection* selection,
    QueueKind kind,
    uint index,
) @private {
    return selection.identity_roles[queue_selection_id(selection, kind, index)];
}

fn void? add_queue_role_identities(
    QueueSelection* selection,
    QueueKind kind,
    ulong[] native_ids,
    ulong[] identity_keys,
) @private {
    foreach (index, native_id : native_ids) {
        uint identity = selection.identity_count;
        for (uint known = 0; known < selection.identity_count; known++) {
            if (identity_keys[known] == native_id) {
                identity = known;
                break;
            }
        }
        if (identity == selection.identity_count) {
            if (selection.identity_count == MAX_SELECTED_QUEUE_IDENTITIES) {
                return INVALID_ARGUMENT~;
            }
            identity_keys[identity] = native_id;
            selection.identity_count++;
        }

        QueueRoles* roles = &selection.identity_roles[identity];
        switch (kind) {
            case QueueKind.GRAPHICS:
                if (roles.graphics) return INVALID_ARGUMENT~;
                roles.graphics = true;
                selection.graphics_ids[index] = (ushort)identity;
            case QueueKind.COMPUTE:
                if (roles.compute) return INVALID_ARGUMENT~;
                roles.compute = true;
                selection.compute_ids[index] = (ushort)identity;
            case QueueKind.TRANSFER:
                if (roles.transfer) return INVALID_ARGUMENT~;
                roles.transfer = true;
                selection.transfer_ids[index] = (ushort)identity;
        }
    }
}

fn QueueSelection? queue_selection_from_native_ids(
    QueueRequirements requirements,
    ulong[] graphics,
    ulong[] compute,
    ulong[] transfer,
) @private {
    if (!queue_requirements_are_valid(requirements)) return INVALID_ARGUMENT~;
    if (graphics.len != (usz)requirements.counts.graphics
        || compute.len != (usz)requirements.counts.compute
        || transfer.len != (usz)requirements.counts.transfer) {
        return INVALID_ARGUMENT~;
    }

    QueueSelection selection = {
        .counts   = requirements.counts,
        .distinct = requirements.distinct,
    };
    ulong[MAX_SELECTED_QUEUE_IDENTITIES] identity_keys;
    add_queue_role_identities(
        selection:     &selection,
        kind:          QueueKind.GRAPHICS,
        native_ids:    graphics,
        identity_keys: identity_keys[..],
    )!;
    add_queue_role_identities(
        selection:     &selection,
        kind:          QueueKind.COMPUTE,
        native_ids:    compute,
        identity_keys: identity_keys[..],
    )!;
    add_queue_role_identities(
        selection:     &selection,
        kind:          QueueKind.TRANSFER,
        native_ids:    transfer,
        identity_keys: identity_keys[..],
    )!;

    for (uint identity = 0; identity < selection.identity_count; identity++) {
        QueueRoles roles = selection.identity_roles[identity];
        if (queue_roles_violate_distinct(roles, requirements.distinct)) {
            return UNSUPPORTED_FEATURE~;
        }
    }
    return selection;
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

    ulong next_native_id;
    ulong graphics_native_id;
    ulong compute_native_id;
    ulong transfer_native_id;
    if (requirements.counts.graphics > 0) {
        graphics_native_id = next_native_id++;
    }
    if (requirements.counts.compute > 0) {
        compute_native_id = requirements.counts.graphics > 0
            && compute_aliases_graphics
            ? graphics_native_id
            : next_native_id++;
    }
    if (requirements.counts.transfer > 0) {
        if (requirements.counts.graphics > 0 && transfer_aliases_graphics) {
            transfer_native_id = graphics_native_id;
        } else if (requirements.counts.compute > 0 && transfer_aliases_compute) {
            transfer_native_id = compute_native_id;
        } else {
            transfer_native_id = next_native_id++;
        }
    }
    ulong[1] graphics = { graphics_native_id };
    ulong[1] compute = { compute_native_id };
    ulong[1] transfer = { transfer_native_id };
    return queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[:requirements.counts.graphics],
        compute:      compute[:requirements.counts.compute],
        transfer:     transfer[:requirements.counts.transfer],
    );
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

fn bool queue_selection_is_well_formed(QueueSelection* selection) @private {
    if (selection == null) return false;
    QueueRequirements requirements = {
        .counts   = selection.counts,
        .distinct = selection.distinct,
    };
    if (!queue_requirements_are_valid(requirements)
        || selection.identity_count == 0
        || selection.identity_count > MAX_SELECTED_QUEUE_IDENTITIES) {
        return false;
    }

    QueueRoles[MAX_SELECTED_QUEUE_IDENTITIES] expected_roles;
    for (uint index = 0; index < selection.counts.graphics; index++) {
        uint identity = queue_selection_id(selection, QueueKind.GRAPHICS, index);
        if (identity >= selection.identity_count
            || expected_roles[identity].graphics) {
            return false;
        }
        expected_roles[identity].graphics = true;
    }
    for (uint index = 0; index < selection.counts.compute; index++) {
        uint identity = queue_selection_id(selection, QueueKind.COMPUTE, index);
        if (identity >= selection.identity_count
            || expected_roles[identity].compute) {
            return false;
        }
        expected_roles[identity].compute = true;
    }
    for (uint index = 0; index < selection.counts.transfer; index++) {
        uint identity = queue_selection_id(selection, QueueKind.TRANSFER, index);
        if (identity >= selection.identity_count
            || expected_roles[identity].transfer) {
            return false;
        }
        expected_roles[identity].transfer = true;
    }

    for (uint identity = 0; identity < selection.identity_count; identity++) {
        QueueRoles roles = expected_roles[identity];
        if ((uint)roles == 0
            || !queue_roles_equal(roles, selection.identity_roles[identity])) {
            return false;
        }
        if (queue_roles_violate_distinct(roles, selection.distinct)) {
            return false;
        }
    }
    return true;
}

fn QueueSelection* selected_queues(DeviceOperation* operation) @private {
    return &operation.data.queues;
}

fn QueueCounts selected_queue_counts(DeviceOperation* operation) @private {
    return selected_queues(operation).counts;
}

fn uint selected_queue_count(DeviceOperation* operation, QueueKind kind) @private {
    return queue_count_for_kind(selected_queue_counts(operation), kind);
}

fn QueueRoles selected_queue_role_mask(DeviceOperation* operation) @private {
    QueueCounts counts = selected_queue_counts(operation);
    return {
        .graphics = counts.graphics > 0,
        .compute  = counts.compute > 0,
        .transfer = counts.transfer > 0,
    };
}

fn bool queue_roles_are_known(QueueRoles roles) @private {
    return ((uint)roles & ~QUEUE_ROLE_MASK) == 0;
}

fn bool queue_roles_are_subset(QueueRoles roles, QueueRoles set) @private {
    return ((uint)roles & ~(uint)set) == 0;
}

fn bool queue_access_allows(QueueRoles access, QueueKind queue) @private {
    if ((uint)access == 0 || !queue_roles_are_known(access)) return false;
    switch (queue) {
        case QueueKind.GRAPHICS: return access.graphics;
        case QueueKind.COMPUTE:  return access.compute;
        case QueueKind.TRANSFER: return access.transfer;
        default:                 return false;
    }
}

fn QueueRoles queue_roles_for_kind(QueueKind kind) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return { .graphics };
        case QueueKind.COMPUTE:  return { .compute };
        case QueueKind.TRANSFER: return { .transfer };
    }
}

fn bool queue_access_intersects(
    QueueRoles access,
    QueueRoles queue_roles,
) @private {
    return (uint)access != 0
        && queue_roles_are_known(access)
        && (uint)queue_roles != 0
        && queue_roles_are_known(queue_roles)
        && ((uint)access & (uint)queue_roles) != 0;
}

fn bool queue_access_is_valid(
    DeviceOperation* operation,
    QueueRoles access,
) @private {
    return operation != null
        && operation.active
        && (uint)access != 0
        && queue_roles_are_known(access)
        && queue_roles_are_subset(access, selected_queue_role_mask(operation));
}

fn void? validate_queue_access(DeviceOperation* operation, QueueRoles access) @private {
    if (!queue_access_is_valid(operation, access)) return INVALID_ARGUMENT~;
}

fn void? validate_selected_queue_kind(
    DeviceOperation* operation,
    QueueKind kind,
) @private {
    if (selected_queue_count(operation, kind) == 0) return INVALID_ARGUMENT~;
}

fn uint selected_queue_id(
    DeviceOperation* operation,
    QueueKind kind,
    uint index,
) @private {
    return queue_selection_id(selected_queues(operation), kind, index);
}

fn QueueRoles selected_queue_roles(
    DeviceOperation* operation,
    QueueKind kind,
    uint index,
) @private {
    return queue_selection_roles(selected_queues(operation), kind, index);
}

fn bool queue_id_is_selected(DeviceOperation* operation, Queue queue) @private {
    QueueSelection* selection = selected_queues(operation);
    return queue.id < selection.identity_count
        && queue_roles_equal(queue.roles, selection.identity_roles[queue.id]);
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

fn void? validate_completion_point(
    DeviceOperation* operation,
    Device* device,
    CompletionPoint point,
) @private {
    if (operation == null
        || !operation.active
        || device == null
        || !point.is_valid()
        || point.device != *device) {
        return INVALID_HANDLE~;
    }
    if (completion_point_queue_id(point)
        >= selected_queues(operation).identity_count) {
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
        .id     = selected_queue_id(&operation, kind, index),
        .roles  = selected_queue_roles(&operation, kind, index),
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

<* Submit executable command lists to `queue` and return reusable progress.
   Successful submission consumes every list. *>
fn CompletionPoint? submit(Queue queue, SubmitDesc* desc) {
    Device device = queue.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, &device, queue)!;
    if (desc == null) return INVALID_ARGUMENT~;
    if (desc.command_lists.len > MAX_SUBMIT_COMMAND_LISTS) {
        return INVALID_ARGUMENT~;
    }
    if (desc.completion_waits.len > MAX_SUBMIT_COMPLETION_WAITS) {
        return INVALID_ARGUMENT~;
    }
    foreach (&commands : desc.command_lists) {
        if (commands.device != device) return INVALID_HANDLE~;
        executable_encoder(commands)!;
        validate_resource_owner(
            &operation,
            commands.handle.owner,
            ResourceOwnerKind.COMMAND,
        )!;
    }
    foreach (wait : desc.completion_waits) {
        validate_completion_point(&operation, &device, wait.point)!;
        if (!submit_wait_stage_mask_valid(wait.before)) {
            return INVALID_ARGUMENT~;
        }
    }
    if (desc.readiness != SWAPCHAIN_READINESS_INVALID) {
        if (!desc.readiness.is_valid()
            || desc.readiness.device != device) {
            return INVALID_HANDLE~;
        }
        if (!queue.roles.graphics) return INVALID_ARGUMENT~;
        if (!submit_wait_stage_mask_valid(desc.readiness_before)) {
            return INVALID_ARGUMENT~;
        }
    } else if ((uint)desc.readiness_before != 0) {
        return INVALID_ARGUMENT~;
    }
    note_command_lifecycle_vtable_dispatch();
    CompletionPoint point = operation.vtable.submit(queue, desc)!;
    foreach (&commands : desc.command_lists) {
        release_executable_command_pin_lease(&operation, commands);
        *commands = {};
    }
    release_device_children(&operation, (ulong)desc.command_lists.len);
    (void)release_retained_device_pins(
        &operation,
        (uint)desc.command_lists.len,
    );
    return point;
}
