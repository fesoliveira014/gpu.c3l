module gpu::internal @private;

import gpu;

const uint QUEUE_ROLE_COUNT @private = 3;
const uint QUEUE_ROLE_MASK @private = 0x7;
const uint COMPLETION_QUEUE_BITS @private = 2;
const ulong COMPLETION_QUEUE_MASK @private =
    (1ull << COMPLETION_QUEUE_BITS) - 1;
const ulong MAX_COMPLETION_SEQUENCE @private =
    ulong::max >> COMPLETION_QUEUE_BITS;
const uint MAX_SELECTED_QUEUE_IDENTITIES @private = QUEUE_ROLE_COUNT;
const uint MAX_SUBMIT_COMPLETION_WAITS @private = 1024;
const uint MAX_SUBMIT_COMMAND_LISTS @private = 1024;
$assert(MAX_SELECTED_QUEUE_IDENTITIES <= 1 << COMPLETION_QUEUE_BITS);


struct QueueSelection @private {
    QueueRoles selected;
    QueueRoles distinct;
    uint identity_count;
    ushort graphics_id;
    ushort compute_id;
    ushort transfer_id;
    QueueRoles[MAX_SELECTED_QUEUE_IDENTITIES] identity_roles;
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
        return gpu::INVALID_HANDLE~;
    }
    if (sequence == 0 || sequence > MAX_COMPLETION_SEQUENCE) {
        return gpu::INVALID_ARGUMENT~;
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
    if (sequence == null) return gpu::INVALID_ARGUMENT~;
    if (*sequence == MAX_COMPLETION_SEQUENCE) return gpu::SLOT_TABLE_FULL~;
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

fn bool queue_role_is_selected(QueueRoles selected, QueueKind kind) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return selected.graphics;
        case QueueKind.COMPUTE:  return selected.compute;
        case QueueKind.TRANSFER: return selected.transfer;
    }
}

fn uint queue_selection_id(
    QueueSelection* selection,
    QueueKind kind,
) @private {
    switch (kind) {
        case QueueKind.GRAPHICS: return (uint)selection.graphics_id;
        case QueueKind.COMPUTE:  return (uint)selection.compute_id;
        case QueueKind.TRANSFER: return (uint)selection.transfer_id;
    }
}

fn QueueRoles queue_selection_roles(
    QueueSelection* selection,
    QueueKind kind,
) @private {
    return selection.identity_roles[queue_selection_id(selection, kind)];
}

fn void? add_queue_role_identity(
    QueueSelection* selection,
    QueueKind kind,
    ulong native_id,
    ulong[] identity_keys,
) @private {
    uint identity = selection.identity_count;
    for (uint known = 0; known < selection.identity_count; known++) {
        if (identity_keys[known] == native_id) {
            identity = known;
            break;
        }
    }
    if (identity == selection.identity_count) {
        if (selection.identity_count == MAX_SELECTED_QUEUE_IDENTITIES) {
            return gpu::INVALID_ARGUMENT~;
        }
        identity_keys[identity] = native_id;
        selection.identity_count++;
    }

    QueueRoles* roles = &selection.identity_roles[identity];
    switch (kind) {
        case QueueKind.GRAPHICS:
            if (roles.graphics) return gpu::INVALID_ARGUMENT~;
            roles.graphics = true;
            selection.graphics_id = (ushort)identity;
        case QueueKind.COMPUTE:
            if (roles.compute) return gpu::INVALID_ARGUMENT~;
            roles.compute = true;
            selection.compute_id = (ushort)identity;
        case QueueKind.TRANSFER:
            if (roles.transfer) return gpu::INVALID_ARGUMENT~;
            roles.transfer = true;
            selection.transfer_id = (ushort)identity;
    }
}

fn QueueSelection? queue_selection_from_native_ids(
    QueueRequest request,
    ulong graphics,
    ulong compute,
    ulong transfer,
) @private {
    if (!gpu::internal::queue_request_is_valid(request)) {
        return gpu::INVALID_ARGUMENT~;
    }

    QueueSelection selection = {
        .selected = request.required,
        .distinct = request.distinct,
    };
    ulong[MAX_SELECTED_QUEUE_IDENTITIES] identity_keys;
    if (request.required.graphics) {
        add_queue_role_identity(
            selection:     &selection,
            kind:          QueueKind.GRAPHICS,
            native_id:     graphics,
            identity_keys: identity_keys[..],
        )!;
    }
    if (request.required.compute) {
        add_queue_role_identity(
            selection:     &selection,
            kind:          QueueKind.COMPUTE,
            native_id:     compute,
            identity_keys: identity_keys[..],
        )!;
    }
    if (request.required.transfer) {
        add_queue_role_identity(
            selection:     &selection,
            kind:          QueueKind.TRANSFER,
            native_id:     transfer,
            identity_keys: identity_keys[..],
        )!;
    }

    for (uint identity = 0; identity < selection.identity_count; identity++) {
        QueueRoles roles = selection.identity_roles[identity];
        if (queue_roles_violate_distinct(roles, request.distinct)) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
    }
    return selection;
}

fn QueueSelection? queue_selection_for_request(
    QueueRequest request,
    bool compute_aliases_graphics,
    bool transfer_aliases_graphics,
    bool transfer_aliases_compute,
) @private {
    if (!gpu::internal::queue_request_is_valid(request)) {
        return gpu::INVALID_ARGUMENT~;
    }

    ulong next_native_id;
    ulong graphics_native_id;
    ulong compute_native_id;
    ulong transfer_native_id;
    if (request.required.graphics) {
        graphics_native_id = next_native_id++;
    }
    if (request.required.compute) {
        compute_native_id = request.required.graphics
            && compute_aliases_graphics
            ? graphics_native_id
            : next_native_id++;
    }
    if (request.required.transfer) {
        if (request.required.graphics && transfer_aliases_graphics) {
            transfer_native_id = graphics_native_id;
        } else if (request.required.compute && transfer_aliases_compute) {
            transfer_native_id = compute_native_id;
        } else {
            transfer_native_id = next_native_id++;
        }
    }
    return queue_selection_from_native_ids(
        request:  request,
        graphics: graphics_native_id,
        compute:  compute_native_id,
        transfer: transfer_native_id,
    );
}

fn QueueSelection queue_selection_for_aliases(
    bool compute_aliases_graphics,
    bool transfer_aliases_graphics,
    bool transfer_aliases_compute,
) @private {
    return queue_selection_for_request(
        request:                    gpu::internal::default_queue_request(),
        compute_aliases_graphics:   compute_aliases_graphics,
        transfer_aliases_graphics:  transfer_aliases_graphics,
        transfer_aliases_compute:   transfer_aliases_compute,
    )!!;
}

fn QueueSelection default_queue_selection(DeviceCaps* caps = null) @private {
    bool async_compute = caps != null && caps.async_compute;
    return queue_selection_for_request(
        request:                    gpu::internal::default_queue_request(),
        compute_aliases_graphics:   !async_compute,
        transfer_aliases_graphics:  true,
        transfer_aliases_compute:   !async_compute,
    )!!;
}

fn bool queue_selection_is_well_formed(QueueSelection* selection) @private {
    if (selection == null) return false;
    QueueRequest request = {
        .required = selection.selected,
        .distinct = selection.distinct,
    };
    if (!gpu::internal::queue_request_is_valid(request)
        || selection.identity_count == 0
        || selection.identity_count > MAX_SELECTED_QUEUE_IDENTITIES) {
        return false;
    }

    QueueRoles[MAX_SELECTED_QUEUE_IDENTITIES] expected_roles;
    if (selection.selected.graphics) {
        uint identity = queue_selection_id(selection, QueueKind.GRAPHICS);
        if (identity >= selection.identity_count) return false;
        expected_roles[identity].graphics = true;
    }
    if (selection.selected.compute) {
        uint identity = queue_selection_id(selection, QueueKind.COMPUTE);
        if (identity >= selection.identity_count) return false;
        expected_roles[identity].compute = true;
    }
    if (selection.selected.transfer) {
        uint identity = queue_selection_id(selection, QueueKind.TRANSFER);
        if (identity >= selection.identity_count) return false;
        expected_roles[identity].transfer = true;
    }

    for (uint identity = 0; identity < selection.identity_count; identity++) {
        QueueRoles roles = expected_roles[identity];
        if ((uint)roles == 0
            || !queue_roles_equal(roles, selection.identity_roles[identity])
            || queue_roles_violate_distinct(roles, selection.distinct)) {
            return false;
        }
    }
    return true;
}

fn QueueSelection* selected_queues(
    gpu::internal::DeviceOperation* operation,
) @private {
    return &operation.data.queues;
}

fn QueueRoles selected_queue_role_mask(
    gpu::internal::DeviceOperation* operation,
) @private {
    return selected_queues(operation).selected;
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
    gpu::internal::DeviceOperation* operation,
    QueueRoles access,
) @private {
    return operation != null
        && operation.active
        && (uint)access != 0
        && queue_roles_are_known(access)
        && queue_roles_are_subset(access, selected_queue_role_mask(operation));
}

fn void? validate_queue_access(
    gpu::internal::DeviceOperation* operation,
    QueueRoles access,
) @private {
    if (!queue_access_is_valid(operation, access)) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn void? validate_selected_queue_kind(
    gpu::internal::DeviceOperation* operation,
    QueueKind kind,
) @private {
    if (!queue_role_is_selected(selected_queue_role_mask(operation), kind)) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn uint selected_queue_id(
    gpu::internal::DeviceOperation* operation,
    QueueKind kind,
) @private {
    return queue_selection_id(selected_queues(operation), kind);
}

fn QueueRoles selected_queue_roles(
    gpu::internal::DeviceOperation* operation,
    QueueKind kind,
) @private {
    return queue_selection_roles(selected_queues(operation), kind);
}

fn bool queue_id_is_selected(
    gpu::internal::DeviceOperation* operation,
    Queue queue,
) @private {
    QueueSelection* selection = selected_queues(operation);
    return queue.id < selection.identity_count
        && queue_roles_equal(queue.roles, selection.identity_roles[queue.id]);
}

fn void? validate_queue(
    gpu::internal::DeviceOperation* operation,
    Device* device,
    Queue queue,
) @private {
    if (operation == null || !operation.active || device == null || !queue.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    if (queue.device != *device || !queue_id_is_selected(operation, queue)) {
        return gpu::INVALID_HANDLE~;
    }
}

fn void? validate_completion_point(
    gpu::internal::DeviceOperation* operation,
    Device* device,
    CompletionPoint point,
) @private {
    if (operation == null
        || !operation.active
        || device == null
        || !point.is_valid()
        || point.device != *device) {
        return gpu::INVALID_HANDLE~;
    }
    if (completion_point_queue_id(point)
        >= selected_queues(operation).identity_count) {
        return gpu::INVALID_HANDLE~;
    }
}
