module gpu::internal @private;

import gpu;

fn usz? indirect_argument_bytes(usz argument_size, uint count) @private {
    if (count == 0) return 0;
    usz size_count = (usz)count;
    if (argument_size > usz::max / size_count) return gpu::INVALID_ARGUMENT~;
    return argument_size * size_count;
}

fn gpu::internal::vk::CommandRecord*? direct_command_record(
    gpu::internal::vk::CommandRecord* record,
    uint generation,
    uint device_identity,
    HandleKind kind,
) @private {
    if (record == null && generation == 0 && device_identity == 0) {
        return reject_handle(kind, HandleCause.ZERO)~;
    }
    gpu::internal::CommandTokenDeviceIdentity identity =
        (gpu::internal::CommandTokenDeviceIdentity)device_identity;
    if (record == null
        || generation == 0
        || identity.reserved != 0
        || identity.generation == 0
        || identity.index >= gpu::internal::MAX_DEVICES) {
        return reject_handle(kind, HandleCause.MALFORMED)~;
    }

    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[identity.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );
    if (control.generation != identity.generation) {
        return reject_handle(kind, HandleCause.STALE)~;
    }
    if (control.state == gpu::internal::DEVICE_SLOT_CLOSING) {
        return reject_handle(kind, HandleCause.DESTROY_OVERLAP)~;
    }
    if (control.state != gpu::internal::DEVICE_SLOT_LIVE) {
        return reject_handle(kind, HandleCause.STALE)~;
    }

    Device device = {
        .index      = identity.index,
        .generation = identity.generation,
    };
    if (record.device != device
        || record.identity.generation != generation) {
        return reject_handle(kind, HandleCause.STALE)~;
    }
    return record;
}

<* Resolve a recording token. A record that has moved past recording was
   consumed through another copy of the token. *>
fn gpu::internal::vk::CommandRecord*? recording_command(
    CommandList* commands,
) @private {
    if (commands == null) return reject_handle(HandleKind.COMMAND_LIST, HandleCause.NULL_POINTER)~;
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
        HandleKind.COMMAND_LIST,
    )!;
    switch (record.state) {
        case gpu::internal::CommandState.RECORDING:
        case gpu::internal::CommandState.RECORDING_RENDER_PASS:
            return record;
        case gpu::internal::CommandState.EXECUTABLE:
        case gpu::internal::CommandState.SUBMITTING:
        case gpu::internal::CommandState.SUBMITTED:
            return reject_handle(HandleKind.COMMAND_LIST, HandleCause.CONSUMED)~;
        default:
            return reject_handle(HandleKind.COMMAND_LIST, HandleCause.STALE)~;
    }
}

fn gpu::internal::vk::CommandRecord* required_recording_command(
    CommandList* commands,
    String operation,
) @private {
    gpu::internal::vk::CommandRecord*? record = recording_command(commands);
    if (catch err = record) {
        terminate_rejected_handle(err, operation);
        abort("gpu::%s: invalid recording command: %s", operation, err);
    }
    return record;
}

<* Resolve an executable token. A submitted record was consumed through
   another copy of the token; a recording record is a phase error. *>
fn gpu::internal::vk::CommandRecord*? executable_command(
    ExecutableCommandList* commands,
) @private {
    if (commands == null) {
        return reject_handle(HandleKind.EXECUTABLE_COMMAND_LIST, HandleCause.NULL_POINTER)~;
    }
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
        HandleKind.EXECUTABLE_COMMAND_LIST,
    )!;
    return executable_record_phase(record);
}

fn gpu::internal::vk::CommandRecord*? executable_record_phase(
    gpu::internal::vk::CommandRecord* record,
) @private {
    switch (record.state) {
        case gpu::internal::CommandState.EXECUTABLE:
            return record;
        case gpu::internal::CommandState.SUBMITTED:
            return reject_handle(HandleKind.EXECUTABLE_COMMAND_LIST, HandleCause.CONSUMED)~;
        case gpu::internal::CommandState.INACTIVE:
            return reject_handle(HandleKind.EXECUTABLE_COMMAND_LIST, HandleCause.STALE)~;
        default:
            return gpu::COMMAND_RECORDING_ERROR~;
    }
}

fn void publish_command_list(
    gpu::internal::vk::CommandRecord* record,
    CommandList* commands,
) @private {
    commands.record = record;
    commands.generation = record.identity.generation;
    commands.device_identity =
        gpu::internal::command_token_device_identity(record.device);
}

fn void publish_executable_command_list(
    gpu::internal::vk::CommandRecord* record,
    ExecutableCommandList* commands,
) @private {
    commands.record = record;
    commands.generation = record.identity.generation;
    commands.device_identity =
        gpu::internal::command_token_device_identity(record.device);
}

fn gpu::internal::DeviceOperation? command_operation(
    gpu::internal::vk::CommandRecord* record,
    bool allow_lost = false,
) @private {
    if (record == null) return reject_handle(HandleKind.COMMAND_LIST, HandleCause.INVARIANT)~;
    return gpu::internal::borrow_retained_device_operation(
        &record.device,
        allow_lost,
    );
}

fn bool generated_work_kind_valid(GeneratedWorkKind kind) @private {
    switch (kind) {
        case GeneratedWorkKind.DRAW:
        case GeneratedWorkKind.DRAW_INDEXED:
        case GeneratedWorkKind.DISPATCH:
            return true;
        default:
            return false;
    }
}

fn void? validate_command_allocator(
    gpu::internal::DeviceOperation* operation,
    CommandAllocator* allocator,
) @private {
    if (allocator == null) {
        return reject_handle(HandleKind.COMMAND_ALLOCATOR, HandleCause.NULL_POINTER)~;
    }
    if (!allocator.handle.is_valid()) {
        return reject_handle(HandleKind.COMMAND_ALLOCATOR, HandleCause.ZERO)~;
    }
    validate_queue(operation, &allocator.device, allocator.queue)!;
    gpu::internal::validate_resource_owner(
        operation,
        allocator.handle.owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    )!;
}
