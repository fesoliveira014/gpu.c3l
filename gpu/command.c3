module gpu;


fn void? validate_dispatch_groups(Vec3u groups, Vec3u limit) @private {
    if (groups.x > limit.x || groups.y > limit.y || groups.z > limit.z) {
        return INVALID_ARGUMENT~;
    }
}

fn usz? indirect_argument_bytes(
    usz argument_size,
    uint count,
    uint max_count,
) @private {
    if (count > max_count) return INVALID_ARGUMENT~;
    if (count == 0) return 0;
    usz size_count = (usz)count;
    if (argument_size > usz::max / size_count) return INVALID_ARGUMENT~;
    return argument_size * size_count;
}

fn usz? generated_work_bytes(
    usz record_size,
    uint count,
    uint max_count,
) @private {
    return indirect_argument_bytes(record_size, count, max_count);
}

fn CommandEncoderState*? command_encoder(
    Device device,
    CommandListHandle handle,
    void* opaque,
    CommandEncoderPhase phase,
    bool allow_lost = false,
) @private {
    if (opaque == null || !device.is_valid() || device.reserved != 0) {
        return INVALID_HANDLE~;
    }
    if (device.index >= MAX_DEVICES || handle.index >= MAX_DEVICE_COMMANDS
        || !handle.is_valid()) {
        return INVALID_HANDLE~;
    }
    CommandEncoderState* encoder = (CommandEncoderState*)opaque;
    if (encoder != &device_slots[device.index].command_encoders[handle.index]) {
        return INVALID_HANDLE~;
    }
    if (encoder.device != device || encoder.handle != handle
        || encoder.ops == null
        || encoder.backend_state == null || encoder.backend_command == null
        || encoder.device_lost == null) {
        return INVALID_HANDLE~;
    }
    if (encoder.phase != phase) {
        if ((phase == CommandEncoderPhase.RECORDING
                && encoder.phase == CommandEncoderPhase.EXECUTABLE)
            || (phase == CommandEncoderPhase.EXECUTABLE
                && encoder.phase == CommandEncoderPhase.RECORDING)) {
            return COMMAND_RECORDING_ERROR~;
        }
        return INVALID_HANDLE~;
    }
    if (!allow_lost
        && encoder.device_lost.load(AtomicOrdering.ACQUIRE)) {
        return DEVICE_LOST~;
    }
    return encoder;
}

fn CommandEncoderState*? recording_encoder(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    CommandEncoderState* encoder = command_encoder(
        device:     commands.device,
        handle:     commands.handle,
        opaque:     commands.encoder,
        phase:      CommandEncoderPhase.RECORDING,
        allow_lost: allow_lost,
    )!;
    if (!allow_lost) note_command_recording_call();
    return encoder;
}

fn CommandEncoderState*? executable_encoder(
    ExecutableCommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    return command_encoder(
        device:     commands.device,
        handle:     commands.handle,
        opaque:     commands.encoder,
        phase:      CommandEncoderPhase.EXECUTABLE,
        allow_lost: allow_lost,
    );
}

fn CommandEncoderState*? publish_command_encoder(
    DeviceOperation* operation,
    CommandList* commands,
    CommandBackendContext* backend,
) @private {
    if (operation == null || operation.slot == null || operation.data == null
        || commands == null || backend == null || backend.state == null
        || backend.command == null || operation.data.command_ops == null) {
        return BACKEND_ERROR~;
    }
    if (commands.handle.index >= MAX_DEVICE_COMMANDS) return BACKEND_ERROR~;
    CommandEncoderState* encoder =
        &operation.slot.command_encoders[commands.handle.index];
    if (encoder.phase != CommandEncoderPhase.INACTIVE) return BACKEND_ERROR~;
    *encoder = {
        .device          = commands.device,
        .handle          = commands.handle,
        .ops             = operation.data.command_ops,
        .backend_state   = backend.state,
        .backend_command = backend.command,
        .device_lost     = &operation.slot.lost,
        .phase           = CommandEncoderPhase.RECORDING,
    };
    note_command_policy_selection();
    commands.encoder = encoder;
    return encoder;
}

fn void transfer_command_encoder_to_executable(
    CommandEncoderState* encoder,
) @private {
    encoder.phase = CommandEncoderPhase.EXECUTABLE;
}

fn void invalidate_command_encoder(CommandEncoderState* encoder) @private {
    *encoder = {};
}

fn DeviceOperation? command_operation(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    note_command_device_registry_resolution();
    note_command_retained_pin_operation();
    if (commands == null) return INVALID_HANDLE~;
    DeviceOperation operation = has_retained_command_pin(commands)
        ? borrow_retained_device_operation(&commands.device, allow_lost)!
        : begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != resource_owner_tag(
        operation.data.resource_owner,
        ResourceOwnerKind.COMMAND,
    )) {
        end_device_operation(&operation);
        return INVALID_HANDLE~;
    }
    return operation;
}

fn DeviceOperation? executable_command_operation(
    ExecutableCommandList* commands,
    bool allow_lost = false,
) @private {
    note_command_device_registry_resolution();
    note_command_retained_pin_operation();
    if (commands == null) return INVALID_HANDLE~;
    DeviceOperation operation = has_retained_executable_command_pin(commands)
        ? borrow_retained_device_operation(&commands.device, allow_lost)!
        : begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != resource_owner_tag(
        operation.data.resource_owner,
        ResourceOwnerKind.COMMAND,
    )) {
        end_device_operation(&operation);
        return INVALID_HANDLE~;
    }
    return operation;
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
    DeviceOperation* operation,
    CommandAllocator* allocator,
) @private {
    if (allocator == null || !allocator.handle.is_valid()) {
        return INVALID_HANDLE~;
    }
    validate_queue(operation, &allocator.device, allocator.queue)!;
    validate_resource_owner(
        operation,
        allocator.handle.owner,
        ResourceOwnerKind.COMMAND_ALLOCATOR,
    )!;
}
