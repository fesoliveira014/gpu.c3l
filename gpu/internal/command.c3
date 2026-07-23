module gpu::internal @private;

import gpu;

const uint COMMAND_LEASE_DEVICE_GENERATION_SHIFT @private = 32;
const uint COMMAND_LEASE_DEVICE_INDEX_SHIFT @private = 56;
const uint COMMAND_LEASE_OWNER_SHIFT @private = 64;
$assert uint::size * 8 == COMMAND_LEASE_DEVICE_GENERATION_SHIFT;
$assert COMMAND_LEASE_DEVICE_GENERATION_SHIFT + 24
    == COMMAND_LEASE_DEVICE_INDEX_SHIFT;
$assert gpu::internal::MAX_DEVICES <= (1 << 6);
$assert COMMAND_LEASE_DEVICE_INDEX_SHIFT + 6 <= COMMAND_LEASE_OWNER_SHIFT;
$assert ulong::size * 8 == COMMAND_LEASE_OWNER_SHIFT;

fn gpu::internal::CommandLease make_command_lease(
    Device device,
    CommandListHandle handle,
) @private {
    return (gpu::internal::CommandLease)(
        ((uint128)handle.owner << COMMAND_LEASE_OWNER_SHIFT)
        | ((uint128)device.index << COMMAND_LEASE_DEVICE_INDEX_SHIFT)
        | ((uint128)device.generation << COMMAND_LEASE_DEVICE_GENERATION_SHIFT)
        | (uint128)handle.generation
    );
}

fn void? validate_dispatch_groups(Vec3u groups, Vec3u limit) @private {
    if (groups.x > limit.x || groups.y > limit.y || groups.z > limit.z) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn usz? indirect_argument_bytes(
    usz argument_size,
    uint count,
    uint max_count,
) @private {
    if (count > max_count) return gpu::INVALID_ARGUMENT~;
    if (count == 0) return 0;
    usz size_count = (usz)count;
    if (argument_size > usz::max / size_count) return gpu::INVALID_ARGUMENT~;
    return argument_size * size_count;
}

fn usz? generated_work_bytes(
    usz record_size,
    uint count,
    uint max_count,
) @private {
    return indirect_argument_bytes(record_size, count, max_count);
}

fn gpu::internal::CommandEncoderState*? command_encoder(
    Device device,
    CommandListHandle handle,
    void* opaque,
    gpu::internal::CommandEncoderPhase phase,
) @private {
    if (opaque == null || !device.is_valid() || device.reserved != 0) {
        return gpu::INVALID_HANDLE~;
    }
    if (device.index >= gpu::internal::MAX_DEVICES || handle.index >= gpu::internal::MAX_DEVICE_COMMANDS
        || !handle.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::internal::note_command_encoder_cell_computation();
    gpu::internal::CommandEncoderState* encoder =
        &gpu::internal::device_slots[device.index].command_encoders[handle.index];
    if (opaque != encoder) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::internal::note_command_encoder_lease_comparison();
    if (encoder.lease != make_command_lease(device, handle)) {
        return gpu::INVALID_HANDLE~;
    }
    if (encoder.phase != phase) {
        if ((phase == gpu::internal::CommandEncoderPhase.RECORDING
                && encoder.phase == gpu::internal::CommandEncoderPhase.EXECUTABLE)
            || (phase == gpu::internal::CommandEncoderPhase.EXECUTABLE
                && encoder.phase == gpu::internal::CommandEncoderPhase.RECORDING)) {
            return gpu::COMMAND_RECORDING_ERROR~;
        }
        return gpu::INVALID_HANDLE~;
    }
    return encoder;
}

fn gpu::internal::CommandEncoderState*? recording_encoder(
    CommandList* commands,
    bool count_command = true,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandEncoderState* encoder = command_encoder(
        device:     commands.device,
        handle:     commands.handle,
        opaque:     commands.encoder,
        phase:      gpu::internal::CommandEncoderPhase.RECORDING,
    )!;
    if (count_command) gpu::internal::note_command_recording_call();
    return encoder;
}

fn gpu::internal::CommandEncoderState*? executable_encoder(
    ExecutableCommandList* commands,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    return command_encoder(
        device:     commands.device,
        handle:     commands.handle,
        opaque:     commands.encoder,
        phase:      gpu::internal::CommandEncoderPhase.EXECUTABLE,
    );
}

fn gpu::internal::CommandEncoderState*? publish_command_encoder(
    gpu::internal::DeviceOperation* operation,
    CommandList* commands,
    gpu::internal::CommandBackendContext* backend,
) @private {
    if (operation == null || operation.slot == null || operation.data == null
        || commands == null || backend == null || backend.state == null
        || backend.command == null || operation.data.command_ops == null) {
        return gpu::BACKEND_ERROR~;
    }
    if (commands.handle.index >= gpu::internal::MAX_DEVICE_COMMANDS) return gpu::BACKEND_ERROR~;
    gpu::internal::CommandEncoderState* encoder =
        &operation.slot.command_encoders[commands.handle.index];
    if (encoder.phase != gpu::internal::CommandEncoderPhase.INACTIVE) return gpu::BACKEND_ERROR~;
    *encoder = {
        .lease           = make_command_lease(commands.device, commands.handle),
        .ops             = operation.data.command_ops,
        .backend_state   = backend.state,
        .backend_command = backend.command,
        .phase           = gpu::internal::CommandEncoderPhase.RECORDING,
    };
    gpu::internal::note_command_policy_selection();
    commands.encoder = encoder;
    return encoder;
}

fn void transfer_command_encoder_to_executable(
    gpu::internal::CommandEncoderState* encoder,
) @private {
    encoder.phase = gpu::internal::CommandEncoderPhase.EXECUTABLE;
}

fn void invalidate_command_encoder(gpu::internal::CommandEncoderState* encoder) @private {
    *encoder = {};
}

fn gpu::internal::DeviceOperation? command_operation(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    gpu::internal::note_command_device_registry_resolution();
    gpu::internal::note_command_retained_pin_operation();
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::has_retained_command_pin(commands)
        ? gpu::internal::borrow_retained_device_operation(&commands.device, allow_lost)!
        : gpu::internal::begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != gpu::internal::resource_owner_tag(
        operation.data.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND,
    )) {
        gpu::internal::end_device_operation(&operation);
        return gpu::INVALID_HANDLE~;
    }
    return operation;
}

fn gpu::internal::DeviceOperation? executable_command_operation(
    ExecutableCommandList* commands,
    bool allow_lost = false,
) @private {
    gpu::internal::note_command_device_registry_resolution();
    gpu::internal::note_command_retained_pin_operation();
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::has_retained_executable_command_pin(commands)
        ? gpu::internal::borrow_retained_device_operation(&commands.device, allow_lost)!
        : gpu::internal::begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != gpu::internal::resource_owner_tag(
        operation.data.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND,
    )) {
        gpu::internal::end_device_operation(&operation);
        return gpu::INVALID_HANDLE~;
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
    gpu::internal::DeviceOperation* operation,
    CommandAllocator* allocator,
) @private {
    if (allocator == null || !allocator.handle.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    validate_queue(operation, &allocator.device, allocator.queue)!;
    gpu::internal::validate_resource_owner(
        operation,
        allocator.handle.owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    )!;
}
