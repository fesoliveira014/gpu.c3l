module gpu::internal @private;

import gpu;

fn bool CommandListHandle.is_valid(self) =>
    self.owner != 0 && self.generation != 0;

fn bool CommandListHandle.equals(
    self,
    CommandListHandle other,
) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
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

fn gpu::internal::CommandRecord*? direct_command_record(
    void* opaque_record,
    uint generation,
) @private {
    if (opaque_record == null || generation == 0) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandRecord* record =
        (gpu::internal::CommandRecord*)opaque_record;
    if (record.identity.generation != generation) return gpu::INVALID_HANDLE~;
    return record;
}

fn gpu::internal::CommandRecord*? recording_command(
    CommandList* commands,
    bool count_command = true,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
    )!;
    if (record.state != gpu::internal::CommandState.RECORDING
        && record.state
            != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (count_command) gpu::internal::note_command_recording_call();
    return record;
}

fn gpu::internal::CommandRecord*? executable_command(
    ExecutableCommandList* commands,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
    )!;
    if (record.state != gpu::internal::CommandState.EXECUTABLE) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    return record;
}

fn void publish_command_list(
    gpu::internal::CommandRecord* record,
    CommandList* commands,
) @private {
    commands.record = record;
    commands.generation = record.identity.generation;
}

fn void publish_executable_command_list(
    gpu::internal::CommandRecord* record,
    ExecutableCommandList* commands,
) @private {
    commands.record = record;
    commands.generation = record.identity.generation;
}

fn gpu::internal::DeviceOperation? command_operation(
    gpu::internal::CommandRecord* record,
    bool allow_lost = false,
) @private {
    gpu::internal::note_command_device_registry_resolution();
    gpu::internal::note_command_retained_pin_operation();
    if (record == null) return gpu::INVALID_HANDLE~;
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
