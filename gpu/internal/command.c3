module gpu::internal @private;

import gpu;

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

fn gpu::internal::vk::CommandRecord*? direct_command_record(
    gpu::internal::vk::CommandRecord* record,
    uint generation,
    uint device_identity,
) @private {
    if (record == null || generation == 0) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandTokenDeviceIdentity identity =
        (gpu::internal::CommandTokenDeviceIdentity)device_identity;
    if (identity.reserved != 0
        || identity.generation == 0
        || identity.index >= gpu::internal::MAX_DEVICES) {
        return gpu::INVALID_HANDLE~;
    }

    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[identity.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );
    if (control.state != gpu::internal::DEVICE_SLOT_LIVE
        || control.generation != identity.generation) {
        return gpu::INVALID_HANDLE~;
    }

    Device device = {
        .index      = identity.index,
        .generation = identity.generation,
    };
    if (record.device != device
        || record.identity.generation != generation) {
        return gpu::INVALID_HANDLE~;
    }
    return record;
}

fn gpu::internal::vk::CommandRecord*? recording_command(
    CommandList* commands,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
    )!;
    if (record.state != gpu::internal::CommandState.RECORDING
        && record.state
            != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    return record;
}

fn gpu::internal::vk::CommandRecord*? executable_command(
    ExecutableCommandList* commands,
) @private {
    if (commands == null) return gpu::INVALID_HANDLE~;
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
    )!;
    if (record.state != gpu::internal::CommandState.EXECUTABLE) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    return record;
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
