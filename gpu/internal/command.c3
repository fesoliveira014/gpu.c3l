module gpu::internal @private;

import gpu;

fn usz indirect_argument_bytes(usz argument_size, uint count) @private {
    if (count == 0) return 0;
    usz size_count = (usz)count;
    assert(
        argument_size <= usz::max / size_count,
        "indirect argument byte size must fit in usz"
    );
    return argument_size * size_count;
}

<* True when the allocator description's command-buffer capacity, after the
   zero default, times its geometry capacity fits uint. *>
fn bool command_allocator_geometry_capacity_fits(gpu::CommandAllocatorDesc* desc) @private {
    uint geometries = desc.max_acceleration_structure_geometries_per_build;
    if (geometries == 0) return true;
    uint capacity = desc.command_buffer_capacity == 0
        ? gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY
        : desc.command_buffer_capacity;
    return capacity <= uint::max / geometries;
}

<* True when the span is nonempty and holds at least `bytes` bytes. *>
fn bool span_holds_bytes(gpu::GpuSpan span, usz bytes) @private {
    return span.size > 0 && span.size >= bytes;
}

<* True when the span is nonempty and holds `count` records of `record_size`
   bytes without the product overflowing. A zero count needs only a nonempty
   span. *>
fn bool span_holds_records(gpu::GpuSpan span, usz record_size, uint count) @private {
    if (span.size == 0) return false;
    if (count == 0) return true;
    usz records = (usz)count;
    return record_size <= usz::max / records && record_size * records <= span.size;
}

<* True when the index span holds `count` indices of the index type. *>
fn bool index_span_holds_indices(
    gpu::GpuSpan span,
    gpu::IndexType index_type,
    uint count,
) @private {
    usz index_bytes = index_type == gpu::IndexType.U16 ? 2 : 4;
    return span_holds_records(span, index_bytes, count);
}

<* True when generated-work spans can hold `max_count` records and the count.
   Nothing is read for a zero maximum. *>
fn bool generated_spans_hold(
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    usz record_size,
    uint max_count,
) @private {
    if (max_count == 0) return true;
    return span_holds_records(records, record_size, max_count)
        && span_holds_bytes(count_span, uint::size);
}

<* Look up a command token's record without terminating: null unless the
   record is live on a live device. *>
fn gpu::internal::vk::CommandRecord* find_command_record(
    gpu::internal::vk::CommandRecord* record,
    uint generation,
    uint device_identity,
) @private {
    gpu::internal::CommandTokenDeviceIdentity identity =
        (gpu::internal::CommandTokenDeviceIdentity)device_identity;
    if (record == null
        || generation == 0
        || identity.reserved != 0
        || identity.generation == 0
        || identity.index >= gpu::internal::MAX_DEVICES) {
        return null;
    }
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[identity.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != identity.generation
        || control.state != gpu::internal::DEVICE_SLOT_LIVE) {
        return null;
    }
    Device device = {
        .index      = identity.index,
        .generation = identity.generation,
    };
    if (record.device != device || record.identity.generation != generation) return null;
    return record;
}

fn gpu::internal::vk::CommandRecord* direct_command_record(
    gpu::internal::vk::CommandRecord* record,
    uint generation,
    uint device_identity,
    HandleKind kind,
) @private {
    gpu::internal::vk::CommandRecord* found = find_command_record(record, generation, device_identity);
    if (found != null) return found;
    if (record == null && generation == 0 && device_identity == 0) {
        abort("gpu: zero %s", kind.type_name);
    }
    gpu::internal::CommandTokenDeviceIdentity identity =
        (gpu::internal::CommandTokenDeviceIdentity)device_identity;
    if (record == null
        || generation == 0
        || identity.reserved != 0
        || identity.generation == 0
        || identity.index >= gpu::internal::MAX_DEVICES) {
        abort("gpu: malformed %s", kind.type_name);
    }
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[identity.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation == identity.generation
        && control.state == gpu::internal::DEVICE_SLOT_CLOSING) {
        abort("gpu: %s used concurrently with destroy_device", kind.type_name);
    }
    abort("gpu: stale %s", kind.type_name);
}

<* Resolve a recording token. A record that has moved past recording was
   consumed through another copy of the token. *>
fn gpu::internal::vk::CommandRecord* recording_command(
    CommandList* commands,
) @private {
    if (commands == null) abort("gpu: null CommandList");
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
        HandleKind.COMMAND_LIST,
    );
    switch (record.state) {
        case gpu::internal::CommandState.RECORDING:
        case gpu::internal::CommandState.RECORDING_RENDER_PASS:
            return record;
        case gpu::internal::CommandState.EXECUTABLE:
        case gpu::internal::CommandState.SUBMITTING:
        case gpu::internal::CommandState.SUBMITTED:
            abort("gpu: consumed CommandList");
        default:
            abort("gpu: stale CommandList");
    }
}

<* Resolve an executable token. A submitted record was consumed through
   another copy of the token; a recording record is a phase error. *>
fn gpu::internal::vk::CommandRecord*? executable_command(
    ExecutableCommandList* commands,
) @private {
    if (commands == null) {
        abort("gpu: null ExecutableCommandList");
    }
    gpu::internal::vk::CommandRecord* record = direct_command_record(
        commands.record,
        commands.generation,
        commands.device_identity,
        HandleKind.EXECUTABLE_COMMAND_LIST,
    );
    return executable_record_phase(record);
}

fn gpu::internal::vk::CommandRecord*? executable_record_phase(
    gpu::internal::vk::CommandRecord* record,
) @private {
    switch (record.state) {
        case gpu::internal::CommandState.EXECUTABLE:
            return record;
        case gpu::internal::CommandState.SUBMITTED:
            abort("gpu: consumed ExecutableCommandList");
        case gpu::internal::CommandState.INACTIVE:
            abort("gpu: stale ExecutableCommandList");
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

fn gpu::internal::DeviceOperation command_operation(
    gpu::internal::vk::CommandRecord* record,
) @private {
    if (record == null) abort("gpu: corrupt CommandList");
    return gpu::internal::borrow_retained_device_operation(&record.device);
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

fn void validate_command_allocator(
    gpu::internal::DeviceOperation* operation,
    CommandAllocator* allocator,
) @private {
    if (allocator == null) {
        abort("gpu: null CommandAllocator");
    }
    if (!allocator.handle.is_valid()) {
        abort("gpu: zero CommandAllocator");
    }
    validate_queue(operation, &allocator.device, allocator.queue);
    gpu::internal::validate_resource_owner(
        operation,
        allocator.handle.owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    );
}
