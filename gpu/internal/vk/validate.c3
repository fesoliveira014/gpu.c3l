module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

struct ResolvedSpan {
    BufferSlot*       slot;
    BufferHandle      backing_buffer;
    usz               offset;
    usz               size;
    gpu::QueueRoles   access;
    gpu::MemoryClass   memory_class;
    void*             mapping;
    gpu::GpuAddress   address;
}

fn ResolvedSpan? resolve_span(VkDeviceState* state, gpu::GpuSpan* span) {
    if (state == null || span == null) return gpu::INVALID_HANDLE~;

    BufferSlot* slot;
    BufferHandle backing_buffer;
    usz offset;
    gpu::QueueRoles access;
    gpu::MemoryClass memory_class;
    if (span.owner == state.buffers.owner) {
        backing_buffer = {
            .owner      = span.owner,
            .index      = span.index,
            .generation = span.generation,
        };
        slot = state.buffers.get(backing_buffer)!;
        offset = span.offset;
        access = slot.access;
        memory_class = slot.memory_class;
    } else if (span.owner == state.allocations.owner) {
        gpu::GpuAllocation allocation = {
            .owner      = span.owner,
            .index      = span.index,
            .generation = span.generation,
        };
        AllocationSlot* allocation_slot = state.allocations.get(allocation)!;
        backing_buffer = BUFFER_HANDLE_INVALID;
        if (allocation_slot.texture_memory) return gpu::INVALID_ARGUMENT~;
        slot = &allocation_slot.backing;
        offset = span.offset;
        access = slot.access;
        memory_class = slot.memory_class;
    } else {
        return gpu::INVALID_HANDLE~;
    }

    if (span.size == 0) return gpu::INVALID_ARGUMENT~;
    if (!gpu::internal::range_fits(slot.size, offset, span.size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    void* mapping;
    if (slot.cpu_base != null) {
        if ((uptr)offset > uptr::max - (uptr)slot.cpu_base) {
            return gpu::INVALID_ARGUMENT~;
        }
        mapping = (char*)slot.cpu_base + offset;
    }
    gpu::GpuAddress address;
    if ((ulong)slot.gpu_base != 0) {
        if ((ulong)offset > ulong::max - (ulong)slot.gpu_base) {
            return gpu::INVALID_ARGUMENT~;
        }
        address = (gpu::GpuAddress)((ulong)slot.gpu_base + (ulong)offset);
    }
    return {
        .slot           = slot,
        .backing_buffer = backing_buffer,
        .offset         = offset,
        .size           = span.size,
        .access         = access,
        .memory_class    = memory_class,
        .mapping        = mapping,
        .address        = address,
    };
}

fn char[]? resolved_span_mapping(ResolvedSpan* span) {
    if (span == null || span.mapping == null || span.size == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return ((char*)span.mapping)[:span.size];
}

fn gpu::GpuAddress? resolved_span_address(ResolvedSpan* span) {
    if (span == null || (ulong)span.address == 0) return gpu::INVALID_ARGUMENT~;
    return span.address;
}

fn void? validate_operation_queue_access(
    VkDeviceState* state,
    gpu::QueueRoles queue_roles,
    gpu::QueueRoles access,
    gpu::DebugResourceKind resource_kind,
    uint index,
    uint generation,
    ZString debug_name,
    ZString operation,
    ZString rejected_field,
) @private {
    if (gpu::internal::queue_access_intersects(access, queue_roles)) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       resource_kind,
        has_public_identity: true,
        index:               index,
        generation:          generation,
        debug_name:          debug_name,
        rejected_field:      rejected_field,
        invariant:           "access must include the operation queue role",
        backend_text:        null,
    )~;
}

fn void? validate_texture_queue_access(
    VkDeviceState* state,
    gpu::QueueRoles queue_roles,
    TextureSlot* slot,
    gpu::TextureHandle handle,
    ZString operation,
) @private {
    return validate_operation_queue_access(
        state:          state,
        queue_roles:    queue_roles,
        access:         slot.access,
        resource_kind:  gpu::DebugResourceKind.TEXTURE,
        index:          handle.index,
        generation:     handle.generation,
        debug_name:     slot_name(&slot.debug_name[0]),
        operation:      operation,
        rejected_field: "TextureDesc.access",
    );
}

fn void? validate_texture_recording_access_only(
    VkDeviceState* state,
    VkCommandRecord* record,
    TextureSlot* slot,
    gpu::TextureHandle handle,
    ZString operation,
) @private {
    validate_texture_queue_access(
        state:     state,
        queue_roles: record.queue.roles,
        slot:      slot,
        handle:    handle,
        operation: operation,
    )!;
}

fn void? track_texture_command_reference(
    VkDeviceState* state,
    VkCommandRecord* record,
    gpu::TextureHandle handle,
) @private {
    return track_command_reference(
        state,
        record,
        handle.owner,
        handle.index,
        handle.generation,
    );
}

fn gpu::DebugResourceKind span_resource_kind(
    VkDeviceState* state,
    gpu::GpuSpan* span,
) {
    if (span != null && span.owner == state.allocations.owner) {
        return gpu::DebugResourceKind.ALLOCATION;
    }
    return gpu::DebugResourceKind.GPU_SPAN;
}

fn void? validate_span_queue_access(
    VkDeviceState* state,
    gpu::QueueRoles queue_roles,
    ResolvedSpan* resolved,
    gpu::GpuSpan* span,
    ZString operation,
) @private {
    gpu::DebugResourceKind resource_kind = span_resource_kind(state, span);
    if ((uint)resolved.access == 0
        || !gpu::internal::queue_roles_are_known(resolved.access)
        || !gpu::internal::queue_roles_are_subset(resolved.access, resolved.slot.access)) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       resource_kind,
            has_public_identity: true,
            index:               span.index,
            generation:          span.generation,
            debug_name:          slot_name(&resolved.slot.debug_name[0]),
            rejected_field:      "GpuSpan",
            invariant:           "span access must be a non-empty subset of its owner access",
            backend_text:        null,
        )~;
    }
    return validate_operation_queue_access(
        state:          state,
        queue_roles:    queue_roles,
        access:         resolved.access,
        resource_kind:  resource_kind,
        index:          span.index,
        generation:     span.generation,
        debug_name:     slot_name(&resolved.slot.debug_name[0]),
        operation:      operation,
        rejected_field: "GpuSpan",
    );
}

fn void? validate_span_recording_access_only(
    VkDeviceState* state,
    VkCommandRecord* record,
    ResolvedSpan* resolved,
    gpu::GpuSpan* span,
    ZString operation,
) @private {
    validate_span_queue_access(
        state:       state,
        queue_roles: record.queue.roles,
        resolved:    resolved,
        span:        span,
        operation:   operation,
    )!;
}

fn void? track_span_command_reference(
    VkDeviceState* state,
    VkCommandRecord* record,
    gpu::GpuSpan* span,
) @private {
    return track_command_reference(
        state,
        record,
        span.owner,
        span.index,
        span.generation,
    );
}

fn void? validate_span_transfer_usage(
    VkDeviceState* state,
    ResolvedSpan* resolved,
    gpu::GpuSpan* span,
    bool source,
    ZString operation,
) @private {
    bool supported = source
        ? resolved.slot.usage.transfer_src
        : resolved.slot.usage.transfer_dst;
    if (supported) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       span_resource_kind(state, span),
        has_public_identity: true,
        index:               span.index,
        generation:          span.generation,
        debug_name:          slot_name(&resolved.slot.debug_name[0]),
        rejected_field:      "GpuSpan",
        invariant:           source
            ? "source span must support copy reads"
            : "destination span must support copy writes",
        backend_text:        null,
    )~;
}

fn ResolvedSpan? resolve_span_slot(
    VkDeviceState* state,
    gpu::GpuSpan* span,
    usz required,
    ZString operation = null,
    ZString rejected_field = null,
) {
    ResolvedSpan? result = resolve_span(state, span);
    if (catch span_fault = result) {
        if (operation == null) return span_fault~;
        bool allocation_owner = span != null
            && span.owner == state.allocations.owner;
        bool has_identity = span_fault != gpu::INVALID_HANDLE;
        ZString debug_name;
        if (has_identity && span.owner == state.buffers.owner) {
            BufferHandle handle = {
                .owner      = span.owner,
                .index      = span.index,
                .generation = span.generation,
            };
            if (try slot = state.buffers.get(handle)) {
                debug_name = slot_name(&slot.debug_name[0]);
            }
        } else if (has_identity && allocation_owner) {
            gpu::GpuAllocation allocation = {
                .owner      = span.owner,
                .index      = span.index,
                .generation = span.generation,
            };
            if (try slot = state.allocations.get(allocation)) {
                debug_name = slot_name(&slot.backing.debug_name[0]);
            }
        }
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        span_fault,
            resource_kind:       span_resource_kind(state, span),
            has_public_identity: has_identity,
            index:               has_identity ? span.index : 0,
            generation:          has_identity ? span.generation : 0,
            debug_name:          debug_name,
            rejected_field:      span_fault == gpu::INVALID_HANDLE
                ? "GpuSpan.identity"
                : rejected_field,
            invariant:           span_fault == gpu::INVALID_HANDLE
                ? "must identify one live span owner"
                : "span range must fit its live owner",
            backend_text:        null,
        )~;
    }
    ResolvedSpan resolved = result;
    if (required > resolved.size) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

<* Bytes per index element for `index_type`. *>
fn usz? index_element_size(gpu::IndexType index_type) {
    switch (index_type) {
        case gpu::IndexType.U16: return 2;
        case gpu::IndexType.U32: return 4;
        default: return gpu::INVALID_ARGUMENT~;
    }
}

fn usz trusted_index_element_size(gpu::IndexType index_type) =>
    index_type == gpu::IndexType.U16 ? 2 : 4;

fn ResolvedSpan? lower_index_span(
    VkDeviceState* state,
    gpu::GpuSpan* span,
    gpu::IndexType index_type,
    usz required,
) {
    ResolvedSpan resolved = resolve_span_slot(state, span, required)!;
    if (resolved.offset % trusted_index_element_size(index_type) != 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return resolved;
}

<* Resolve an index span and validate bounds, usage, alignment, and queue access.
   `required` is zero when GPU-written arguments provide the index count. *>
fn ResolvedSpan? validate_and_lower_index_span(
    VkDeviceState* state,
    VkCommandRecord* record,
    gpu::GpuSpan* span,
    gpu::IndexType index_type,
    usz required,
    ZString operation,
) {
    ResolvedSpan resolved = resolve_span_slot(state, span, required)!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      span,
        operation: operation,
    )!;
    if (!resolved.slot.usage.index) return gpu::INVALID_ARGUMENT~;
    usz index_size = index_element_size(index_type)!;
    if (resolved.offset % index_size != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn bool pipeline_kind_matches(VkDeviceState* state, PipelineSlot* slot, PipelineKind kind) {
    return (PipelineKind)state.pipeline_cache.entries[slot.entry].key.kind == kind;
}

<* The handle's pipeline must be of `kind`, else the command targets the wrong bind point. *>
fn void? validate_pipeline_kind(
    VkDeviceState* state,
    gpu::PipelineHandle handle,
    PipelineKind kind,
    ZString operation,
) {
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if (!pipeline_kind_matches(state, slot, kind)) {
        ZString invariant = kind == PipelineKind.COMPUTE
            ? "pipeline kind must match the COMPUTE bind point"
            : "pipeline kind must match the GRAPHICS bind point";
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PIPELINE,
            has_public_identity: true,
            index:               handle.index,
            generation:          handle.generation,
            debug_name:          null,
            rejected_field:      "pipeline",
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
}
