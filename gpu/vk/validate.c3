module gpu::vk @private;

import gpu @public;
import vk;

struct ResolvedSpan {
    BufferSlot*       slot;
    gpu::BufferHandle backing_buffer;
    usz               offset;
    usz               size;
    gpu::QueueRoles   access;
    gpu::MemoryKind   memory_kind;
    void*             mapping;
    gpu::GpuAddress   address;
}

fn ResolvedSpan? resolve_span(VkDeviceState* state, gpu::GpuSpan* span) {
    if (state == null || span == null) return gpu::INVALID_HANDLE~;
    if (span.size == 0) return gpu::INVALID_ARGUMENT~;

    BufferSlot* slot;
    gpu::BufferHandle backing_buffer;
    usz offset;
    gpu::QueueRoles access;
    gpu::MemoryKind memory_kind;
    if (span.owner == state.buffers.owner) {
        backing_buffer = {
            .owner      = span.owner,
            .index      = span.index,
            .generation = span.generation,
        };
        slot = state.buffers.get(backing_buffer)!;
        offset = span.offset;
        access = slot.access;
        memory_kind = slot.memory_kind;
    } else if (span.owner == state.persistent_arena.allocations.owner) {
        PersistentAllocationHandle allocation_handle = {
            .owner      = span.owner,
            .index      = span.index,
            .generation = span.generation,
        };
        PersistentAllocationState* allocation =
            state.persistent_arena.allocations.get(allocation_handle)!;
        if (!gpu::range_fits(allocation.size, span.offset, span.size)) {
            return gpu::INVALID_ARGUMENT~;
        }
        if (span.offset > usz::max - allocation.offset) {
            return gpu::INVALID_ARGUMENT~;
        }
        offset = allocation.offset + span.offset;
        backing_buffer = state.persistent_arena.backing_buffer;
        slot = state.buffers.get(backing_buffer)!;
        access = allocation.access;
        memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD;
    } else {
        return gpu::INVALID_HANDLE~;
    }

    if (!gpu::range_fits(slot.size, offset, span.size)) {
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
        .memory_kind    = memory_kind,
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
    if (gpu::queue_access_intersects(access, queue_roles)) return;
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

fn void? validate_buffer_queue_access(
    VkDeviceState* state,
    gpu::QueueRoles queue_roles,
    BufferSlot* slot,
    gpu::BufferHandle handle,
    ZString operation,
) @private {
    return validate_operation_queue_access(
        state:          state,
        queue_roles:    queue_roles,
        access:         slot.access,
        resource_kind:  gpu::DebugResourceKind.BUFFER,
        index:          handle.index,
        generation:     handle.generation,
        debug_name:     slot_name(&slot.debug_name[0]),
        operation:      operation,
        rejected_field: "BufferDesc.access",
    );
}

fn void? validate_buffer_recording_access(
    VkDeviceState* state,
    CommandRecord* record,
    BufferSlot* slot,
    gpu::BufferHandle handle,
    ZString operation,
) @private {
    return validate_buffer_queue_access(
        state:     state,
        queue_roles: record.queue.roles,
        slot:      slot,
        handle:    handle,
        operation: operation,
    );
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

fn void? validate_texture_recording_access(
    VkDeviceState* state,
    CommandRecord* record,
    TextureSlot* slot,
    gpu::TextureHandle handle,
    ZString operation,
) @private {
    return validate_texture_queue_access(
        state:     state,
        queue_roles: record.queue.roles,
        slot:      slot,
        handle:    handle,
        operation: operation,
    );
}

fn void? validate_span_recording_access(
    VkDeviceState* state,
    CommandRecord* record,
    ResolvedSpan* resolved,
    gpu::GpuSpan* span,
    ZString operation,
) @private {
    gpu::DebugResourceKind resource_kind = span.owner
        == state.persistent_arena.allocations.owner
        ? gpu::DebugResourceKind.PERSISTENT_SPAN
        : gpu::DebugResourceKind.BUFFER;
    if ((uint)resolved.access == 0
        || !gpu::queue_roles_are_known(resolved.access)
        || !gpu::queue_roles_are_subset(resolved.access, resolved.slot.access)) {
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
            invariant:           "resolved access must be a non-empty subset of the backing buffer access",
            backend_text:        null,
        )~;
    }
    return validate_operation_queue_access(
        state:          state,
        queue_roles:    record.queue.roles,
        access:         resolved.access,
        resource_kind:  resource_kind,
        index:          span.index,
        generation:     span.generation,
        debug_name:     slot_name(&resolved.slot.debug_name[0]),
        operation:      operation,
        rejected_field: "GpuSpan",
    );
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
        bool persistent = span != null
            && span.owner == state.persistent_arena.allocations.owner;
        bool has_identity = span_fault != gpu::INVALID_HANDLE;
        ZString debug_name;
        if (has_identity && span.owner == state.buffers.owner) {
            gpu::BufferHandle handle = {
                .owner      = span.owner,
                .index      = span.index,
                .generation = span.generation,
            };
            if (try slot = state.buffers.get(handle)) {
                debug_name = slot_name(&slot.debug_name[0]);
            }
        }
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        span_fault,
            resource_kind:       persistent
                ? gpu::DebugResourceKind.PERSISTENT_SPAN
                : gpu::DebugResourceKind.BUFFER,
            has_public_identity: has_identity,
            index:               has_identity ? span.index : 0,
            generation:          has_identity ? span.generation : 0,
            debug_name:          debug_name,
            rejected_field:      span_fault == gpu::INVALID_HANDLE
                ? "GpuSpan.identity"
                : rejected_field,
            invariant:           span_fault == gpu::INVALID_HANDLE
                ? "must identify one live span owner"
                : persistent
                    ? "span range must fit its live owner"
                    : "span offset and size must fit the backing buffer",
            backend_text:        null,
        )~;
    }
    ResolvedSpan resolved = result;
    if (required > resolved.size) return gpu::INVALID_ARGUMENT~;
    return resolved;
}
<* Bytes per index element for `index_type`. *>
fn usz index_element_size(gpu::IndexType index_type) => index_type == gpu::IndexType.U16 ? 2 : 4;

<* Resolve an index span and validate bounds, usage, alignment, and queue access.
   `required` is zero when GPU-written arguments provide the index count. *>
fn ResolvedSpan? resolve_index_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    gpu::IndexType index_type,
    usz required,
    ZString operation,
) {
    ResolvedSpan resolved = resolve_span_slot(state, span, required)!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      span,
        operation: operation,
    )!;
    if (!resolved.slot.usage.index) return gpu::INVALID_ARGUMENT~;
    usz index_size = index_element_size(index_type);
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

<* The shader's compiled stage must match its pipeline role. *>
fn void? validate_shader_stage(
    VkDeviceState* state,
    ShaderSlot* slot,
    gpu::ShaderStage expected,
    ZString operation,
    ZString rejected_field,
    gpu::ShaderHandle handle,
) {
    if (slot.stage == expected) return;
    ZString invariant = expected == gpu::ShaderStage.COMPUTE
        ? "shader stage must be COMPUTE"
        : expected == gpu::ShaderStage.VERTEX
            ? "shader stage must be VERTEX"
            : "shader stage must be FRAGMENT";
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.SHADER,
        has_public_identity: true,
        index:               handle.index,
        generation:          handle.generation,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        null,
    )~;
}
