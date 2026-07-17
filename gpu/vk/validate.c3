module gpu::vk @private;

import gpu @public;
import vk;

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
    BufferSlot* slot,
    gpu::GpuSpan* span,
    ZString operation,
) @private {
    PersistentSpanAllocationMatch allocation_match = persistent_span_allocation_match(
        state,
        span,
    );
    if (allocation_match == PersistentSpanAllocationMatch.UNRESOLVED
        || allocation_match == PersistentSpanAllocationMatch.METADATA_MISMATCH) {
        bool allocation_resolved = allocation_match
            == PersistentSpanAllocationMatch.METADATA_MISMATCH;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: allocation_resolved,
            index:               allocation_resolved ? span.allocation.index : 0,
            generation:          allocation_resolved ? span.allocation.generation : 0,
            debug_name:          null,
            rejected_field:      allocation_resolved ? "GpuSpan" : "GpuSpan.allocation",
            invariant:           "must identify one live allocation and preserve its immutable bounds and access",
            backend_text:        null,
        )~;
    }
    if ((uint)span.access == 0
        || !gpu::queue_roles_are_known(span.access)
        || !gpu::queue_roles_are_subset(span.access, slot.access)) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               span.buffer.index,
            generation:          span.buffer.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "GpuSpan.access",
            invariant:           "span access must be a non-empty subset of the backing buffer access",
            backend_text:        null,
        )~;
    }
    return validate_operation_queue_access(
        state:          state,
        queue_roles:    record.queue.roles,
        access:         span.access,
        resource_kind:  gpu::DebugResourceKind.BUFFER,
        index:          span.buffer.index,
        generation:     span.buffer.generation,
        debug_name:     slot_name(&slot.debug_name[0]),
        operation:      operation,
        rejected_field: "GpuSpan.access",
    );
}
<* Backing-buffer slot for `span`: the handle resolves, `required` bytes fit
   within the span itself, and the span's own offset/size fit the backing
   buffer. Usage and alignment checks stay with the caller. *>
fn BufferSlot*? resolve_span_slot(
    VkDeviceState* state,
    gpu::GpuSpan* span,
    usz required,
    ZString operation = null,
    ZString rejected_field = null,
) {
    BufferSlot* slot = state.buffers.get(span.buffer)!;
    if (required > span.size) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(slot.size, span.offset, span.size)) {
        if (operation == null) return gpu::INVALID_ARGUMENT~;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               span.buffer.index,
            generation:          span.buffer.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      rejected_field,
            invariant:           "span offset and size must fit the backing buffer",
            backend_text:        null,
        )~;
    }
    return slot;
}

<* Bytes per index element for `index_type`. *>
fn usz index_element_size(gpu::IndexType index_type) => index_type == gpu::IndexType.U16 ? 2 : 4;

<* Resolve an index span and validate bounds, usage, alignment, and queue access.
   `required` is zero when GPU-written arguments provide the index count. *>
fn BufferSlot*? resolve_index_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    gpu::IndexType index_type,
    usz required,
    ZString operation,
) {
    BufferSlot* slot = resolve_span_slot(state, span, required)!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        span:      span,
        operation: operation,
    )!;
    if (!slot.usage.index) return gpu::INVALID_ARGUMENT~;
    usz index_size = index_element_size(index_type);
    if (span.offset % index_size != 0) return gpu::INVALID_ARGUMENT~;
    return slot;
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
