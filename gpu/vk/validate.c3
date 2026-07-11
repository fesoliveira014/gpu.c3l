module gpu::vk;

import gpu;
import vk;

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

<* Index buffer usage, offset alignment to the index element size, and the
   backing slot lookup in one call; mirrors resolve_indirect_span. `required`
   is index_count * index_size when the caller knows the element count up
   front, else 0 (indirect draws read it from GPU-written arguments). *>
fn BufferSlot*? resolve_index_span(VkDeviceState* state, gpu::GpuSpan* span, gpu::IndexType index_type, usz required) {
    BufferSlot* slot = resolve_span_slot(state, span, required)!;
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
