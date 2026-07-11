module gpu::vk;

import gpu;
import vk;

<* Backing-buffer slot for `span`: the handle resolves, `required` bytes fit
   within the span itself, and the span's own offset/size fit the backing
   buffer. Usage and alignment checks stay with the caller. *>
fn BufferSlot*? resolve_span_slot(VkDeviceState* state, gpu::GpuSpan* span, usz required) {
    BufferSlot* slot = state.buffers.get(span.buffer)!;
    if (required > span.size) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(slot.size, span.offset, span.size)) return gpu::INVALID_ARGUMENT~;
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
fn void? validate_pipeline_kind(VkDeviceState* state, gpu::PipelineHandle handle, PipelineKind kind) {
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if (!pipeline_kind_matches(state, slot, kind)) return gpu::INVALID_ARGUMENT~;
}

<* The shader's compiled stage must match what the pipeline create builds it into. *>
fn void? validate_shader_stage(ShaderSlot* slot, gpu::ShaderStage expected) {
    if (slot.stage != expected) return gpu::INVALID_ARGUMENT~;
}
