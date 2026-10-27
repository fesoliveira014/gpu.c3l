module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

struct ResolvedSpan {
    BufferSlot*       slot;
    usz               offset;
    usz               size;
    gpu::MemoryClass   memory_class;
    void*             mapping;
    gpu::GpuAddress   address;
}

fn ResolvedSpan resolve_span(VkDeviceState* state, gpu::GpuSpan* span) {
    if (state == null || span == null) abort("gpu: corrupt GpuSpan");

    if (span.owner != state.allocations.owner) abort("gpu: foreign GpuSpan");
    gpu::GpuAllocation allocation = {
        .owner      = span.owner,
        .index      = span.index,
        .generation = span.generation,
    };
    AllocationSlot* allocation_slot = state.allocations.get(allocation);
    assert(!allocation_slot.texture_memory, "span must not reference a TEXTURE allocation");
    BufferSlot* slot = &allocation_slot.backing;
    usz offset = span.offset;

    assert(span.size > 0, "span.size must be greater than 0");
    assert(
        gpu::internal::range_fits(slot.size, offset, span.size),
        "span must lie within its allocation"
    );
    void* mapping;
    if (slot.cpu_base != null) {
        assert(
            (uptr)offset <= uptr::max - (uptr)slot.cpu_base,
            "mapped span pointer must not overflow"
        );
        mapping = (char*)slot.cpu_base + offset;
    }
    gpu::GpuAddress address;
    if ((ulong)slot.gpu_base != 0) {
        assert(
            (ulong)offset <= ulong::max - (ulong)slot.gpu_base,
            "span GPU address must not overflow"
        );
        address = (gpu::GpuAddress)((ulong)slot.gpu_base + (ulong)offset);
    }
    return {
        .slot           = slot,
        .offset         = offset,
        .size           = span.size,
        .memory_class   = slot.memory_class,
        .mapping        = mapping,
        .address        = address,
    };
}

fn char[] resolved_span_mapping(ResolvedSpan* span) {
    assert(span.mapping != null, "span's allocation must be host-mapped");
    return ((char*)span.mapping)[:span.size];
}

fn gpu::GpuAddress resolved_span_address(ResolvedSpan* span) {
    assert((ulong)span.address != 0, "span must have a GPU address");
    return span.address;
}

<* Resolve a span the caller has already checked. Public entry points require
   a nonempty span and, where the size follows from their arguments, that it
   covers the bytes they use. Spans whose required size comes from the device
   (acceleration-structure and scratch sizes, shader-binding-table strides) are
   checked here: `required` must not exceed `span.size`, and a violation
   panics in safe builds. *>
fn ResolvedSpan resolve_span_slot(
    VkDeviceState* state,
    gpu::GpuSpan* span,
    usz required,
) {
    ResolvedSpan resolved = resolve_span(state, span);
    assert(
        required <= resolved.size,
        "span.size must cover the bytes required by the call (%d required, %d given)",
        required,
        resolved.size
    );
    return resolved;
}

fn usz index_element_bytes(gpu::IndexType index_type) =>
    index_type == gpu::IndexType.U16 ? 2 : 4;


fn bool pipeline_kind_matches(VkDeviceState* state, PipelineSlot* slot, PipelineKind kind) {
    return (PipelineKind)state.pipeline_cache.entries[slot.entry].key.kind == kind;
}
