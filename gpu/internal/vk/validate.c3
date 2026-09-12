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

fn ResolvedSpan? resolve_span(VkDeviceState* state, gpu::GpuSpan* span) {
    if (state == null || span == null) return gpu::INVALID_HANDLE~;

    if (span.owner != state.allocations.owner) return gpu::INVALID_HANDLE~;
    gpu::GpuAllocation allocation = {
        .owner      = span.owner,
        .index      = span.index,
        .generation = span.generation,
    };
    AllocationSlot* allocation_slot = state.allocations.get(allocation)!;
    if (allocation_slot.texture_memory) return gpu::INVALID_ARGUMENT~;
    BufferSlot* slot = &allocation_slot.backing;
    usz offset = span.offset;

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
        .offset         = offset,
        .size           = span.size,
        .memory_class   = slot.memory_class,
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

fn gpu::DebugResourceKind span_resource_kind(
    VkDeviceState* state,
    gpu::GpuSpan* span,
) {
    if (span != null && span.owner == state.allocations.owner) {
        return gpu::DebugResourceKind.ALLOCATION;
    }
    return gpu::DebugResourceKind.GPU_SPAN;
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
        if (has_identity && allocation_owner) {
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

fn usz index_element_bytes(gpu::IndexType index_type) =>
    index_type == gpu::IndexType.U16 ? 2 : 4;


fn bool pipeline_kind_matches(VkDeviceState* state, PipelineSlot* slot, PipelineKind kind) {
    return (PipelineKind)state.pipeline_cache.entries[slot.entry].key.kind == kind;
}
