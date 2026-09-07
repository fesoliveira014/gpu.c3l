module gpu::internal @private;

import gpu;
import gpu::internal::vk @public;

<* Overflow-safe bounds check: offset + size must fit in limit. A zero size
   only checks the offset; each caller defines what zero means. *>
fn bool range_fits(usz limit, usz offset, usz size) @private {
    if (offset > limit) return false;
    return size <= limit - offset;
}

fn bool acceleration_structure_storage_fits(
    AccelerationStructureRequirements* requirements,
    usz allocation_size,
    usz allocation_alignment,
    usz offset,
) @private {
    if (requirements == null
        || requirements.storage_size == 0
        || requirements.storage_alignment == 0
        || (requirements.storage_alignment
            & (requirements.storage_alignment - 1)) != 0) return false;
    return allocation_alignment >= requirements.storage_alignment
        && offset % requirements.storage_alignment == 0
        && range_fits(
            allocation_size,
            offset,
            requirements.storage_size,
        );
}

fn bool acceleration_structure_scratch_fits(
    AccelerationStructureRequirements* requirements,
    usz span_size,
    GpuAddress address,
    bool update,
) @private {
    if (requirements == null
        || requirements.scratch_alignment == 0
        || (requirements.scratch_alignment
            & (requirements.scratch_alignment - 1)) != 0
        || address == (GpuAddress)0) return false;
    usz required_size = update
        ? requirements.update_scratch_size
        : requirements.build_scratch_size;
    return required_size > 0
        && span_size >= required_size
        && (ulong)address % (ulong)requirements.scratch_alignment == 0;
}

fn bool memory_class_is_known(MemoryClass memory_class) @private {
    switch (memory_class) {
        case MemoryClass.CPU_WRITE:
        case MemoryClass.GPU_PRIVATE:
        case MemoryClass.CPU_READ:
        case MemoryClass.TEXTURE:
        case MemoryClass.CPU_WRITE_GPU_LOCAL:
            return true;
        default:
            return false;
    }
}

fn usz? normalized_allocation_alignment(usz alignment) @private {
    const usz MIN_ALLOCATION_ALIGNMENT = 16;
    if (alignment == 0) return MIN_ALLOCATION_ALIGNMENT;
    if ((alignment & (alignment - 1)) != 0) return gpu::INVALID_ARGUMENT~;
    return max(alignment, MIN_ALLOCATION_ALIGNMENT);
}

fn void? validate_texture_allocation(
    gpu::internal::DeviceOperation* operation,
    AllocationDesc* desc,
    usz* alignment,
    bool allow_dedicated = false,
) @private {
    if (desc.memory_class != MemoryClass.TEXTURE) {
        if (desc.texture_requirements.len != 0) return gpu::INVALID_ARGUMENT~;
        return;
    }
    if (desc.texture_requirements.len == 0) return gpu::INVALID_ARGUMENT~;

    uint compatibility = uint::max;
    foreach (&requirements : desc.texture_requirements) {
        if (requirements.size == 0 || requirements.alignment == 0) {
            return gpu::INVALID_ARGUMENT~;
        }
        compatibility &= texture_compatibility_value(requirements.compatibility);
        if ((requirements.alignment & (requirements.alignment - 1)) != 0
            || !requirements.compatibility.is_valid()) {
            return gpu::INVALID_ARGUMENT~;
        }
        gpu::internal::validate_resource_owner(
            operation,
            texture_compatibility_owner(requirements.compatibility),
            gpu::internal::ResourceOwnerKind.ALLOCATION,
        )!;
        if (requirements.dedicated_only && !allow_dedicated) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        if (desc.size < requirements.size) return gpu::INVALID_ARGUMENT~;
        *alignment = max(*alignment, requirements.alignment);
    }
    if (compatibility == 0) return gpu::INVALID_ARGUMENT~;
}

<* Validation shared by `allocate_memory` and `allocate_mapped_allocation`.

   @return "A normalized descriptor, or a recoverable invalid-argument, unsupported-feature, or ownership fault." *>
fn AllocationDesc? validated_allocation_desc(
    DeviceOperation* operation,
    AllocationDesc* desc,
) @private {
    if (desc == null || desc.size == 0) return gpu::INVALID_ARGUMENT~;
    if (!memory_class_is_known(desc.memory_class)) return gpu::INVALID_ARGUMENT~;
    validate_queue_access(operation, desc.access)!;
    AllocationDesc normalized = *desc;
    normalized.alignment = normalized_allocation_alignment(desc.alignment)!;
    validate_texture_allocation(operation, desc, &normalized.alignment)!;
    return normalized;
}

<* Compose the span, mapping, and address for a live allocation using an
   already-pinned device operation; acquires no additional pin. *>
fn MappedGpuSpan? pinned_allocation_view(
    gpu::internal::vk::VkDeviceState* state,
    GpuAllocation allocation,
) @private {
    GpuSpan span = gpu::internal::vk::vk_get_allocation_span(state, allocation)!;
    char[] bytes = gpu::internal::vk::vk_get_span_mapping(state, span)!;
    if (bytes.len != span.size) return gpu::BACKEND_ERROR~;
    GpuAddress address = gpu::internal::vk::vk_get_span_address(state, span)!;
    return {
        .span    = span,
        .bytes   = bytes,
        .address = address,
    };
}

<* Free an allocation whose mapped view could not be built, inside the
   caller's still-held device operation pin. A free fault leaves the
   allocation live with no owner: it is reported through the debug callback,
   then treated as unrecoverable. *>
fn void discard_failed_mapped_allocation(
    DeviceOperation* operation,
    GpuAllocation allocation,
    ZString debug_name,
) @private {
    if (catch gpu::internal::vk::vk_free_allocation(operation.state, allocation)) {
        gpu::internal::vk::vk_report_abandoned_allocation(operation.state, allocation, debug_name);
        unreachable(
            "gpu: failed mapped allocation rollback could not free allocation %d (generation %d)",
            allocation.index,
            allocation.generation,
        );
    }
}

<* Allocate a mapped class and borrow its complete span, mapping, and address
   under one device operation pin, so the create is transactional.

   @return "Consistent borrowed views of the new allocation, or a recoverable invalid-argument, unsupported-feature, allocation, busy, device-loss, or backend fault." *>
fn MappedGpuSpan? allocate_mapped_allocation(
    Device* device,
    AllocationDesc* desc,
) @private {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    AllocationDesc normalized = validated_allocation_desc(&operation, desc)!;
    GpuAllocation allocation =
        gpu::internal::vk::vk_allocate_memory(operation.state, &normalized)!;
    MappedGpuSpan? mapped = pinned_allocation_view(operation.state, allocation);
    if (catch err = mapped) {
        discard_failed_mapped_allocation(&operation, allocation, desc.debug_name);
        return err~;
    }
    retain_device_children(&operation);
    return mapped;
}
