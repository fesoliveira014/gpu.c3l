module gpu::internal @private;

import gpu;

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
