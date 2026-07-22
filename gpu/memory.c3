module gpu;


<* Overflow-safe bounds check: offset + size must fit in limit. A zero size
   only checks the offset; each caller defines what zero means. *>
fn bool range_fits(usz limit, usz offset, usz size) @private {
    if (offset > limit) return false;
    return size <= limit - offset;
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
    if ((alignment & (alignment - 1)) != 0) return INVALID_ARGUMENT~;
    return max(alignment, MIN_ALLOCATION_ALIGNMENT);
}

fn void? validate_texture_allocation(
    DeviceOperation* operation,
    AllocationDesc* desc,
    usz* alignment,
    bool allow_dedicated = false,
) @private {
    if (desc.memory_class != MemoryClass.TEXTURE) {
        if (desc.texture_requirements.len != 0) return INVALID_ARGUMENT~;
        return;
    }
    if (desc.texture_requirements.len == 0) return INVALID_ARGUMENT~;

    uint compatibility = uint::max;
    foreach (&requirements : desc.texture_requirements) {
        if (requirements.size == 0 || requirements.alignment == 0) {
            return INVALID_ARGUMENT~;
        }
        compatibility &= texture_compatibility_value(requirements.compatibility);
        if ((requirements.alignment & (requirements.alignment - 1)) != 0
            || !requirements.compatibility.is_valid()) {
            return INVALID_ARGUMENT~;
        }
        validate_resource_owner(
            operation,
            texture_compatibility_owner(requirements.compatibility),
            ResourceOwnerKind.ALLOCATION,
        )!;
        if (requirements.dedicated_only && !allow_dedicated) {
            return UNSUPPORTED_FEATURE~;
        }
        if (desc.size < requirements.size) return INVALID_ARGUMENT~;
        *alignment = max(*alignment, requirements.alignment);
    }
    if (compatibility == 0) return INVALID_ARGUMENT~;
}
