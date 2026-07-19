module gpu;

// A shader-visible 64-bit GPU address. Distinct from a plain ulong so it can't
// be confused with sizes, offsets, or handles.
typedef GpuAddress = ulong;

enum MemoryClass : int {
    CPU_WRITE,
    GPU_PRIVATE,
    CPU_READ,
    TEXTURE,
}

<* Non-owning identity and range into generic GPU data. *>
struct GpuSpan {
    ulong owner;
    uint  index;
    uint  generation;
    usz   offset;
    usz   size;
}

<* `size` must be nonzero; alignment zero selects 16 bytes. *>
struct AllocationDesc {
    usz         size;
    usz         alignment;
    MemoryClass memory_class;
    QueueRoles  access;
    TextureRequirements[] texture_requirements;
    ZString     debug_name;
}

<* Immutable allocation properties and actual mapping behavior. *>
struct AllocationInfo {
    usz         size;
    usz         alignment;
    MemoryClass memory_class;
    QueueRoles  access;
    bool        mapped;
    bool        coherent;
    bool        addressable;
}

const usz MAX_MEMORY_HEAPS = 16;

// Per-heap memory usage snapshot.
struct MemoryHeapBudget {
    ulong usage;
    ulong budget;
    ulong allocation_bytes;
    ulong block_bytes;
}

// Device memory snapshot. `heaps[0..heap_count]` are valid.
struct MemoryStats {
    MemoryHeapBudget[MAX_MEMORY_HEAPS] heaps;
    uint  heap_count;
    ulong texture_count;
    ulong live_allocation_count;
}

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

<* Allocate caller-managed GPU storage. Free with `free_allocation`. *>
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null || desc.size == 0) return INVALID_ARGUMENT~;
    if (!memory_class_is_known(desc.memory_class)) return INVALID_ARGUMENT~;
    validate_queue_access(&operation, desc.access)!;

    AllocationDesc normalized = *desc;
    normalized.alignment = normalized_allocation_alignment(desc.alignment)!;
    validate_texture_allocation(&operation, desc, &normalized.alignment)!;
    GpuAllocation allocation = operation.vtable.allocate_memory(
        device,
        &normalized,
    )!;
    retain_device_children(&operation);
    return allocation;
}

<* Free quiescent storage; faults preserve the owning token. *>
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (allocation == null || !allocation.is_valid()) return INVALID_HANDLE~;
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    operation.vtable.free_allocation(device, *allocation)!;
    release_device_children(&operation);
    *allocation = GPU_ALLOCATION_INVALID;
}

<* Return immutable properties for a live allocation. *>
fn AllocationInfo? get_allocation_info(
    Device* device,
    GpuAllocation allocation,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_info(device, allocation);
}

<* Borrow the allocation's complete range. *>
fn GpuSpan? get_allocation_span(Device* device, GpuAllocation allocation) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        allocation.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_span(device, allocation);
}

<* Borrow the CPU mapping for a live host-visible span. *>
fn char[]? get_span_mapping(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_span_mapping(device, span);
}

<* Return the stable nonzero address of a live span. *>
fn GpuAddress? get_span_address(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_span_address(device, span);
}

<* Make CPU writes to a live mapped allocation range visible to the GPU.
   Coherent ranges are no-ops. *>
fn void? flush_mapped_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        span.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.flush_mapped_span(device, span);
}

<* After the relevant GPU completion point, make its writes visible to CPU
   reads. This call does not wait; coherent ranges are no-ops. *>
fn void? invalidate_mapped_span(Device* device, GpuSpan span) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(
        &operation,
        span.owner,
        ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.invalidate_mapped_span(device, span);
}

<* Derive a sub-range without bounds or overflow checks. Use only when the
   caller has already proved the range, or deliberately needs an invalid span
   for validation testing. *>
fn GpuSpan GpuSpan.unchecked_subspan(self, usz offset, usz size) {
    return {
        .owner      = self.owner,
        .index      = self.index,
        .generation = self.generation,
        .offset     = self.offset + offset,
        .size       = size,
    };
}

<* Derive a non-empty exact-sized range within this immediate parent. *>
fn GpuSpan? GpuSpan.checked_subspan(self, usz offset, usz size) {
    if (size == 0 || !range_fits(self.size, offset, size)) {
        return INVALID_ARGUMENT~;
    }
    if (offset > usz::max - self.offset) return INVALID_ARGUMENT~;
    return self.unchecked_subspan(offset, size);
}
