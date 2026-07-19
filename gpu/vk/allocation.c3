module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::core::mem;
import std::atomic::types;

alias AllocationCreateBufferFn = fn vk::Result (
    void*, vma::Allocator, vk::BufferCreateInfo*,
    vma::AllocationCreateInfo*, vk::DeviceSize,
    vk::Buffer*, vma::Allocation*, vma::AllocationInfo*
);
alias AllocationGetRequirementsFn = fn void (
    void*, vk::Device, vk::Buffer, vk::MemoryRequirements*
);
alias AllocationGetMemoryPropertiesFn = fn void (
    void*, vma::Allocator, vma::Allocation, vk::MemoryPropertyFlags*
);
alias AllocationGetAddressFn = fn vk::DeviceAddress (
    void*, vk::Device, vk::BufferDeviceAddressInfo*
);
alias AllocationDestroyBufferFn = fn void (
    void*, vma::Allocator, vk::Buffer, vma::Allocation
);
alias AllocationAllocateMemoryFn = fn vk::Result (
    void*, vma::Allocator, vk::MemoryRequirements*,
    vma::AllocationCreateInfo*, vma::Allocation*, vma::AllocationInfo*
);
alias AllocationFreeMemoryFn = fn void (
    void*, vma::Allocator, vma::Allocation
);

alias AllocationVisibilityFn = fn vk::Result (
    void*, vma::Allocator, vma::Allocation, vk::DeviceSize, vk::DeviceSize
);

const uint ALLOCATION_CAPACITY = 4096;
const uint ALLOCATION_FREE_NONE = 0;

struct AllocationSlot {
    vma::Allocation     raw_allocation;
    vma::AllocationInfo raw_info;
    usz                 size;
    gpu::QueueRoles     access;
    uint                memory_type_bits;
    bool                texture_memory;
    bool                dedicated_texture;
    BufferSlot          backing;
    usz                 alignment;
    gpu::MemoryClass    memory_class;
    bool                coherent;
    Atomic{uint}       validation_references;
    uint                live_placements;
    char[DEBUG_NAME_CAP] debug_name;
}

struct AllocationCell {
    AllocationSlot value;
    uint           generation;
    uint           next_free;
    bool           used;
}

struct AllocationTable {
    AllocationCell[] slots;
    ulong            owner;
    uint             free_head;
    uint             next_new;
}

struct AllocationNativeOps {
    void*                           user_data;
    AllocationCreateBufferFn        create_buffer;
    AllocationGetRequirementsFn     get_memory_requirements;
    AllocationAllocateMemoryFn      allocate_memory;
    AllocationFreeMemoryFn          free_memory;
    AllocationGetMemoryPropertiesFn get_memory_properties;
    AllocationGetAddressFn          get_buffer_device_address;
    AllocationDestroyBufferFn       destroy_buffer;
    AllocationVisibilityFn          flush_allocation;
    AllocationVisibilityFn          invalidate_allocation;
}

struct NonCoherentRange {
    vk::DeviceSize offset;
    vk::DeviceSize size;
}

fn gpu::GpuAllocation? AllocationTable.alloc(&self, AllocationSlot value) {
    uint index;
    bool found;
    while (self.free_head != ALLOCATION_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = ALLOCATION_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }
    self.slots[index].value = value;
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = self.slots[index].generation,
    };
}

fn AllocationSlot*? AllocationTable.get(&self, gpu::GpuAllocation allocation) {
    if (allocation.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint index = allocation.index;
    if (index >= self.slots.len || !self.slots[index].used) {
        return gpu::INVALID_HANDLE~;
    }
    if (self.slots[index].generation != allocation.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &self.slots[index].value;
}

fn void? AllocationTable.free(&self, gpu::GpuAllocation allocation) {
    AllocationSlot* slot = self.get(allocation)!;
    if (slot.live_placements > 0) return gpu::RESOURCE_IN_USE~;
    uint index = allocation.index;
    self.slots[index].used = false;
    if (!should_recycle_handle_generation(self.slots[index].generation)) {
        self.slots[index].generation = HANDLE_GENERATION_MAX;
        self.slots[index].next_free = ALLOCATION_FREE_NONE;
        return;
    }
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].next_free = self.free_head;
    self.free_head = index + 1;
}

fn vk::Result allocation_create_buffer_real(
    void* user_data,
    vma::Allocator allocator,
    vk::BufferCreateInfo* buffer_info,
    vma::AllocationCreateInfo* allocation_info,
    vk::DeviceSize min_alignment,
    vk::Buffer* out_buffer,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)user_data;
    return allocator.create_buffer_with_alignment(
        bi:            buffer_info,
        ci:            allocation_info,
        min_alignment: min_alignment,
        out_buffer:    out_buffer,
        out_alloc:     out_allocation,
        out_info:      out_info,
    );
}

fn void allocation_get_requirements_real(
    void* user_data,
    vk::Device device,
    vk::Buffer buffer,
    vk::MemoryRequirements* out_requirements,
) {
    (void)user_data;
    vk::get_buffer_memory_requirements(device, buffer, out_requirements);
}

fn void allocation_get_properties_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::MemoryPropertyFlags* out_properties,
) {
    (void)user_data;
    allocator.get_allocation_memory_properties(allocation, out_properties);
}

fn vk::DeviceAddress allocation_get_address_real(
    void* user_data,
    vk::Device device,
    vk::BufferDeviceAddressInfo* info,
) {
    (void)user_data;
    return vk::get_buffer_device_address(device, info);
}

fn void allocation_destroy_buffer_real(
    void* user_data,
    vma::Allocator allocator,
    vk::Buffer buffer,
    vma::Allocation allocation,
) {
    (void)user_data;
    allocator.destroy_buffer(buffer, allocation);
}

fn vk::Result allocation_flush_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::DeviceSize size,
) {
    (void)user_data;
    return allocator.flush_allocation(allocation, offset, size);
}

fn vk::Result allocation_invalidate_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::DeviceSize size,
) {
    (void)user_data;
    return allocator.invalidate_allocation(allocation, offset, size);
}

fn vk::Result allocation_allocate_memory_real(
    void* user_data,
    vma::Allocator allocator,
    vk::MemoryRequirements* requirements,
    vma::AllocationCreateInfo* allocation_info,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)user_data;
    return allocator.allocate_memory(
        requirements,
        allocation_info,
        out_allocation,
        out_info,
    );
}

fn void allocation_free_memory_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
) {
    (void)user_data;
    allocator.free_memory(allocation);
}

fn AllocationNativeOps allocation_native_ops_real() {
    return {
        .create_buffer             = &allocation_create_buffer_real,
        .get_memory_requirements   = &allocation_get_requirements_real,
        .allocate_memory           = &allocation_allocate_memory_real,
        .free_memory               = &allocation_free_memory_real,
        .get_memory_properties     = &allocation_get_properties_real,
        .get_buffer_device_address = &allocation_get_address_real,
        .destroy_buffer            = &allocation_destroy_buffer_real,
        .flush_allocation          = &allocation_flush_real,
        .invalidate_allocation     = &allocation_invalidate_real,
    };
}

fn BufferUsage allocation_buffer_usage() {
    return {
        .transfer_src,
        .transfer_dst,
        .uniform,
        .storage,
        .index,
        .vertex,
        .indirect,
        .addressable,
    };
}

fn gpu::MemoryKind? allocation_memory_kind(gpu::MemoryClass memory_class) {
    switch (memory_class) {
        case gpu::MemoryClass.CPU_WRITE:   return gpu::MemoryKind.PERSISTENT_UPLOAD;
        case gpu::MemoryClass.GPU_PRIVATE: return gpu::MemoryKind.DEVICE;
        case gpu::MemoryClass.CPU_READ:    return gpu::MemoryKind.READBACK;
        default:                           return gpu::INVALID_ARGUMENT~;
    }
}

fn vma::AllocationCreateInfo? memory_class_to_vma(
    gpu::MemoryClass memory_class,
) {
    vk::MemoryPropertyFlags host_visible =
        (vk::MemoryPropertyFlags)vk::MemoryPropertyFlagBits.HOST_VISIBLE_BIT;
    switch (memory_class) {
        case gpu::MemoryClass.CPU_WRITE:
            return {
                .usage          = vma::MemoryUsage.AUTO_PREFER_HOST,
                .flags          = { .mapped, .host_access_sequential_write },
                .required_flags = host_visible,
            };
        case gpu::MemoryClass.GPU_PRIVATE:
            return { .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE };
        case gpu::MemoryClass.CPU_READ:
            return {
                .usage          = vma::MemoryUsage.AUTO_PREFER_HOST,
                .flags          = { .mapped, .host_access_random },
                .required_flags = host_visible,
            };
        default:
            return gpu::INVALID_ARGUMENT~;
    }
}

fn vk::BufferCreateInfo? allocation_buffer_create_info(
    gpu::AllocationDesc* desc,
    uint[] queue_families,
) {
    BufferDesc backing = {
        .size        = desc.size,
        .usage       = allocation_buffer_usage(),
        .memory_kind = allocation_memory_kind(desc.memory_class)!,
        .access      = desc.access,
    };
    return make_buffer_create_info(&backing, queue_families);
}

fn uint? texture_memory_type_bits(gpu::AllocationDesc* desc) {
    uint memory_type_bits = uint::max;
    foreach (&requirements : desc.texture_requirements) {
        memory_type_bits &= texture_compatibility_value(requirements.compatibility);
    }
    if (memory_type_bits == 0) return gpu::INVALID_ARGUMENT~;
    return memory_type_bits;
}

fn gpu::GpuAllocation? allocate_texture_memory_for_state_with_ops(
    VkDeviceState* state,
    gpu::AllocationDesc* desc,
    AllocationNativeOps* ops,
) {
    uint memory_type_bits = texture_memory_type_bits(desc)!;
    vk::MemoryRequirements requirements = {
        .size             = (vk::DeviceSize)desc.size,
        .alignment        = (vk::DeviceSize)desc.alignment,
        .memory_type_bits = memory_type_bits,
    };
    vma::AllocationCreateInfo allocation_info = {
        .flags            = { .can_alias },
        .usage            = vma::MemoryUsage.GPU_ONLY,
        .memory_type_bits = memory_type_bits,
    };
    vma::Allocation allocation;
    vma::AllocationInfo native_info;
    bool owns_allocation;
    defer catch {
        if (owns_allocation) {
            ops.free_memory(ops.user_data, state.allocator, allocation);
        }
    }

    check_device_result(state, ops.allocate_memory(
        ops.user_data,
        state.allocator,
        &requirements,
        &allocation_info,
        &allocation,
        &native_info,
    ))!;
    owns_allocation = true;

    if (native_info.size < (vk::DeviceSize)desc.size
        || native_info.memory_type >= 32
        || native_info.offset % (vk::DeviceSize)desc.alignment != 0) {
        return gpu::BACKEND_ERROR~;
    }
    uint selected_memory_type = 1 << native_info.memory_type;
    if ((selected_memory_type & memory_type_bits) == 0) {
        return gpu::BACKEND_ERROR~;
    }

    char[DEBUG_NAME_CAP] debug_name;
    copy_debug_name(&debug_name[0], desc.debug_name);
    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(allocation, desc.debug_name);
    }
    AllocationSlot slot = {
        .raw_allocation   = allocation,
        .raw_info         = native_info,
        .size             = desc.size,
        .access           = desc.access,
        .memory_type_bits = memory_type_bits,
        .texture_memory   = true,
        .alignment        = desc.alignment,
        .debug_name       = debug_name,
        .memory_class     = gpu::MemoryClass.TEXTURE,
    };

    state.resource_mutex.lock()!!;
    gpu::GpuAllocation? published = state.allocations.alloc(slot);
    state.resource_mutex.unlock();
    gpu::GpuAllocation result = published!;
    owns_allocation = false;
    return result;
}

fn gpu::GpuAllocation? allocate_memory_for_state_with_ops(
    VkDeviceState* state,
    gpu::AllocationDesc* desc,
    AllocationNativeOps* ops,
) {
    if (desc.memory_class == gpu::MemoryClass.TEXTURE) {
        return allocate_texture_memory_for_state_with_ops(state, desc, ops);
    }
    uint[MAX_QUEUE_FAMILIES] families;
    uint family_count = resource_queue_families(
        state,
        desc.access,
        families[..],
    );
    vk::BufferCreateInfo buffer_info = allocation_buffer_create_info(
        desc,
        families[:family_count],
    )!;
    vma::AllocationCreateInfo allocation_info =
        memory_class_to_vma(desc.memory_class)!;

    vk::Buffer buffer;
    vma::Allocation allocation;
    vma::AllocationInfo native_info;
    bool owns_pair = false;
    defer catch {
        if (owns_pair) ops.destroy_buffer(
            ops.user_data,
            state.allocator,
            buffer,
            allocation,
        );
    }

    check_device_result(state, ops.create_buffer(
        ops.user_data,
        state.allocator,
        &buffer_info,
        &allocation_info,
        (vk::DeviceSize)desc.alignment,
        &buffer,
        &allocation,
        &native_info,
    ))!;
    owns_pair = true;

    bool requires_mapping = desc.memory_class != gpu::MemoryClass.GPU_PRIVATE;
    if (requires_mapping && native_info.mapped_data == null) {
        return gpu::BACKEND_ERROR~;
    }

    vk::MemoryRequirements requirements;
    ops.get_memory_requirements(
        ops.user_data,
        state.device,
        buffer,
        &requirements,
    );
    usz actual_alignment = max(
        (usz)requirements.alignment,
        desc.alignment,
    );
    if (native_info.offset % (vk::DeviceSize)actual_alignment != 0) {
        return gpu::BACKEND_ERROR~;
    }

    vk::MemoryPropertyFlags memory_properties;
    ops.get_memory_properties(
        ops.user_data,
        state.allocator,
        allocation,
        &memory_properties,
    );

    char[DEBUG_NAME_CAP] debug_name;
    copy_debug_name(&debug_name[0], desc.debug_name);
    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(allocation, desc.debug_name);
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.BUFFER,
            object_handle: (ulong)buffer,
            name:          desc.debug_name,
        );
    }

    vk::BufferDeviceAddressInfo address_info =
        vk::buffer_device_address_info().set_buffer(buffer);
    vk::DeviceAddress address = ops.get_buffer_device_address(
        ops.user_data,
        state.device,
        &address_info,
    );
    if (address == 0) return gpu::BACKEND_ERROR~;

    AllocationSlot slot = {
        .backing = {
            .buffer          = buffer,
            .allocation      = allocation,
            .allocation_info = native_info,
            .gpu_base        = (gpu::GpuAddress)address,
            .cpu_base        = requires_mapping ? native_info.mapped_data : null,
            .size            = desc.size,
            .usage           = allocation_buffer_usage(),
            .access          = desc.access,
            .memory_kind     = allocation_memory_kind(desc.memory_class)!,
            .debug_name      = debug_name,
            .internal        = true,
        },
        .size            = desc.size,
        .access          = desc.access,
        .alignment       = actual_alignment,
        .memory_class    = desc.memory_class,
        .coherent        = requires_mapping
            && native_info.mapped_data != null
            && ((uint)memory_properties
                & (uint)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT) != 0,
        .debug_name      = debug_name,
        .live_placements = 0,
    };

    state.resource_mutex.lock()!!;
    gpu::GpuAllocation? published = state.allocations.alloc(slot);
    state.resource_mutex.unlock();
    gpu::GpuAllocation result = published!;
    owns_pair = false;
    return result;
}

fn gpu::AllocationInfo? allocation_info_for_state(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
) {
    AllocationSlot* slot = state.allocations.get(allocation)!;
    return {
        .size         = slot.size,
        .alignment    = slot.alignment,
        .memory_class = slot.memory_class,
        .access       = slot.access,
        .mapped       = !slot.texture_memory && slot.backing.cpu_base != null,
        .coherent     = !slot.texture_memory && slot.coherent,
        .addressable  = !slot.texture_memory
            && (ulong)slot.backing.gpu_base != 0,
    };
}

fn gpu::GpuSpan? allocation_span_for_state(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
) {
    AllocationSlot* slot = state.allocations.get(allocation)!;
    if (slot.texture_memory) return gpu::INVALID_ARGUMENT~;
    return {
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
        .size       = slot.size,
    };
}

fn void? free_allocation_for_state_with_ops(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
    AllocationNativeOps* ops,
) {
    AllocationSlot released;
    {
        state.resource_mutex.lock()!;
        defer state.resource_mutex.unlock();
        AllocationSlot* slot = state.allocations.get(allocation)!;
        if (slot.live_placements > 0) return gpu::RESOURCE_IN_USE~;
        if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
            drain_completed_submitted_commands(state)!;
            if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
                return gpu::RESOURCE_IN_USE~;
            }
        }
        released = *slot;
        state.allocations.free(allocation)!;
    }
    if (released.texture_memory) {
        ops.free_memory(
            ops.user_data,
            state.allocator,
            released.raw_allocation,
        );
    } else {
        ops.destroy_buffer(
            ops.user_data,
            state.allocator,
            released.backing.buffer,
            released.backing.allocation,
        );
    }
}

fn NonCoherentRange? normalize_mapped_memory_range(
    usz offset,
    usz size,
    usz allocation_size,
    usz atom_size,
) {
    if (size == 0 || !gpu::range_fits(allocation_size, offset, size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!is_pow2(atom_size)) return gpu::BACKEND_ERROR~;

    usz mask = atom_size - 1;
    usz start = offset & ~mask;
    usz end = offset + size;
    usz remainder = end & mask;
    if (remainder != 0) {
        usz padding = atom_size - remainder;
        usz remaining = allocation_size - end;
        end = padding <= remaining ? end + padding : allocation_size;
    }
    return {
        .offset = (vk::DeviceSize)start,
        .size   = (vk::DeviceSize)(end - start),
    };
}

fn AllocationSlot*? resolve_mapped_allocation_span(
    VkDeviceState* state,
    gpu::GpuSpan* span,
) {
    if (state == null || span == null) return gpu::INVALID_HANDLE~;
    if (span.owner != state.allocations.owner) return gpu::INVALID_HANDLE~;
    gpu::GpuAllocation allocation = {
        .owner      = span.owner,
        .index      = span.index,
        .generation = span.generation,
    };
    AllocationSlot* slot = state.allocations.get(allocation)!;
    if (slot.texture_memory) return gpu::INVALID_ARGUMENT~;
    if (span.size == 0
        || !gpu::range_fits(slot.backing.size, span.offset, span.size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (slot.backing.cpu_base == null) return gpu::INVALID_ARGUMENT~;
    return slot;
}

fn NonCoherentRange? mapped_allocation_range(
    VkDeviceState* state,
    AllocationSlot* slot,
    gpu::GpuSpan* span,
) {
    if (state == null || slot == null || span == null) {
        return gpu::INVALID_HANDLE~;
    }
    vk::DeviceSize native_size = slot.backing.allocation_info.size;
    usz allocation_size = (usz)native_size;
    if (native_size == 0 || (vk::DeviceSize)allocation_size != native_size) {
        return gpu::BACKEND_ERROR~;
    }
    return normalize_mapped_memory_range(
        offset:          span.offset,
        size:            span.size,
        allocation_size: allocation_size,
        atom_size:       state.non_coherent_atom_size,
    );
}

fn void? flush_mapped_span_for_state_with_ops(
    VkDeviceState* state,
    gpu::GpuSpan span,
    AllocationNativeOps* ops,
) {
    AllocationSlot* slot = resolve_mapped_allocation_span(state, &span)!;
    if (slot.coherent) return;
    NonCoherentRange range = mapped_allocation_range(state, slot, &span)!;
    check_device_result(state, ops.flush_allocation(
        ops.user_data,
        state.allocator,
        slot.backing.allocation,
        range.offset,
        range.size,
    ))!;
}

fn void? invalidate_mapped_span_for_state_with_ops(
    VkDeviceState* state,
    gpu::GpuSpan span,
    AllocationNativeOps* ops,
) {
    AllocationSlot* slot = resolve_mapped_allocation_span(state, &span)!;
    if (slot.coherent) return;
    NonCoherentRange range = mapped_allocation_range(state, slot, &span)!;
    check_device_result(state, ops.invalidate_allocation(
        ops.user_data,
        state.allocator,
        slot.backing.allocation,
        range.offset,
        range.size,
    ))!;
}

fn void? vk_flush_mapped_span(
    gpu::Device* device,
    gpu::GpuSpan span,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    AllocationNativeOps ops = allocation_native_ops_real();
    return flush_mapped_span_for_state_with_ops(state, span, &ops);
}

fn void? vk_invalidate_mapped_span(
    gpu::Device* device,
    gpu::GpuSpan span,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    AllocationNativeOps ops = allocation_native_ops_real();
    return invalidate_mapped_span_for_state_with_ops(state, span, &ops);
}

fn gpu::GpuAllocation? vk_allocate_memory(
    gpu::Device* device,
    gpu::AllocationDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    AllocationNativeOps ops = allocation_native_ops_real();
    return allocate_memory_for_state_with_ops(state, desc, &ops);
}

fn void? vk_free_allocation(
    gpu::Device* device,
    gpu::GpuAllocation allocation,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    AllocationNativeOps ops = allocation_native_ops_real();
    return free_allocation_for_state_with_ops(state, allocation, &ops);
}

fn gpu::AllocationInfo? vk_get_allocation_info(
    gpu::Device* device,
    gpu::GpuAllocation allocation,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    return allocation_info_for_state(state, allocation);
}

fn gpu::GpuSpan? vk_get_allocation_span(
    gpu::Device* device,
    gpu::GpuAllocation allocation,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    return allocation_span_for_state(state, allocation);
}

fn void destroy_allocation_table_with_ops(
    VkDeviceState* state,
    AllocationNativeOps* ops,
) {
    if (state.allocations.slots.ptr == null) return;
    foreach (&cell : state.allocations.slots[:state.allocations.next_new]) {
        if (!cell.used) continue;
        if (cell.value.texture_memory) {
            ops.free_memory(
                ops.user_data,
                state.allocator,
                cell.value.raw_allocation,
            );
        } else {
            ops.destroy_buffer(
                ops.user_data,
                state.allocator,
                cell.value.backing.buffer,
                cell.value.backing.allocation,
            );
        }
        cell.used = false;
    }
    mem::free(state.allocations.slots.ptr);
    state.allocations = {};
}

fn void destroy_allocation_table(VkDeviceState* state) {
    AllocationNativeOps ops = allocation_native_ops_real();
    destroy_allocation_table_with_ops(state, &ops);
}