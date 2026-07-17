module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import vma;

// Concrete (non-generic) slot table for buffers. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint BUFFER_FREE_NONE = 0;

// Backend record stored per buffer slot.
struct BufferSlot {
    vk::Buffer          buffer;
    vma::Allocation     allocation;
    vma::AllocationInfo allocation_info;
    gpu::GpuAddress     gpu_base;
    void*               cpu_base;
    usz                 size;
    gpu::BufferUsage    usage;
    gpu::QueueRoles     access;
    gpu::MemoryKind     memory_kind;
    char[DEBUG_NAME_CAP] debug_name;
    bool                internal;
}

struct BufferCell {
    BufferSlot value;
    uint       generation;
    uint       next_free;
    bool       used;
}

struct BufferTable {
    BufferCell[] slots;
    ulong        owner;
    uint         free_head;
    uint         next_new;
}

fn gpu::BufferHandle? BufferTable.alloc(&self, BufferSlot value) {
    uint idx;
    bool found = false;
    while (self.free_head != BUFFER_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = BUFFER_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::BufferHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn BufferSlot*? BufferTable.get(&self, gpu::BufferHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? BufferTable.free(&self, gpu::BufferHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = BUFFER_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn vk::BufferUsageFlags buffer_usage_to_vk(gpu::BufferUsage usage) @private {
    uint flags = 0;
    if (usage.transfer_src) flags |= (uint)vk::BufferUsageFlagBits.TRANSFER_SRC_BIT;
    if (usage.transfer_dst) flags |= (uint)vk::BufferUsageFlagBits.TRANSFER_DST_BIT;
    if (usage.uniform)      flags |= (uint)vk::BufferUsageFlagBits.UNIFORM_BUFFER_BIT;
    if (usage.storage)      flags |= (uint)vk::BufferUsageFlagBits.STORAGE_BUFFER_BIT;
    if (usage.addressable)  flags |= (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    if (usage.indirect)     flags |= (uint)vk::BufferUsageFlagBits.INDIRECT_BUFFER_BIT;
    if (usage.index)        flags |= (uint)vk::BufferUsageFlagBits.INDEX_BUFFER_BIT;
    if (usage.vertex)       flags |= (uint)vk::BufferUsageFlagBits.VERTEX_BUFFER_BIT;
    return (vk::BufferUsageFlags)flags;
}

fn vma::AllocationCreateInfo memory_kind_to_vma(gpu::MemoryKind kind) @private {
    switch (kind) {
        case gpu::MemoryKind.FRAME_UPLOAD:
            // Roots are written mapped with no flush hook — require coherent
            // memory (a HOST_VISIBLE|HOST_COHERENT type is spec-guaranteed).
            return {
                .usage          = vma::MemoryUsage.AUTO_PREFER_HOST,
                .flags          = { .host_access_sequential_write, .mapped },
                .required_flags = (vk::MemoryPropertyFlags)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT,
            };
        case gpu::MemoryKind.PERSISTENT_UPLOAD:
            return { .usage = vma::MemoryUsage.AUTO, .flags = { .host_access_sequential_write, .mapped } };
        case gpu::MemoryKind.DEVICE:
            return { .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE };
        case gpu::MemoryKind.READBACK:
            return { .usage = vma::MemoryUsage.AUTO_PREFER_HOST, .flags = { .host_access_random, .mapped } };
        case gpu::MemoryKind.STAGING:
            return { .usage = vma::MemoryUsage.AUTO, .flags = { .host_access_sequential_write, .mapped } };
        default:
            return { .usage = vma::MemoryUsage.AUTO };
    }
}

fn vk::BufferCreateInfo? make_buffer_create_info(gpu::BufferDesc* desc, uint[] queue_families) @private {
    if (desc.size == 0) return gpu::INVALID_ARGUMENT~;
    vk::BufferUsageFlags usage = buffer_usage_to_vk(desc.usage);
    if ((uint)usage == 0) return gpu::INVALID_ARGUMENT~;
    vk::BufferCreateInfo info = vk::buffer_create_info()
        .set_size((vk::DeviceSize)desc.size)
        .set_usage(usage);
    if (queue_families.len >= 2) {
        info = info
            .set_sharing_mode(vk::SharingMode.CONCURRENT)
            .set_queue_family_indices(queue_families);
    }
    return info;
}

fn gpu::BufferHandle? create_buffer_with_families_for_state(
    VkDeviceState* state,
    gpu::BufferDesc* desc,
    uint[] queue_families,
) @private {
    vk::BufferCreateInfo buffer_info = make_buffer_create_info(desc, queue_families)!;
    vma::AllocationCreateInfo alloc_info = memory_kind_to_vma(desc.memory_kind);

    BufferSlot slot = {
        .size        = desc.size,
        .usage       = desc.usage,
        .access      = desc.access,
        .memory_kind = desc.memory_kind,
    };
    check_device_result(state, state.allocator.create_buffer(
        buffer_info:    &buffer_info,
        alloc_info:     &alloc_info,
        out_buffer:     &slot.buffer,
        out_allocation: &slot.allocation,
        out_info:       &slot.allocation_info,
    ))!;
    slot.cpu_base = slot.allocation_info.mapped_data;

    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(slot.allocation, desc.debug_name);
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.BUFFER,
            object_handle: (ulong)slot.buffer,
            name:          desc.debug_name,
        );
    }

    if (desc.usage.addressable) {
        vk::BufferDeviceAddressInfo address_info = vk::buffer_device_address_info().set_buffer(slot.buffer);
        vk::DeviceAddress address = vk::get_buffer_device_address(state.device, &address_info);
        if (address == 0) {
            state.allocator.destroy_buffer(slot.buffer, slot.allocation);
            return gpu::BACKEND_ERROR~;
        }
        slot.gpu_base = (gpu::GpuAddress)address;
    }

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        if (try handle = state.buffers.alloc(slot)) {
            return handle;
        }
    }
    state.allocator.destroy_buffer(slot.buffer, slot.allocation);
    return gpu::SLOT_TABLE_FULL~;
}

fn gpu::BufferHandle? create_buffer_for_state(VkDeviceState* state, gpu::BufferDesc* desc) @private {
    uint[MAX_QUEUE_FAMILIES] families;
    uint count = resource_queue_families(state, desc.access, families[..]);
    return create_buffer_with_families_for_state(state, desc, families[:count]);
}

fn gpu::BufferHandle? vk_create_buffer(gpu::Device* device, gpu::BufferDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    if (desc.size == 0) {
        gpu::DebugMessage message = {
            .severity       = gpu::DebugMessageSeverity.ERROR,
            .category       = { .public_contract },
            .operation      = "create_buffer",
            .has_fault      = true,
            .public_fault   = gpu::INVALID_ARGUMENT,
            .resource       = { .kind = gpu::DebugResourceKind.BUFFER, .debug_name = desc.debug_name },
            .rejected_field = "BufferDesc.size",
            .invariant      = "must be greater than zero",
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
        return gpu::INVALID_ARGUMENT~;
    }
    if ((uint)buffer_usage_to_vk(desc.usage) == 0) {
        gpu::DebugMessage message = {
            .severity       = gpu::DebugMessageSeverity.ERROR,
            .category       = { .public_contract },
            .operation      = "create_buffer",
            .has_fault      = true,
            .public_fault   = gpu::INVALID_ARGUMENT,
            .resource       = { .kind = gpu::DebugResourceKind.BUFFER, .debug_name = desc.debug_name },
            .rejected_field = "BufferDesc.usage",
            .invariant      = "must include at least one usage flag",
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
        return gpu::INVALID_ARGUMENT~;
    }
    return create_buffer_for_state(state, desc);
}

<* Frees the handle now; buffer + allocation release is deferred until every
   referencing frame retires. *>
fn void? vk_destroy_buffer(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    BufferSlot* slot = state.buffers.get(handle)!;
    vk::Buffer buffer = slot.buffer;
    vma::Allocation allocation = slot.allocation;
    ulong retire = retire_timeline_value(state);
    state.buffers.free(handle)!;
    enqueue_deferred(
        state:          state,
        kind:           DeferredKind.BUFFER,
        timeline_value: retire,
        object:         (void*)buffer,
        allocation:     allocation,
    );
}

fn gpu::GpuAddress? vk_get_buffer_address(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if ((ulong)slot.gpu_base == 0) return gpu::INVALID_ARGUMENT~;
    return slot.gpu_base;
}

fn void? vk_flush_buffer(gpu::Device* device, gpu::BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    // WHOLE_SIZE requires offset strictly inside the buffer.
    if (size == 0 && offset >= slot.size) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check_device_result(state, state.allocator.flush_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn void? vk_invalidate_buffer(gpu::Device* device, gpu::BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    if (size == 0 && offset >= slot.size) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check_device_result(state, state.allocator.invalidate_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn gpu::GpuSpan? get_buffer_span_for_state(VkDeviceState* state, gpu::BufferHandle handle) @private {
    BufferSlot* slot = state.buffers.get(handle)!;
    return {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
        .size       = slot.size,
    };
}

fn gpu::GpuSpan? vk_get_buffer_span(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return get_buffer_span_for_state(state, handle);
}

fn char[]? vk_get_span_mapping(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_mapping(&resolved);
}

fn gpu::GpuAddress? vk_get_span_address(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_address(&resolved);
}
