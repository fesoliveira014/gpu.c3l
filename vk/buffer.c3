module gpu::vk;

import gpu;
import vk;
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
    gpu::MemoryKind     memory_kind;
}

struct BufferCell {
    BufferSlot value;
    uint       generation;
    uint       next_free;
    bool       used;
}

struct BufferTable {
    BufferCell[] slots;
    uint         free_head;
    uint         next_new;
}

fn gpu::BufferHandle? BufferTable.alloc(&self, BufferSlot value) {
    uint idx;
    if (self.free_head != BUFFER_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation++;
    self.slots[idx].used = true;
    return (gpu::BufferHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn BufferSlot*? BufferTable.get(&self, gpu::BufferHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? BufferTable.free(&self, gpu::BufferHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
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
            return { .usage = vma::MemoryUsage.AUTO_PREFER_HOST, .flags = { .host_access_sequential_write, .mapped } };
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

fn gpu::BufferHandle? vk_create_buffer(gpu::Device* device, gpu::BufferDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;

    vk::BufferCreateInfo buffer_info = vk::buffer_create_info()
        .set_size((vk::DeviceSize)desc.size)
        .set_usage(buffer_usage_to_vk(desc.usage));
    vma::AllocationCreateInfo alloc_info = memory_kind_to_vma(desc.memory_kind);

    BufferSlot slot = {
        .size        = desc.size,
        .usage       = desc.usage,
        .memory_kind = desc.memory_kind,
    };
    check(state.allocator.create_buffer(&buffer_info, &alloc_info, &slot.buffer, &slot.allocation, &slot.allocation_info))!;
    slot.cpu_base = slot.allocation_info.mapped_data;

    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(slot.allocation, desc.debug_name);
    }

    if (desc.usage.addressable) {
        vk::BufferDeviceAddressInfo address_info = vk::buffer_device_address_info().set_buffer(slot.buffer);
        vk::DeviceAddress address = vk::get_buffer_device_address(state.device, &address_info);
        if (address == 0) {
            state.allocator.destroy_buffer(slot.buffer, slot.allocation);
            return gpu::DEVICE_LOST~;
        }
        slot.gpu_base = (gpu::GpuAddress)address;
    }

    if (try handle = state.buffers.alloc(slot)) {
        return handle;
    }
    state.allocator.destroy_buffer(slot.buffer, slot.allocation);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_buffer(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    BufferSlot* slot = state.buffers.get(handle)!;
    state.allocator.destroy_buffer(slot.buffer, slot.allocation);
    state.buffers.free(handle)!;
}

fn gpu::GpuAddress? vk_get_buffer_address(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    BufferSlot* slot = state.buffers.get(handle)!;
    if ((ulong)slot.gpu_base == 0) return gpu::INVALID_ARGUMENT~;
    return slot.gpu_base;
}

fn void? vk_flush_buffer(gpu::Device* device, gpu::BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    BufferSlot* slot = state.buffers.get(handle)!;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check(state.allocator.flush_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn void? vk_invalidate_buffer(gpu::Device* device, gpu::BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    BufferSlot* slot = state.buffers.get(handle)!;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check(state.allocator.invalidate_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn gpu::GpuSpan? vk_get_buffer_span(gpu::Device* device, gpu::BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    BufferSlot* slot = state.buffers.get(handle)!;
    return (gpu::GpuSpan){
        .gpu    = slot.gpu_base,
        .cpu    = slot.cpu_base,
        .size   = slot.size,
        .buffer = handle,
        .offset = 0,
        .kind   = slot.memory_kind,
    };
}
