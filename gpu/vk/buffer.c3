module gpu::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::thread;
import vma;
import std::atomic::types;

const usz MAX_BUFFERS = 4096;
const uint BUFFER_FREE_NONE = 0;
const BufferHandle BUFFER_HANDLE_INVALID = {};

bitstruct BufferUsage : uint {
    bool transfer_src : 0;
    bool transfer_dst : 1;
    bool uniform      : 2;
    bool storage      : 3;
    bool addressable  : 4;
    bool indirect     : 5;
    bool index        : 6;
    bool vertex       : 7;
}

struct BufferHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert BufferHandle::size <= ulong::size * 2;

struct BufferDesc {
    usz              size;
    BufferUsage      usage;
    gpu::MemoryClass memory_class;
    gpu::QueueRoles  access;
    ZString          debug_name;
}

fn bool BufferHandle.is_valid(self) => self.owner != 0 && self.generation != 0;

fn bool BufferHandle.equals(self, BufferHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}

struct BufferSlot {
    vk::Buffer          buffer;
    vma::Allocation     allocation;
    vma::AllocationInfo allocation_info;
    gpu::GpuAddress     gpu_base;
    void*               cpu_base;
    usz                 size;
    BufferUsage         usage;
    gpu::QueueRoles     access;
    gpu::MemoryClass    memory_class;
    Atomic{uint}        tracked_command_references;
    char[DEBUG_NAME_CAP] debug_name;
    bool                internal;
}

struct BufferCell {
    BufferSlot value;
    uint       generation;
    uint       next_free;
    bool       used;
}

// Monomorphized twin of gpu::slots avoids a c3c 0.8.0 codegen failure.
struct BufferTable {
    BufferCell[] slots;
    ulong        owner;
    uint         free_head;
    uint         next_new;
}

fn BufferHandle? BufferTable.alloc(&self, BufferSlot value) {
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
    return (BufferHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn BufferSlot*? BufferTable.get(&self, BufferHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? BufferTable.free(&self, BufferHandle h) {
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

fn vk::BufferUsageFlags buffer_usage_to_vk(BufferUsage usage) @private {
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

fn vk::BufferCreateInfo? make_buffer_create_info(BufferDesc* desc, uint[] queue_families) @private {
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

fn BufferHandle? create_buffer_with_families_for_state(
    VkDeviceState* state,
    BufferDesc* desc,
    uint[] queue_families,
) @private {
    vk::BufferCreateInfo buffer_info = make_buffer_create_info(desc, queue_families)!;
    vma::AllocationCreateInfo alloc_info = memory_class_to_vma(desc.memory_class)!;

    BufferSlot slot = {
        .size         = desc.size,
        .usage        = desc.usage,
        .access       = desc.access,
        .memory_class = desc.memory_class,
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

fn BufferHandle? create_buffer_for_state(VkDeviceState* state, BufferDesc* desc) @private {
    uint[MAX_QUEUE_FAMILIES] families;
    uint count = resource_queue_families(state, desc.access, families[..]);
    return create_buffer_with_families_for_state(state, desc, families[:count]);
}

fn BufferHandle? vk_create_buffer(gpu::Device* device, BufferDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    return create_buffer_for_state(state, desc);
}

fn void? vk_destroy_buffer(gpu::Device* device, BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    BufferSlot* slot = state.buffers.get(handle)!;
    if (slot.tracked_command_references.load(AtomicOrdering.RELAXED) > 0) {
        drain_completed_submitted_commands(state)!;
        if (slot.tracked_command_references.load(AtomicOrdering.RELAXED) > 0) {
            return gpu::RESOURCE_IN_USE~;
        }
    }
    vk::Buffer buffer = slot.buffer;
    vma::Allocation allocation = slot.allocation;
    state.buffers.free(handle)!;
    state.allocator.destroy_buffer(buffer, allocation);
}

fn gpu::GpuAddress? vk_get_buffer_address(gpu::Device* device, BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if ((ulong)slot.gpu_base == 0) return gpu::INVALID_ARGUMENT~;
    return slot.gpu_base;
}

fn void? vk_flush_buffer(gpu::Device* device, BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    // WHOLE_SIZE requires offset strictly inside the buffer.
    if (size == 0 && offset >= slot.size) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check_device_result(state, state.allocator.flush_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn void? vk_invalidate_buffer(gpu::Device* device, BufferHandle handle, usz offset, usz size) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    BufferSlot* slot = state.buffers.get(handle)!;
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    if (size == 0 && offset >= slot.size) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize range = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    check_device_result(state, state.allocator.invalidate_allocation(slot.allocation, (vk::DeviceSize)offset, range))!;
}

fn gpu::GpuSpan? get_buffer_span_for_state(VkDeviceState* state, BufferHandle handle) @private {
    BufferSlot* slot = state.buffers.get(handle)!;
    return {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
        .size       = slot.size,
    };
}

fn gpu::GpuSpan? vk_get_buffer_span(gpu::Device* device, BufferHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return get_buffer_span_for_state(state, handle);
}

fn char[]? vk_get_span_mapping(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_mapping(&resolved);
}

fn gpu::GpuAddress? vk_get_span_address(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_address(&resolved);
}
