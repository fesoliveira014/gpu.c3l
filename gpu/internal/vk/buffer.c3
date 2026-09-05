module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import vma;

bitstruct BufferUsage : uint {
    bool transfer_src : 0;
    bool transfer_dst : 1;
    bool uniform      : 2;
    bool storage      : 3;
    bool addressable  : 4;
    bool indirect     : 5;
    bool index        : 6;
    bool vertex       : 7;
    bool acceleration_structure_storage     : 8;
    bool acceleration_structure_build_input : 9;
    bool shader_binding_table               : 10;
}

struct BufferDesc {
    usz         size;
    BufferUsage usage;
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
    char[DEBUG_NAME_CAP] debug_name;
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
    if (usage.acceleration_structure_storage) {
        flags |= (uint)vk::BufferUsageFlagBits
            .ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    }
    if (usage.acceleration_structure_build_input) {
        flags |= (uint)vk::BufferUsageFlagBits
            .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    if (usage.shader_binding_table) {
        flags |= (uint)vk::BufferUsageFlagBits.SHADER_BINDING_TABLE_BIT_KHR;
    }
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

fn char[]? vk_get_span_mapping(VkDeviceState* state, gpu::GpuSpan span) {
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_mapping(&resolved);
}

fn gpu::GpuAddress? vk_get_span_address(
    VkDeviceState* state,
    gpu::GpuSpan span,
) {
    ResolvedSpan resolved = resolve_span(state, &span)!;
    return resolved_span_address(&resolved);
}
