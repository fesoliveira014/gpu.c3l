module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::core::mem;

<* Create the per-device VMA allocator. Static-linked VMA needs no Vulkan
   function table (`vulkan_functions = null`). The buffer-device-address flag is
   mandatory; the memory-budget flag tracks the enabled extension. *>
fn void? create_vk_allocator(VkDeviceState* state) {
    vma::AllocatorCreateInfo info = {
        .instance           = state.instance,
        .physical_device    = state.physical_device,
        .device             = state.device,
        .vulkan_api_version = API_VERSION_1_3,
        .vulkan_functions   = null,
        .flags = {
            .buffer_device_address = true,
            .ext_memory_budget     = state.memory_budget_enabled,
            .khr_maintenance5       = state.generated_work.enabled,
        },
    };
    check_backend_create_result(vma::create_allocator(&info, &state.allocator))!;
}

fn gpu::MemoryStats? vk_get_memory_stats(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || state.allocator == null) return gpu::INVALID_ARGUMENT~;

    vk::PhysicalDeviceMemoryProperties mem_props;
    vk::get_physical_device_memory_properties(state.physical_device, &mem_props);
    uint heap_count = mem_props.memory_heap_count;
    if (heap_count > (uint)gpu::MAX_MEMORY_HEAPS) heap_count = (uint)gpu::MAX_MEMORY_HEAPS;

    vma::Budget[vk::MAX_MEMORY_HEAPS] budgets;
    state.allocator.get_heap_budgets(&budgets[0]);

    gpu::MemoryStats stats = {};
    stats.heap_count = heap_count;
    for (uint i = 0; i < heap_count; i++) {
        stats.heaps[i] = (gpu::MemoryHeapBudget){
            .usage            = (ulong)budgets[i].usage,
            .budget           = (ulong)budgets[i].budget,
            .allocation_bytes = (ulong)budgets[i].statistics.allocation_bytes,
            .block_bytes      = (ulong)budgets[i].statistics.block_bytes,
        };
        stats.live_allocation_count += (ulong)budgets[i].statistics.allocation_count;
    }
    if (state.textures.slots.ptr != null) {
        foreach (&cell : state.textures.slots[:state.textures.next_new]) {
            if (cell.used) stats.texture_count++;
        }
    }
    return stats;
}

fn String? vk_build_memory_report(gpu::Device* device, bool detailed) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || state.allocator == null) return gpu::INVALID_ARGUMENT~;

    ZString vma_string;
    state.allocator.build_stats_string(&vma_string, detailed ? vk::TRUE : vk::FALSE);
    defer state.allocator.free_stats_string(vma_string);
    return vma_string.str_view().copy(mem);
}
