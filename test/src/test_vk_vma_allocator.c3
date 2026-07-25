module gpu_test;

import gpu;
import gpu::internal @public;
import gpu::internal::vk @public;
import std::core::mem;

// Headless (lavapipe): the device's VMA allocator + memory stats.


fn void memory_queries_reject_uninitialized_allocator() @test {
    gpu::internal::vk::VkDeviceState* state =
        mem::new(gpu::internal::vk::VkDeviceState);
    gpu::Device device = gpu::internal::register_device(
        state:  state,
        caps:   {},
    )!!;

    assert(@catch(gpu::get_memory_stats(&device)) == gpu::INVALID_ARGUMENT,
        "stats reject an uninitialized allocator before native work");
    assert(@catch(gpu::build_memory_report(&device, false))
        == gpu::INVALID_ARGUMENT,
        "reports reject an uninitialized allocator before native work");

    gpu::destroy_device(&device)!!;
}

fn void allocator_stats_and_report() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vma_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::MemoryStats stats = gpu::get_memory_stats(&device)!!;
    assert(stats.heap_count >= 1, "device reports at least one memory heap");

    bool any_budget = false;
    for (uint i = 0; i < stats.heap_count; i++) {
        if (stats.heaps[i].budget > 0) any_budget = true;
    }
    assert(any_budget, "at least one heap reports a budget > 0");

    String report = gpu::build_memory_report(&device, false)!!;
    defer report.free(mem);
    assert(report.len > 0, "memory report is non-empty");
}
