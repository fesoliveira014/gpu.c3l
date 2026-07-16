module gpu_test;

import gpu;
import std::core::mem;

// Headless (lavapipe): the device's VMA allocator + memory stats.

fn void allocator_stats_and_report() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_vma_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
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
