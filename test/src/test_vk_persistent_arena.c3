module gpu_test;

import gpu;

// Headless (lavapipe): persistent arena alloc/free/reuse, alignment, stable
// addresses, stats.

fn void persistent_alloc_free_stats() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_persistent_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::PersistentArenaStats empty = gpu::get_persistent_stats(&device)!!;
    assert(empty.used_bytes == 0, "arena starts empty");
    assert(empty.capacity_bytes == gpu::PERSISTENT_ARENA_SIZE, "capacity is the arena size");

    gpu::PersistentAllocDesc desc_a = {
        .size = 256, .align = 64, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "a",
    };
    gpu::GpuSpan a = gpu::alloc_persistent_span(&device, &desc_a)!!;
    assert(a.cpu != null, "persistent span is mapped");
    assert((ulong)a.gpu != 0, "non-zero GPU address");
    assert(a.offset % 64 == 0, "offset honors alignment");
    ulong a_gpu = (ulong)a.gpu;

    gpu::PersistentArenaStats after_a = gpu::get_persistent_stats(&device)!!;
    assert(after_a.used_bytes >= 256, "used bytes rose");
    assert(after_a.allocation_count == 1, "one live allocation");

    gpu::PersistentAllocDesc desc_b = { .size = 512, .align = 256, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::GpuSpan b = gpu::alloc_persistent_span(&device, &desc_b)!!;
    assert((ulong)b.gpu != a_gpu, "B gets a distinct address");
    assert(gpu::get_persistent_stats(&device)!!.allocation_count == 2, "two live allocations");

    // Free B; A's address is unchanged, count drops.
    gpu::free_persistent_span(&device, b)!!;
    gpu::PersistentArenaStats after_free_b = gpu::get_persistent_stats(&device)!!;
    assert(after_free_b.allocation_count == 1, "back to one after free");
    assert(after_free_b.used_bytes < after_a.used_bytes + 512, "used dropped after free");
    assert((ulong)a.gpu == a_gpu, "A's GPU address is stable");

    // Reuse: C allocates into freed space, A still distinct/stable.
    gpu::PersistentAllocDesc desc_c = { .size = 128, .align = 16, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::GpuSpan c = gpu::alloc_persistent_span(&device, &desc_c)!!;
    assert((ulong)c.gpu != a_gpu, "C distinct from live A");

    gpu::free_persistent_span(&device, a)!!;
    gpu::free_persistent_span(&device, c)!!;
    gpu::PersistentArenaStats drained = gpu::get_persistent_stats(&device)!!;
    assert(drained.allocation_count == 0, "arena empty after freeing all");
    assert(drained.used_bytes == 0, "used back to 0");

    // Wrong memory kind faults.
    gpu::PersistentAllocDesc bad_kind = { .size = 64, .align = 16, .memory_kind = gpu::MemoryKind.DEVICE };
    fault kind_fault = @catch(gpu::alloc_persistent_span(&device, &bad_kind));
    assert(kind_fault == gpu::INVALID_ARGUMENT, "non-persistent memory kind faults INVALID_ARGUMENT");

    // Bogus free faults.
    gpu::GpuSpan bogus = { .buffer = a.buffer, .offset = 999_999 };
    fault free_fault = @catch(gpu::free_persistent_span(&device, bogus));
    assert(free_fault == gpu::INVALID_HANDLE, "freeing a bogus span faults INVALID_HANDLE");
}
