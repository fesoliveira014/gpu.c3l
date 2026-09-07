module gpu_test;

import gpu;

fn void mapped_gpu_span_factory_uses_exact_public_views() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_mapped_span_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "mapped_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;

    char[] expected_bytes = gpu::get_span_mapping(&device, owned.span)!!;
    gpu::GpuAddress expected_address =
        gpu::get_span_address(&device, owned.span)!!;
    gpu::MappedGpuSpan mapped = gpu::mapped_gpu_span(&device, owned.span)!!;

    assert(mapped.span.owner == owned.span.owner
        && mapped.span.index == owned.span.index
        && mapped.span.generation == owned.span.generation
        && mapped.span.offset == owned.span.offset
        && mapped.span.size == owned.span.size,
        "factory preserves the exact input span");
    assert(mapped.bytes.ptr == expected_bytes.ptr
        && mapped.bytes.len == expected_bytes.len,
        "factory preserves the exact public mapping");
    assert(mapped.address == expected_address,
        "factory preserves the exact public address");

    mapped.bytes[7] = 0x5A;
    gpu::flush_mapped_span(&device, mapped.span)!!;
    char[] requeried = gpu::get_span_mapping(&device, owned.span)!!;
    assert(requeried[7] == 0x5A,
        "a flushed write is visible through a fresh public mapping query");
}

fn void mapped_gpu_span_factory_propagates_unmapped_fault() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_unmapped_span_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "unmapped_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;

    fault expected = @catch(gpu::get_span_mapping(&device, owned.span));
    assert(expected == gpu::INVALID_ARGUMENT,
        "GPU-private spans use the exact public unmapped fault");
    fault actual = @catch(gpu::mapped_gpu_span(&device, owned.span));
    assert(actual == expected,
        "factory propagates the public mapping query fault");
}

fn void allocate_mapped_memory_returns_consistent_views_and_frees_through_span() @test {
    gpu::RuntimeDesc desc = { .application_name = "gpu_allocate_mapped_test" };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::MemoryStats baseline = gpu::get_memory_stats(&device)!!;

    gpu::AllocationDesc alloc_desc = {
        .size         = 64,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .compute },
        .debug_name   = "allocate_mapped",
    };
    gpu::MappedGpuSpan mapped = gpu::allocate_mapped_memory(&device, &alloc_desc)!!;
    assert(mapped.bytes.len == mapped.span.size && mapped.span.size == 64,
        "the mapping covers the whole allocation");
    assert(mapped.address == gpu::get_span_address(&device, mapped.span)!!,
        "the address matches the public query");
    assert(gpu::get_memory_stats(&device)!!.live_allocation_count == baseline.live_allocation_count + 1,
        "one allocation is live");
    gpu::AllocationInfo info = gpu::get_allocation_info(&device, mapped.span.allocation())!!;
    assert(info.mapped && info.size == 64, "the span names a live mapped allocation");
    gpu::flush_mapped_span(&device, mapped.span)!!;
    gpu::invalidate_mapped_span(&device, mapped.span)!!;

    gpu::GpuAllocation allocation = mapped.span.allocation();
    gpu::free_allocation(&device, &allocation)!!;
    assert(gpu::get_memory_stats(&device)!!.live_allocation_count == baseline.live_allocation_count,
        "freeing through the span accessor releases the allocation");
}

fn void allocate_mapped_memory_rejects_unmapped_classes_without_allocating() @test {
    gpu::RuntimeDesc desc = { .application_name = "gpu_allocate_unmapped_test" };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::MemoryStats baseline = gpu::get_memory_stats(&device)!!;

    gpu::AllocationDesc private_desc = {
        .size         = 64,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .compute },
    };
    assert(@catch(gpu::allocate_mapped_memory(&device, &private_desc)) == gpu::INVALID_ARGUMENT,
        "GPU_PRIVATE faults INVALID_ARGUMENT");
    assert(@catch(gpu::allocate_mapped_memory(&device, null)) == gpu::INVALID_ARGUMENT,
        "a null descriptor faults INVALID_ARGUMENT");
    assert(gpu::get_memory_stats(&device)!!.live_allocation_count == baseline.live_allocation_count,
        "nothing is left allocated");
}
