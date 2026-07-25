module gpu_test;

import gpu;

fn void allocation_spans_expose_mapping_address_and_liveness() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_span_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan device_data = allocate_owned_span(
        device:       &device,
        size:         4096,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "device_data",
    )!!;
    defer free_owned_span(&device, &device_data)!!;
    assert(device_data.allocation.is_valid(), "allocation is valid");
    assert(device_data.span.size == 4096, "span reports the allocation size");

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "upload_data",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    char[] upload_mapping = gpu::get_span_mapping(&device, upload.span)!!;
    assert(upload_mapping.ptr != null && upload_mapping.len == 256,
        "CPU_WRITE span exposes its exact mapping");

    gpu::GpuAddress address = gpu::get_span_address(&device, device_data.span)!!;
    assert((ulong)address != 0, "allocation span has a non-zero GPU address");

    gpu::GpuSpan stale = device_data.span;
    free_owned_span(&device, &device_data)!!;
    assert(@catch(gpu::get_span_address(&device, stale)) == gpu::INVALID_HANDLE,
        "freed allocation invalidates its spans");
}

fn void live_allocations_block_device_destruction() @test {
    gpu::RuntimeDesc device_desc = {
        .application_name = "gpu_allocation_lifetime_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    gpu::Device original = device;
    OwnedSpan data = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "live_data",
    )!!;

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "a live allocation blocks device destruction");
    assert(device == original, "live-child rejection preserves the device token");

    free_owned_span(&device, &data)!!;
    destroy_test_device(&device)!!;
}

fn void span_range_and_alignment_validation() @test {
    gpu::RuntimeDesc device_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_range_validation_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::AllocationDesc zero = {
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
    };
    assert(@catch(gpu::allocate_memory(&device, &zero)) == gpu::INVALID_ARGUMENT,
        "zero-size allocation faults");

    OwnedSpan mapped = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "range_data",
    )!!;
    defer free_owned_span(&device, &mapped)!!;

    gpu::GpuSpan past_end = mapped.span.unchecked_subspan(300, 1);
    assert(@catch(gpu::flush_mapped_span(&device, past_end)) == gpu::INVALID_ARGUMENT,
        "flush range past the end faults");
    gpu::GpuSpan overflow = mapped.span.unchecked_subspan(200, usz::max);
    assert(@catch(gpu::flush_mapped_span(&device, overflow)) == gpu::INVALID_ARGUMENT,
        "overflowing flush range faults");
    gpu::GpuSpan oversized_mapping = mapped.span.unchecked_subspan(0, 512);
    assert(@catch(gpu::invalidate_mapped_span(&device, oversized_mapping))
        == gpu::INVALID_ARGUMENT,
        "oversized invalidate range faults");

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::GpuSpan misaligned_offset = mapped.span.checked_subspan(2, 4)!!;
    assert(@catch(gpu::cmd_fill_buffer(&commands, misaligned_offset, 0xAB))
        == gpu::INVALID_ARGUMENT,
        "misaligned fill offset faults");
    gpu::GpuSpan misaligned_size = mapped.span.checked_subspan(0, 6)!!;
    assert(@catch(gpu::cmd_fill_buffer(&commands, misaligned_size, 0xAB))
        == gpu::INVALID_ARGUMENT,
        "misaligned fill size faults");
    gpu::GpuSpan oversized = mapped.span.unchecked_subspan(0, 512);
    assert(@catch(gpu::cmd_fill_buffer(&commands, oversized, 0xAB))
        == gpu::INVALID_ARGUMENT,
        "oversized fill faults");
    gpu::GpuSpan zero_span = mapped.span.unchecked_subspan(0, 0);
    assert(@catch(gpu::cmd_fill_buffer(&commands, zero_span, 0xAB))
        == gpu::INVALID_ARGUMENT,
        "zero-size fill span faults");
    gpu::cmd_fill_buffer(&commands, mapped.span, 0xAB)!!;

    gpu::BufferCopyDesc overlap = {
        .src = mapped.span.checked_subspan(0, 64)!!,
        .dst = mapped.span.checked_subspan(32, 64)!!,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &overlap)) == gpu::INVALID_ARGUMENT,
        "overlapping same-allocation copy faults");
    gpu::BufferCopyDesc touching = {
        .src = mapped.span.checked_subspan(0, 64)!!,
        .dst = mapped.span.checked_subspan(64, 64)!!,
    };
    gpu::cmd_copy_buffer(&commands, &touching)!!;
    gpu::BufferCopyDesc reverse_touching = {
        .src = mapped.span.checked_subspan(64, 64)!!,
        .dst = mapped.span.checked_subspan(0, 64)!!,
    };
    gpu::cmd_copy_buffer(&commands, &reverse_touching)!!;
    gpu::BufferCopyDesc disjoint = {
        .src = mapped.span.checked_subspan(0, 64)!!,
        .dst = mapped.span.checked_subspan(128, 64)!!,
    };
    gpu::cmd_copy_buffer(&commands, &disjoint)!!;
    gpu::BufferCopyDesc unequal = {
        .src = mapped.span.checked_subspan(0, 64)!!,
        .dst = mapped.span.checked_subspan(128, 32)!!,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &unequal)) == gpu::INVALID_ARGUMENT,
        "unequal copy spans fault");
    gpu::BufferCopyDesc zero_copy = {
        .src = zero_span,
        .dst = mapped.span.checked_subspan(128, 64)!!,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &zero_copy)) == gpu::INVALID_ARGUMENT,
        "zero-size copy span faults");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}
