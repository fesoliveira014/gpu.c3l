module gpu_test;

import gpu;

fn void allocation_spans_expose_mapping_address_and_liveness() @test {
    gpu::RuntimeDesc desc = {
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
    char[] upload_mapping = gpu::get_span_mapping(&device, upload.span);
    assert(upload_mapping.ptr != null && upload_mapping.len == 256,
        "CPU_WRITE span exposes its exact mapping");

    gpu::GpuAddress address = gpu::get_span_address(&device, device_data.span);
    assert((ulong)address != 0, "allocation span has a non-zero GPU address");

    free_owned_span(&device, &device_data)!!;
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

fn void span_ranges_preserve_native_bounds() @test {
    gpu::RuntimeDesc device_desc = {
        .enable_vulkan_validation = true,
        .application_name  = "gpu_range_validation_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan mapped = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "range_data",
    )!!;
    defer free_owned_span(&device, &mapped)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::cmd_fill_buffer(&commands, mapped.span, 0xAB)!!;

    gpu::BufferCopyDesc touching = {
        .src = mapped.span.subspan(0, 64),
        .dst = mapped.span.subspan(64, 64),
    };
    gpu::cmd_copy_buffer(&commands, &touching)!!;
    gpu::BufferCopyDesc reverse_touching = {
        .src = mapped.span.subspan(64, 64),
        .dst = mapped.span.subspan(0, 64),
    };
    gpu::cmd_copy_buffer(&commands, &reverse_touching)!!;
    gpu::BufferCopyDesc disjoint = {
        .src = mapped.span.subspan(0, 64),
        .dst = mapped.span.subspan(128, 64),
    };
    gpu::cmd_copy_buffer(&commands, &disjoint)!!;

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}
