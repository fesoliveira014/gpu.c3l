module gpu_test;

import gpu;

// Headless (lavapipe): VMA-backed buffer create/destroy, address, span, handle
// validation.

fn void buffer_create_address_span() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_buffer_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    // Device-local storage buffer.
    gpu::BufferDesc device_desc = {
        .size        = 4096,
        .usage       = { .storage, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "device_buffer",
    };
    gpu::BufferHandle device_buffer = gpu::create_buffer(&device, &device_desc)!!;
    assert(device_buffer.is_valid(), "device buffer handle is valid");

    // Host-visible upload buffer is persistently mapped.
    gpu::BufferDesc upload_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
    };
    gpu::BufferHandle upload_buffer = gpu::create_buffer(&device, &upload_desc)!!;
    gpu::GpuSpan upload_span = gpu::get_buffer_span(&device, upload_buffer)!!;
    assert(upload_span.cpu != null, "upload buffer span is mapped");
    assert(upload_span.size == 256, "span reports buffer size");

    // Addressable buffer exposes a non-zero GPU address.
    gpu::BufferDesc addressable_desc = {
        .size        = 1024,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle addressable_buffer = gpu::create_buffer(&device, &addressable_desc)!!;
    gpu::GpuAddress address = gpu::get_buffer_address(&device, addressable_buffer)!!;
    assert((ulong)address != 0, "addressable buffer has a non-zero GPU address");

    // A non-addressable buffer faults on get_buffer_address.
    fault address_fault = @catch(gpu::get_buffer_address(&device, device_buffer));
    assert(address_fault == gpu::INVALID_ARGUMENT, "non-addressable buffer get_buffer_address faults INVALID_ARGUMENT");

    // Destroying retires the handle.
    gpu::destroy_buffer(&device, device_buffer)!!;
    fault stale_fault = @catch(gpu::get_buffer_span(&device, device_buffer));
    assert(stale_fault == gpu::INVALID_HANDLE, "destroyed handle returns INVALID_HANDLE");

    gpu::destroy_buffer(&device, upload_buffer)!!;
    gpu::destroy_buffer(&device, addressable_buffer)!!;
}

fn void range_and_alignment_validation() @test {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_range_validation_test",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc zero = { .size = 0, .usage = { .storage }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    fault f1 = @catch(gpu::create_buffer(&device, &zero));
    assert(f1 == gpu::INVALID_ARGUMENT, "zero-size buffer faults");

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage, .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::BufferHandle buf = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buf)!!;

    fault f2 = @catch(gpu::flush_buffer(&device, buf, 300, 0));
    assert(f2 == gpu::INVALID_ARGUMENT, "flush offset past end faults");
    fault f3 = @catch(gpu::flush_buffer(&device, buf, 200, usz::max));
    assert(f3 == gpu::INVALID_ARGUMENT, "overflowing flush range faults");
    fault f4 = @catch(gpu::invalidate_buffer(&device, buf, 0, 512));
    assert(f4 == gpu::INVALID_ARGUMENT, "oversized invalidate faults");

    gpu::begin_frame(&device)!!;
    fault f5 = @catch(gpu::alloc_frame_span(&device, 64, 0));
    assert(f5 == gpu::INVALID_ARGUMENT, "zero alignment faults");
    fault f6 = @catch(gpu::alloc_frame_span(&device, 64, 3));
    assert(f6 == gpu::INVALID_ARGUMENT, "non-power-of-two alignment faults");
    fault f7 = @catch(gpu::alloc_frame_span(&device, 0, 16));
    assert(f7 == gpu::INVALID_ARGUMENT, "zero-size span faults");

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    fault f8 = @catch(gpu::cmd_fill_buffer(&cmd, buf, 2, 4, 0xAB));
    assert(f8 == gpu::INVALID_ARGUMENT, "misaligned fill offset faults");
    fault f9 = @catch(gpu::cmd_fill_buffer(&cmd, buf, 0, 6, 0xAB));
    assert(f9 == gpu::INVALID_ARGUMENT, "misaligned fill size faults");
    fault f10 = @catch(gpu::cmd_fill_buffer(&cmd, buf, 0, 512, 0xAB));
    assert(f10 == gpu::INVALID_ARGUMENT, "oversized fill faults");
    gpu::cmd_fill_buffer(&cmd, buf, 0, 0, 0xAB)!!;

    gpu::BufferCopyDesc overlap = { .src = buf, .dst = buf, .src_offset = 0, .dst_offset = 32, .size = 64 };
    fault f11 = @catch(gpu::cmd_copy_buffer(&cmd, &overlap));
    assert(f11 == gpu::INVALID_ARGUMENT, "overlapping same-buffer copy faults");
    gpu::BufferCopyDesc disjoint = { .src = buf, .dst = buf, .src_offset = 0, .dst_offset = 128, .size = 64 };
    gpu::cmd_copy_buffer(&cmd, &disjoint)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}
