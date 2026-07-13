module gpu_test;

import gpu::compat;

// Headless (lavapipe): VMA-backed buffer create/destroy, address, span, handle
// validation.

fn void buffer_create_address_span() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_buffer_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    // Device-local storage buffer.
    gpu::compat::BufferDesc device_desc = {
        .size        = 4096,
        .usage       = { .storage, .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "device_buffer",
    };
    gpu::compat::BufferHandle device_buffer = gpu::compat::create_buffer(&device, &device_desc)!!;
    assert(device_buffer.is_valid(), "device buffer handle is valid");

    // Host-visible upload buffer is persistently mapped.
    gpu::compat::BufferDesc upload_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.FRAME_UPLOAD,
    };
    gpu::compat::BufferHandle upload_buffer = gpu::compat::create_buffer(&device, &upload_desc)!!;
    gpu::compat::GpuSpan upload_span = gpu::compat::get_buffer_span(&device, upload_buffer)!!;
    assert(upload_span.cpu != null, "upload buffer span is mapped");
    assert(upload_span.size == 256, "span reports buffer size");

    // Addressable buffer exposes a non-zero GPU address.
    gpu::compat::BufferDesc addressable_desc = {
        .size        = 1024,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
    };
    gpu::compat::BufferHandle addressable_buffer = gpu::compat::create_buffer(&device, &addressable_desc)!!;
    gpu::compat::GpuAddress address = gpu::compat::get_buffer_address(&device, addressable_buffer)!!;
    assert((ulong)address != 0, "addressable buffer has a non-zero GPU address");

    // A non-addressable buffer faults on get_buffer_address.
    fault address_fault = @catch(gpu::compat::get_buffer_address(&device, device_buffer));
    assert(address_fault == gpu::compat::INVALID_ARGUMENT, "non-addressable buffer get_buffer_address faults INVALID_ARGUMENT");

    // Destroying retires the handle.
    gpu::compat::destroy_buffer(&device, device_buffer)!!;
    fault stale_fault = @catch(gpu::compat::get_buffer_span(&device, device_buffer));
    assert(stale_fault == gpu::compat::INVALID_HANDLE, "destroyed handle returns INVALID_HANDLE");

    gpu::compat::destroy_buffer(&device, upload_buffer)!!;
    gpu::compat::destroy_buffer(&device, addressable_buffer)!!;
}

fn void range_and_alignment_validation() @test {
    gpu::compat::DeviceDesc device_desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_range_validation_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc zero = { .size = 0, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    fault f1 = @catch(gpu::compat::create_buffer(&device, &zero));
    assert(f1 == gpu::compat::INVALID_ARGUMENT, "zero-size buffer faults");

    gpu::compat::BufferDesc desc = { .size = 256, .usage = { .storage, .transfer_src, .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    gpu::compat::BufferHandle buf = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buf)!!;

    fault f2 = @catch(gpu::compat::flush_buffer(&device, buf, 300, 0));
    assert(f2 == gpu::compat::INVALID_ARGUMENT, "flush offset past end faults");
    fault f3 = @catch(gpu::compat::flush_buffer(&device, buf, 200, usz::max));
    assert(f3 == gpu::compat::INVALID_ARGUMENT, "overflowing flush range faults");
    fault f4 = @catch(gpu::compat::invalidate_buffer(&device, buf, 0, 512));
    assert(f4 == gpu::compat::INVALID_ARGUMENT, "oversized invalidate faults");

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    fault f5 = @catch(gpu::compat::alloc_frame_span(&frame_token, 64, 0));
    assert(f5 == gpu::compat::INVALID_ARGUMENT, "zero alignment faults");
    fault f6 = @catch(gpu::compat::alloc_frame_span(&frame_token, 64, 3));
    assert(f6 == gpu::compat::INVALID_ARGUMENT, "non-power-of-two alignment faults");
    fault f7 = @catch(gpu::compat::alloc_frame_span(&frame_token, 0, 16));
    assert(f7 == gpu::compat::INVALID_ARGUMENT, "zero-size span faults");

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    fault f8 = @catch(gpu::compat::cmd_fill_buffer(&cmd, buf, 2, 4, 0xAB));
    assert(f8 == gpu::compat::INVALID_ARGUMENT, "misaligned fill offset faults");
    fault f9 = @catch(gpu::compat::cmd_fill_buffer(&cmd, buf, 0, 6, 0xAB));
    assert(f9 == gpu::compat::INVALID_ARGUMENT, "misaligned fill size faults");
    fault f10 = @catch(gpu::compat::cmd_fill_buffer(&cmd, buf, 0, 512, 0xAB));
    assert(f10 == gpu::compat::INVALID_ARGUMENT, "oversized fill faults");
    gpu::compat::cmd_fill_buffer(&cmd, buf, 0, 0, 0xAB)!!;

    gpu::compat::BufferCopyDesc overlap = { .src = buf, .dst = buf, .src_offset = 0, .dst_offset = 32, .size = 64 };
    fault f11 = @catch(gpu::compat::cmd_copy_buffer(&cmd, &overlap));
    assert(f11 == gpu::compat::INVALID_ARGUMENT, "overlapping same-buffer copy faults");
    gpu::compat::BufferCopyDesc touching = { .src = buf, .dst = buf, .src_offset = 0, .dst_offset = 64, .size = 64 };
    gpu::compat::cmd_copy_buffer(&cmd, &touching)!!;
    gpu::compat::BufferCopyDesc disjoint = { .src = buf, .dst = buf, .src_offset = 0, .dst_offset = 128, .size = 64 };
    gpu::compat::cmd_copy_buffer(&cmd, &disjoint)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;
}
