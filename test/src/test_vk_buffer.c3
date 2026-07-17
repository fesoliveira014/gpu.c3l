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
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    // Device-local storage buffer.
    gpu::BufferDesc device_desc = {
        .size        = 4096,
        .usage       = { .storage, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "device_buffer",
    };
    gpu::BufferHandle device_buffer = gpu::create_buffer(&device, &device_desc)!!;
    assert(device_buffer.is_valid(), "device buffer handle is valid");

    // Host-visible upload buffer is persistently mapped.
    gpu::BufferDesc upload_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
        .access      = { .compute },
    };
    gpu::BufferHandle upload_buffer = gpu::create_buffer(&device, &upload_desc)!!;
    gpu::GpuSpan upload_span = gpu::get_buffer_span(&device, upload_buffer)!!;
    char[] upload_mapping = gpu::get_span_mapping(&device, upload_span)!!;
    assert(upload_mapping.ptr != null && upload_mapping.len == 256,
        "upload buffer exposes its exact mapping");
    assert(upload_span.size == 256, "span reports buffer size");
    assert(upload_span.owner == upload_buffer.owner
        && upload_span.index == upload_buffer.index
        && upload_span.generation == upload_buffer.generation,
        "span retains the buffer identity");

    // Addressable buffer exposes a non-zero GPU address.
    gpu::BufferDesc addressable_desc = {
        .size        = 1024,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = { .graphics, .compute, .transfer },
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

fn void live_buffers_block_device_destruction() @test {
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_buffer_lifetime_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    gpu::Device original = device;
    gpu::BufferDesc buffer_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = { .compute },
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buffer_desc)!!;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    assert(destroy_fault == gpu::RESOURCE_IN_USE, "a live buffer blocks device destruction");
    assert(device == original, "live-child rejection preserves the device token");

    gpu::destroy_buffer(&device, buffer)!!;
    gpu::destroy_device(&device)!!;
}

fn void range_and_alignment_validation() @test {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_range_validation_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::BufferDesc zero = { .size = 0, .usage = { .storage }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .access = { .graphics, .compute, .transfer }};
    fault f1 = @catch(gpu::create_buffer(&device, &zero));
    assert(f1 == gpu::INVALID_ARGUMENT, "zero-size buffer faults");

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage, .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .access = { .graphics, .compute, .transfer }};
    gpu::BufferHandle buf = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buf)!!;

    fault f2 = @catch(gpu::flush_buffer(&device, buf, 300, 0));
    assert(f2 == gpu::INVALID_ARGUMENT, "flush offset past end faults");
    fault f3 = @catch(gpu::flush_buffer(&device, buf, 200, usz::max));
    assert(f3 == gpu::INVALID_ARGUMENT, "overflowing flush range faults");
    fault f4 = @catch(gpu::invalidate_buffer(&device, buf, 0, 512));
    assert(f4 == gpu::INVALID_ARGUMENT, "oversized invalidate faults");

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    fault f5 = @catch(gpu::alloc_frame_span(&frame_token, 64, 0));
    assert(f5 == gpu::INVALID_ARGUMENT, "zero alignment faults");
    fault f6 = @catch(gpu::alloc_frame_span(&frame_token, 64, 3));
    assert(f6 == gpu::INVALID_ARGUMENT, "non-power-of-two alignment faults");
    fault f7 = @catch(gpu::alloc_frame_span(&frame_token, 0, 16));
    assert(f7 == gpu::INVALID_ARGUMENT, "zero-size span faults");

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::GpuSpan buffer_span = gpu::get_buffer_span(&device, buf)!!;

    gpu::GpuSpan misaligned_offset = buffer_span.checked_subspan(2, 4)!!;
    fault f8 = @catch(gpu::cmd_fill_buffer(&cmd, misaligned_offset, 0xAB));
    assert(f8 == gpu::INVALID_ARGUMENT, "misaligned fill offset faults");
    gpu::GpuSpan misaligned_size = buffer_span.checked_subspan(0, 6)!!;
    fault f9 = @catch(gpu::cmd_fill_buffer(&cmd, misaligned_size, 0xAB));
    assert(f9 == gpu::INVALID_ARGUMENT, "misaligned fill size faults");
    gpu::GpuSpan oversized = buffer_span.unchecked_subspan(0, 512);
    fault f10 = @catch(gpu::cmd_fill_buffer(&cmd, oversized, 0xAB));
    assert(f10 == gpu::INVALID_ARGUMENT, "oversized fill faults");
    gpu::GpuSpan zero_span = buffer_span.unchecked_subspan(0, 0);
    fault zero_fill = @catch(gpu::cmd_fill_buffer(&cmd, zero_span, 0xAB));
    assert(zero_fill == gpu::INVALID_ARGUMENT, "zero-size fill span faults");
    gpu::cmd_fill_buffer(&cmd, buffer_span, 0xAB)!!;

    gpu::BufferCopyDesc overlap = {
        .src = buffer_span.checked_subspan(0, 64)!!,
        .dst = buffer_span.checked_subspan(32, 64)!!,
    };
    fault f11 = @catch(gpu::cmd_copy_buffer(&cmd, &overlap));
    assert(f11 == gpu::INVALID_ARGUMENT, "overlapping same-buffer copy faults");
    gpu::BufferCopyDesc touching = {
        .src = buffer_span.checked_subspan(0, 64)!!,
        .dst = buffer_span.checked_subspan(64, 64)!!,
    };
    gpu::cmd_copy_buffer(&cmd, &touching)!!;
    gpu::BufferCopyDesc disjoint = {
        .src = buffer_span.checked_subspan(0, 64)!!,
        .dst = buffer_span.checked_subspan(128, 64)!!,
    };
    gpu::cmd_copy_buffer(&cmd, &disjoint)!!;
    gpu::BufferCopyDesc unequal = {
        .src = buffer_span.checked_subspan(0, 64)!!,
        .dst = buffer_span.checked_subspan(128, 32)!!,
    };
    fault unequal_fault = @catch(gpu::cmd_copy_buffer(&cmd, &unequal));
    assert(unequal_fault == gpu::INVALID_ARGUMENT, "unequal copy spans fault");
    gpu::BufferCopyDesc zero_copy = {
        .src = zero_span,
        .dst = buffer_span.checked_subspan(128, 64)!!,
    };
    fault zero_copy_fault = @catch(gpu::cmd_copy_buffer(&cmd, &zero_copy));
    assert(zero_copy_fault == gpu::INVALID_ARGUMENT, "zero-size copy span faults");
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}
