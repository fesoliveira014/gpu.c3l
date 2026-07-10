module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

// Headless (lavapipe): frame upload arena bump allocation, reset across the
// frame loop, ARENA_FULL, and the timeline-gated reset.

fn void frame_arena_range_preserves_fault_taxonomy() @test {
    fault zero_size = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 0,
        align: 16,
    ));
    fault zero_align = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 1,
        align: 0,
    ));
    fault bad_align = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 1,
        align: 3,
    ));
    fault huge_size = @catch(gpu::vk::frame_arena_range(
        cursor: 1,
        capacity: 256,
        size: usz::max,
        align: 1,
    ));

    assert(zero_size == gpu::INVALID_ARGUMENT, "zero size remains invalid");
    assert(zero_align == gpu::INVALID_ARGUMENT, "zero alignment remains invalid");
    assert(bad_align == gpu::INVALID_ARGUMENT, "non-power-of-two alignment remains invalid");
    assert(huge_size == gpu::ARENA_FULL, "valid huge size exhausts the arena");
}

fn void frame_arena_range_handles_exact_end() @test {
    gpu::vk::FrameArenaRange exact = gpu::vk::frame_arena_range(
        cursor: 241,
        capacity: 272,
        size: 16,
        align: 16,
    )!!;
    assert(exact.offset == 256, "padding resolves to the next aligned offset");
    assert(exact.next_cursor == 272, "exact fit reaches the arena end");

    fault one_over = @catch(gpu::vk::frame_arena_range(
        cursor: 241,
        capacity: 271,
        size: 16,
        align: 16,
    ));
    fault at_end = @catch(gpu::vk::frame_arena_range(
        cursor: 256,
        capacity: 256,
        size: 1,
        align: 1,
    ));
    assert(one_over == gpu::ARENA_FULL, "one-byte overflow faults ARENA_FULL");
    assert(at_end == gpu::ARENA_FULL, "nonzero allocation at arena end faults ARENA_FULL");
}

fn void frame_arena_range_handles_max_boundaries() @test {
    gpu::vk::FrameArenaRange high_exact = gpu::vk::frame_arena_range(
        cursor: usz::max - 1,
        capacity: usz::max,
        size: 1,
        align: 2,
    )!!;
    assert(high_exact.offset == usz::max - 1, "already-aligned high cursor is preserved");
    assert(high_exact.next_cursor == usz::max, "representable exact fit reaches usz::max");

    gpu::vk::FrameArenaRange full_exact = gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: usz::max,
        size: usz::max,
        align: 1,
    )!!;
    assert(full_exact.offset == 0 && full_exact.next_cursor == usz::max, "maximum capacity exact fit remains valid");

    fault align_overflow = @catch(gpu::vk::frame_arena_range(
        cursor: usz::max,
        capacity: usz::max,
        size: 1,
        align: 2,
    ));
    fault extent_overflow = @catch(gpu::vk::frame_arena_range(
        cursor: usz::max - 1,
        capacity: usz::max,
        size: 2,
        align: 2,
    ));
    assert(align_overflow == gpu::ARENA_FULL, "unrepresentable alignment faults ARENA_FULL");
    assert(extent_overflow == gpu::ARENA_FULL, "unrepresentable extent faults ARENA_FULL");
}

fn void frame_arena_alloc_and_reset() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_frame_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan a = gpu::alloc_frame_span(&device, 100, 16)!!;
    assert(a.cpu != null, "arena span is mapped");
    assert(a.offset == 0, "first allocation offset is 0");
    assert((ulong)a.gpu != 0, "arena GPU address is non-zero");

    ulong base_gpu = (ulong)a.gpu;
    char* base_cpu = (char*)a.cpu;

    gpu::GpuSpan b = gpu::alloc_frame_span(&device, 50, 64)!!;
    assert(b.offset == 128, "second allocation aligns up to 64 (align_up(100,64)=128)");
    assert((ulong)b.gpu == base_gpu + 128, "GPU address is base + aligned offset");
    assert((char*)b.cpu == base_cpu + 128, "CPU pointer is base + aligned offset");

    fault full = @catch(gpu::alloc_frame_span(&device, gpu::DEFAULT_FRAME_ARENA_SIZE, 16));
    assert(full == gpu::ARENA_FULL, "over-allocation returns ARENA_FULL");

    gpu::end_frame(&device)!!;

    // Next frame slot: cursor reset to 0.
    gpu::begin_frame(&device)!!;
    gpu::GpuSpan c = gpu::alloc_frame_span(&device, 10, 16)!!;
    assert(c.offset == 0, "new frame slot starts at offset 0");
    gpu::end_frame(&device)!!;

    // Wrap back to the first slot (waits on its host-signaled timeline value).
    gpu::begin_frame(&device)!!;
    gpu::GpuSpan d = gpu::alloc_frame_span(&device, 10, 16)!!;
    assert(d.offset == 0, "wrapped slot cursor is reset");
    gpu::end_frame(&device)!!;
}

fn void reset_before_retire_rejected() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)device.backend_state;

    // Slot demands a timeline value the counter has not reached.
    state.frame_arenas[0].frame_timeline_value = 999;
    fault rejected = @catch(gpu::vk::reset_frame_arena(state, 0));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "premature reset returns INVALID_RESOURCE_STATE");

    // A retired slot resets fine.
    state.frame_arenas[0].frame_timeline_value = 0;
    assert(!@catch(gpu::vk::reset_frame_arena(state, 0)), "retired slot may be reset");
}

fn void custom_frame_arena_size_holds_larger_spans() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .frame_arena_size  = 32 << 20,
        .application_name  = "gpu_frame_arena_size_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    // Well past the old 1 MiB default and the 16 MiB current default.
    gpu::GpuSpan span = gpu::alloc_frame_span(&device, 20 << 20, 16)!!;
    assert(span.size == 20 << 20, "oversized span allocates in a custom-sized arena");
    gpu::end_frame(&device)!!;
}

fn void frame_upload_memory_is_coherent() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_coherency_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
        .debug_name  = "coherency_probe",
    };
    gpu::BufferHandle buf = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, buf)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::BufferSlot* slot = state.buffers.get(buf)!!;
    vk::MemoryPropertyFlags props = state.allocator.allocation_memory_properties(slot.allocation);
    assert(((uint)props & (uint)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT) != 0,
        "frame-upload memory is host-coherent");
}
