module gpu_test;

import gpu;
import gpu::vk;

// Headless (lavapipe): frame upload arena bump allocation, reset across the
// frame loop, ARENA_FULL, and the timeline-gated reset.

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

    fault full = @catch(gpu::alloc_frame_span(&device, gpu::ARENA_SIZE, 16));
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
