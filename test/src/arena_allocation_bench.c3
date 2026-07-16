module gpu_bench;

import gpu @public;
import std::io;
import std::time;
import std::core::mem;

const uint FRAME_ALLOCATION_ITERATIONS = 100_000;
const uint PERSISTENT_ALLOCATION_ITERATIONS = 4_096;
const usz ALLOCATION_SIZE = 64;
const usz ALLOCATION_ALIGNMENT = 16;

fn void run_frame_allocations(gpu::Device* device) {
    gpu::FrameToken frame = gpu::begin_frame(device)!!;
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < FRAME_ALLOCATION_ITERATIONS; i++) {
        (void)gpu::alloc_frame_span(&frame, ALLOCATION_SIZE, ALLOCATION_ALIGNMENT)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::end_frame(&frame)!!;
    io::printfn(
        "frame: iterations=%d size=%d align=%d %.1f ns/allocation",
        FRAME_ALLOCATION_ITERATIONS,
        ALLOCATION_SIZE,
        ALLOCATION_ALIGNMENT,
        (double)elapsed_ns / (double)FRAME_ALLOCATION_ITERATIONS,
    );
}

fn void run_persistent_allocations(gpu::Device* device) {
    gpu::GpuSpan[] spans = mem::new_array(gpu::GpuSpan, PERSISTENT_ALLOCATION_ITERATIONS);
    defer mem::free(spans.ptr);
    gpu::PersistentAllocDesc desc = {
        .size        = ALLOCATION_SIZE,
        .align       = ALLOCATION_ALIGNMENT,
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = { .graphics, .compute, .transfer },
    };

    std::time::Clock allocation_start = std::time::clock::now();
    for (uint i = 0; i < PERSISTENT_ALLOCATION_ITERATIONS; i++) {
        spans[i] = gpu::alloc_persistent_span(device, &desc)!!;
    }
    long allocation_ns = (long)allocation_start.to_now();

    std::time::Clock free_start = std::time::clock::now();
    for (uint i = 0; i < PERSISTENT_ALLOCATION_ITERATIONS; i++) {
        gpu::free_persistent_span(device, spans[i])!!;
    }
    long free_ns = (long)free_start.to_now();

    io::printfn(
        "persistent allocate: iterations=%d size=%d align=%d %.1f ns/allocation",
        PERSISTENT_ALLOCATION_ITERATIONS,
        ALLOCATION_SIZE,
        ALLOCATION_ALIGNMENT,
        (double)allocation_ns / (double)PERSISTENT_ALLOCATION_ITERATIONS,
    );
    io::printfn(
        "persistent free: iterations=%d %.1f ns/free",
        PERSISTENT_ALLOCATION_ITERATIONS,
        (double)free_ns / (double)PERSISTENT_ALLOCATION_ITERATIONS,
    );
}

fn int main() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .frames_in_flight  = 2,
        .application_name  = "gpu_arena_allocation_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_bench_device(&device)!!;

    run_frame_allocations(&device);
    run_persistent_allocations(&device);
    return 0;
}
