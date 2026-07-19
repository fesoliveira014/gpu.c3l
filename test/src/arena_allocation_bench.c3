module gpu_bench;

import gpu @public;
import std::io;
import std::time;

const uint FRAME_ALLOCATION_ITERATIONS = 100_000;
const uint CPU_WRITE_ALLOCATION_ITERATIONS = 4_096;
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

fn void run_cpu_write_allocations(gpu::Device* device) {
    gpu::GpuAllocation[CPU_WRITE_ALLOCATION_ITERATIONS] allocations;
    gpu::AllocationDesc desc = {
        .size         = ALLOCATION_SIZE,
        .alignment    = ALLOCATION_ALIGNMENT,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "cpu_write_bench",
    };

    std::time::Clock allocation_start = std::time::clock::now();
    for (uint i = 0; i < CPU_WRITE_ALLOCATION_ITERATIONS; i++) {
        allocations[i] = gpu::allocate_memory(device, &desc)!!;
    }
    long allocation_ns = (long)allocation_start.to_now();

    std::time::Clock free_start = std::time::clock::now();
    for (uint i = 0; i < CPU_WRITE_ALLOCATION_ITERATIONS; i++) {
        gpu::free_allocation(device, &allocations[i])!!;
    }
    long free_ns = (long)free_start.to_now();

    io::printfn(
        "cpu_write allocate: iterations=%d size=%d align=%d %.1f ns/allocation",
        CPU_WRITE_ALLOCATION_ITERATIONS,
        ALLOCATION_SIZE,
        ALLOCATION_ALIGNMENT,
        (double)allocation_ns / (double)CPU_WRITE_ALLOCATION_ITERATIONS,
    );
    io::printfn(
        "cpu_write free: iterations=%d %.1f ns/free",
        CPU_WRITE_ALLOCATION_ITERATIONS,
        (double)free_ns / (double)CPU_WRITE_ALLOCATION_ITERATIONS,
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
    run_cpu_write_allocations(&device);
    return 0;
}
