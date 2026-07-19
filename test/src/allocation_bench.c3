module gpu_bench;

import gpu @public;
import std::io;
import std::time;

const uint ALLOCATION_ITERATIONS = 4_000;
const usz ALLOCATION_SIZE = 64;
const usz ALLOCATION_ALIGNMENT = 16;

fn void run_cpu_write_allocations(gpu::Device* device) {
    gpu::GpuAllocation[ALLOCATION_ITERATIONS] allocations;
    gpu::AllocationDesc desc = {
        .size         = ALLOCATION_SIZE,
        .alignment    = ALLOCATION_ALIGNMENT,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "cpu_write_bench",
    };

    std::time::Clock allocation_start = std::time::clock::now();
    for (uint i = 0; i < ALLOCATION_ITERATIONS; i++) {
        allocations[i] = gpu::allocate_memory(device, &desc)!!;
    }
    long allocation_ns = (long)allocation_start.to_now();

    std::time::Clock free_start = std::time::clock::now();
    for (uint i = 0; i < ALLOCATION_ITERATIONS; i++) {
        gpu::free_allocation(device, &allocations[i])!!;
    }
    long free_ns = (long)free_start.to_now();

    io::printfn(
        "cpu_write allocate: iterations=%d size=%d align=%d %.1f ns/allocation",
        ALLOCATION_ITERATIONS,
        ALLOCATION_SIZE,
        ALLOCATION_ALIGNMENT,
        (double)allocation_ns / (double)ALLOCATION_ITERATIONS,
    );
    io::printfn(
        "cpu_write free: iterations=%d size=%d align=%d %.1f ns/free",
        ALLOCATION_ITERATIONS,
        ALLOCATION_SIZE,
        ALLOCATION_ALIGNMENT,
        (double)free_ns / (double)ALLOCATION_ITERATIONS,
    );
}

fn int main() {
    gpu::DeviceDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_allocation_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_bench_device(&device)!!;

    run_cpu_write_allocations(&device);
    return 0;
}