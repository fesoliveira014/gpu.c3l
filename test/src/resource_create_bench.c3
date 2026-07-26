module gpu_bench;

import gpu @public;
import std::io;
import std::thread;
import std::time;
import std::core::mem::allocators;

// Advisory allocation/texture create-destroy at 1/2/4 workers, reporting
// ns/op and scaling against one worker.

const uint BENCH_ITERS_PER_WORKER = 300;
const uint[3] BENCH_WORKER_COUNTS = { 1, 2, 4 };
const uint BENCH_MAX_WORKERS = 4;
const uint CPU_WRITE_ITERATIONS = 4_000;
const uint DESCRIPTOR_ITERATIONS = 320;
const uint DESCRIPTOR_BATCH_SIZE = 16;

enum BenchPhase : int {
    TEXTURE,
    ALLOCATION,
    MIXED,
}

struct BenchWorkerArgs {
    gpu::Device* device;
    BenchPhase   phase;
    uint         seed;
    uint         iters;
}

fn ZString phase_name(BenchPhase phase) {
    switch (phase) {
        case BenchPhase.TEXTURE:    return "textures";
        case BenchPhase.ALLOCATION: return "allocations";
        default:                    return "mixed";
    }
}

fn void run_texture_op(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle handle = gpu::create_texture(device, &desc)!!;
    gpu::destroy_texture(device, handle)!!;
}

fn void run_allocation_op(gpu::Device* device) {
    gpu::AllocationDesc desc = {
        .size         = 256,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!!;
    gpu::free_allocation(device, &allocation)!!;
}

fn void run_phase_op(gpu::Device* device, BenchPhase phase, uint mix_index) {
    switch (phase) {
        case BenchPhase.TEXTURE:
            run_texture_op(device);
        case BenchPhase.ALLOCATION:
            run_allocation_op(device);
        case BenchPhase.MIXED:
            switch (mix_index % 2) {
                case 0:  run_allocation_op(device);
                default: run_texture_op(device);
            }
    }
}

fn int bench_worker(void* arg) {
    BenchWorkerArgs* args = (BenchWorkerArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        for (uint i = 0; i < args.iters; i++) {
            run_phase_op(args.device, args.phase, args.seed * args.iters + i);
        }
    };
    return 0;
}

<* Wall-clock ns/op across `workers` concurrent workers, each running
   `BENCH_ITERS_PER_WORKER` operations for `phase`. *>
fn double run_phase_worker_count(gpu::Device* device, BenchPhase phase, uint workers) {
    BenchWorkerArgs[BENCH_MAX_WORKERS] args;
    Thread[BENCH_MAX_WORKERS] threads;
    for (uint w = 0; w < workers; w++) {
        args[w] = { .device = device, .phase = phase, .seed = w, .iters = BENCH_ITERS_PER_WORKER };
    }

    std::time::Clock start = std::time::clock::now();
    for (uint w = 0; w < workers; w++) threads[w].create(&bench_worker, &args[w])!!;
    for (uint w = 0; w < workers; w++) threads[w].join()!!;
    long elapsed_ns = (long)start.to_now();

    uint total_ops = workers * BENCH_ITERS_PER_WORKER;
    return (double)elapsed_ns / (double)total_ops;
}

fn void run_phase(gpu::Device* device, BenchPhase phase) {
    (void)run_phase_worker_count(device, phase, 1);
    double baseline_ns_per_op = 0;
    foreach (workers : BENCH_WORKER_COUNTS) {
        double ns_per_op = run_phase_worker_count(device, phase, workers);
        if (workers == 1) baseline_ns_per_op = ns_per_op;
        double scaling = baseline_ns_per_op / ns_per_op;
        io::printfn("phase %s workers=%d: %.1f ns/op, %.2fx scaling vs 1 thread",
            phase_name(phase), workers, ns_per_op, scaling);
    }
}

fn void run_cpu_write_allocations(gpu::Device* device) {
    gpu::GpuAllocation[CPU_WRITE_ITERATIONS] allocations;
    gpu::AllocationDesc desc = {
        .size         = 64,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "cpu_write_bench",
    };

    std::time::Clock allocation_start = std::time::clock::now();
    foreach (&allocation : allocations) {
        *allocation = gpu::allocate_memory(device, &desc)!!;
    }
    long allocation_ns = (long)allocation_start.to_now();

    std::time::Clock free_start = std::time::clock::now();
    foreach (&allocation : allocations) {
        gpu::free_allocation(device, allocation)!!;
    }
    long free_ns = (long)free_start.to_now();

    io::printfn(
        "cpu-write allocations: %d iterations, %.1f ns/allocate, %.1f ns/free",
        CPU_WRITE_ITERATIONS,
        (double)allocation_ns / CPU_WRITE_ITERATIONS,
        (double)free_ns / CPU_WRITE_ITERATIONS,
    );
}

fn void run_descriptor_churn(gpu::Device* device) {
    gpu::TextureDesc texture_desc = {
        .access = { .graphics, .compute },
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    gpu::TextureHandle texture =
        gpu::create_texture(device, &texture_desc)!!;
    defer gpu::destroy_texture(device, texture)!!;

    std::time::Clock single_start = std::time::clock::now();
    for (uint i = 0; i < DESCRIPTOR_ITERATIONS; i++) {
        gpu::TextureView view =
            gpu::create_texture_view(device, texture, null)!!;
        gpu::destroy_texture_view(device, view)!!;
    }
    long single_ns = (long)single_start.to_now();

    gpu::TextureViewCreateDesc[DESCRIPTOR_BATCH_SIZE] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[DESCRIPTOR_BATCH_SIZE] views;
    std::time::Clock batch_start = std::time::clock::now();
    for (uint i = 0; i < DESCRIPTOR_ITERATIONS / DESCRIPTOR_BATCH_SIZE; i++) {
        gpu::create_texture_views(device, descs[..], views[..])!!;
        foreach (view : views) {
            gpu::destroy_texture_view(device, view)!!;
        }
    }
    long batch_ns = (long)batch_start.to_now();

    gpu::SamplerDesc sampler_desc;
    std::time::Clock sampler_start = std::time::clock::now();
    for (uint i = 0; i < DESCRIPTOR_ITERATIONS; i++) {
        (void)gpu::intern_sampler(device, &sampler_desc)!!;
    }
    long sampler_ns = (long)sampler_start.to_now();

    io::printfn(
        "descriptor churn: %d views, %.1f ns/single, %.1f ns/batched, %.1f ns/sampler-hit",
        DESCRIPTOR_ITERATIONS,
        (double)single_ns / DESCRIPTOR_ITERATIONS,
        (double)batch_ns / DESCRIPTOR_ITERATIONS,
        (double)sampler_ns / DESCRIPTOR_ITERATIONS,
    );
}

fn int main() {
    io::printfn("iterations=%d/worker units=ns/op", BENCH_ITERS_PER_WORKER);
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_resource_create_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;

    run_phase(&device, BenchPhase.TEXTURE);
    run_phase(&device, BenchPhase.ALLOCATION);
    run_phase(&device, BenchPhase.MIXED);
    run_cpu_write_allocations(&device);
    run_descriptor_churn(&device);

    return 0;
}
