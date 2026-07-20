module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::thread;
import std::time;
import std::core::mem::allocators;

// Advisory allocation/texture create-destroy and shader-code preparation at
// 1/2/4 workers, reporting ns/op and scaling against one worker.

const char[*] BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint BENCH_ITERS_PER_WORKER = 300;
const uint[3] BENCH_WORKER_COUNTS = { 1, 2, 4 };
const uint BENCH_MAX_WORKERS = 4;

enum BenchPhase : int {
    TEXTURE,
    SHADER_CODE,
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
        case BenchPhase.TEXTURE:     return "textures";
        case BenchPhase.SHADER_CODE: return "shader_code";
        case BenchPhase.ALLOCATION:  return "allocations";
        default:                    return "mixed";
    }
}

fn void run_texture_op(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle handle = gpu::create_texture(device, &desc)!!;
    gpu::destroy_texture(device, handle)!!;
}

fn void run_shader_code_op() {
    gpu::ShaderDesc desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = BENCH_SHADER_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode code = gpu::prepare_shader_code(&desc)!!;
    (void)code;
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
        case BenchPhase.SHADER_CODE:
            run_shader_code_op();
        case BenchPhase.ALLOCATION:
            run_allocation_op(device);
        case BenchPhase.MIXED:
            switch (mix_index % 3) {
                case 0:  run_allocation_op(device);
                case 1:  run_texture_op(device);
                default: run_shader_code_op();
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

fn void run_phase(gpu::Device* device, vk::VkDeviceState* state, BenchPhase phase) {
    (void)run_phase_worker_count(device, phase, 1);
    double baseline_ns_per_op = 0;
    foreach (workers : BENCH_WORKER_COUNTS) {
        double ns_per_op = run_phase_worker_count(device, phase, workers);
        if (workers == 1) baseline_ns_per_op = ns_per_op;
        double scaling = baseline_ns_per_op / ns_per_op;
        io::printfn("phase %s workers=%d: %.1f ns/op, %.2fx scaling vs 1 thread",
            phase_name(phase), workers, ns_per_op, scaling);
    }
    uint leaks = vk::report_resource_leaks(state);
    io::printfn("phase %s: leaks=%d", phase_name(phase), leaks);
}

fn int main() {
    io::printfn("iterations=%d/worker units=ns/op", BENCH_ITERS_PER_WORKER);
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_resource_create_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    run_phase(&device, state, BenchPhase.TEXTURE);
    run_phase(&device, state, BenchPhase.SHADER_CODE);
    run_phase(&device, state, BenchPhase.ALLOCATION);
    run_phase(&device, state, BenchPhase.MIXED);

    return 0;
}
