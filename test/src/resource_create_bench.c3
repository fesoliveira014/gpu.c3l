module gpu_bench;

import gpu;
import gpu::vk;
import std::io;
import std::thread;
import std::time;
import std::core::mem::allocators;

// Advisory, not gating: create/destroy round trips at 1/2/4 workers,
// reporting ns/op and scaling vs one thread. ns/op includes the still-locked
// destroy half, so create-path wins are understated. Indicative on lavapipe.

const char[*] BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint BENCH_ITERS_PER_WORKER = 300;
const uint[3] BENCH_WORKER_COUNTS = { 1, 2, 4 };
const uint BENCH_MAX_WORKERS = 4;

enum BenchPhase : int {
    BUFFER,
    TEXTURE,
    SHADER,
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
        case BenchPhase.BUFFER:  return "buffers";
        case BenchPhase.TEXTURE: return "textures";
        case BenchPhase.SHADER:  return "shaders";
        default:                 return "mixed";
    }
}

fn void run_buffer_op(gpu::Device* device) {
    gpu::BufferDesc desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle handle = gpu::create_buffer(device, &desc)!!;
    gpu::destroy_buffer(device, handle)!!;
}

fn void run_texture_op(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle handle = gpu::create_texture(device, &desc)!!;
    gpu::destroy_texture(device, handle)!!;
}

fn void run_shader_op(gpu::Device* device) {
    gpu::ShaderDesc desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = BENCH_SHADER_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle handle = gpu::create_shader(device, &desc)!!;
    gpu::destroy_shader(device, handle)!!;
}

fn void run_phase_op(gpu::Device* device, BenchPhase phase, uint mix_index) {
    switch (phase) {
        case BenchPhase.BUFFER:
            run_buffer_op(device);
        case BenchPhase.TEXTURE:
            run_texture_op(device);
        case BenchPhase.SHADER:
            run_shader_op(device);
        case BenchPhase.MIXED:
            switch (mix_index % 3) {
                case 0:  run_buffer_op(device);
                case 1:  run_texture_op(device);
                default: run_shader_op(device);
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
   `BENCH_ITERS_PER_WORKER` create+destroy round trips of `phase`. *>
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
    gpu::DeviceDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_resource_create_bench",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    run_phase(&device, state, BenchPhase.BUFFER);
    run_phase(&device, state, BenchPhase.TEXTURE);
    run_phase(&device, state, BenchPhase.SHADER);
    run_phase(&device, state, BenchPhase.MIXED);

    return 0;
}
