module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::thread;
import std::time;
import std::core::mem::allocators;

// Advisory, not gating: texture-view singles vs batched (N=16) vs
// sampler intern-hit plus stable-publication lookups at 1/2/4 workers.
// Reports ns/descriptor or ns/op, scaling vs one thread, and the
// batch-vs-single ratio. Indicative on lavapipe.

const uint BENCH_ITERS_PER_WORKER = 320;
const uint BENCH_BATCH_SIZE = 16;
const uint[3] BENCH_WORKER_COUNTS = { 1, 2, 4 };
const uint BENCH_MAX_WORKERS = 4;

enum BenchPhase : int {
    TEXTURE_SINGLE,
    TEXTURE_BATCHED,
    SAMPLER,
}

struct BenchWorkerArgs {
    gpu::Device*        device;
    BenchPhase          phase;
    gpu::TextureHandle  texture;
    uint                iters;
}

fn ZString phase_name(BenchPhase phase) {
    switch (phase) {
        case BenchPhase.TEXTURE_SINGLE:  return "texture singles";
        case BenchPhase.TEXTURE_BATCHED: return "texture batched(16)";
        default:                         return "sampler intern+publish hits";
    }
}

fn uint descriptors_per_op(BenchPhase phase) {
    return phase == BenchPhase.TEXTURE_BATCHED ? BENCH_BATCH_SIZE : 1;
}

fn void run_texture_single_op(gpu::Device* device, gpu::TextureHandle texture) {
    gpu::TextureView view = gpu::create_texture_view(device, texture, null)!!;
    gpu::destroy_texture_view(device, view)!!;
}

fn void run_texture_batched_op(gpu::Device* device, gpu::TextureHandle texture) {
    gpu::TextureViewCreateDesc[BENCH_BATCH_SIZE] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[BENCH_BATCH_SIZE] views;
    gpu::create_texture_views(device, descs[..], views[..])!!;
    foreach (view : views) gpu::destroy_texture_view(device, view)!!;
}

fn void run_sampler_op(gpu::Device* device) {
    gpu::SamplerDesc desc = {};
    gpu::Sampler sampler = gpu::intern_sampler(device, &desc)!!;
    gpu::SamplerIndex index = gpu::publish_sampler(device, sampler)!!;
}

fn void run_phase_op(gpu::Device* device, BenchPhase phase, gpu::TextureHandle texture) {
    switch (phase) {
        case BenchPhase.TEXTURE_SINGLE:  run_texture_single_op(device, texture);
        case BenchPhase.TEXTURE_BATCHED: run_texture_batched_op(device, texture);
        case BenchPhase.SAMPLER:         run_sampler_op(device);
    }
}

fn int bench_worker(void* arg) {
    BenchWorkerArgs* args = (BenchWorkerArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        uint ops = args.iters / descriptors_per_op(args.phase);
        for (uint i = 0; i < ops; i++) {
            run_phase_op(args.device, args.phase, args.texture);
        }
    };
    return 0;
}

<* Wall-clock work across `workers` concurrent workers. Texture phases publish
   and destroy views; the sampler phase measures intern and stable
   publication lookup hits. *>
fn double run_phase_worker_count(gpu::Device* device, BenchPhase phase, gpu::TextureHandle[] textures, uint workers) {
    BenchWorkerArgs[BENCH_MAX_WORKERS] args;
    Thread[BENCH_MAX_WORKERS] threads;
    for (uint w = 0; w < workers; w++) {
        args[w] = { .device = device, .phase = phase, .texture = textures[w], .iters = BENCH_ITERS_PER_WORKER };
    }

    std::time::Clock start = std::time::clock::now();
    for (uint w = 0; w < workers; w++) threads[w].create(&bench_worker, &args[w])!!;
    for (uint w = 0; w < workers; w++) threads[w].join()!!;
    long elapsed_ns = (long)start.to_now();

    uint total_descriptors = workers * BENCH_ITERS_PER_WORKER;
    return (double)elapsed_ns / (double)total_descriptors;
}

fn double run_phase(gpu::Device* device, BenchPhase phase, gpu::TextureHandle[] textures) {
    (void)run_phase_worker_count(device, phase, textures, 1);
    double baseline_ns_per_op = 0;
    double single_worker_ns = 0;
    foreach (workers : BENCH_WORKER_COUNTS) {
        double ns_per_op = run_phase_worker_count(device, phase, textures, workers);
        if (workers == 1) { baseline_ns_per_op = ns_per_op; single_worker_ns = ns_per_op; }
        double scaling = baseline_ns_per_op / ns_per_op;
        ZString unit = phase == BenchPhase.SAMPLER ? "ns/op" : "ns/descriptor";
        io::printfn("phase %s workers=%d: %.1f %s, %.2fx scaling vs 1 thread",
            phase_name(phase), workers, ns_per_op, unit, scaling);
    }
    return single_worker_ns;
}

fn gpu::TextureHandle[BENCH_MAX_WORKERS] make_texture_pool(gpu::Device* device) {
    gpu::TextureHandle[BENCH_MAX_WORKERS] textures;
    foreach (&handle : textures) {
        gpu::TextureDesc desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .access    = { .graphics, .compute, .transfer },
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled, .storage },
        };
        *handle = gpu::create_texture(device, &desc)!!;
    }
    return textures;
}

fn void run_bench_suite(gpu::RuntimeDesc* desc, ZString label) {
    gpu::Device device = create_bench_device(desc)!!;
    defer destroy_bench_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::TextureHandle[BENCH_MAX_WORKERS] textures = make_texture_pool(&device);

    double single_ns = run_phase(&device, BenchPhase.TEXTURE_SINGLE, textures[..]);
    double batched_ns = run_phase(&device, BenchPhase.TEXTURE_BATCHED, textures[..]);
    run_phase(&device, BenchPhase.SAMPLER, textures[..]);
    io::printfn("%s: batch/single ratio (1 worker) = %.2fx", label, single_ns / batched_ns);

    foreach (handle : textures) (void)gpu::destroy_texture(&device, handle);
    uint leaks = vk::report_resource_leaks(state);
    io::printfn("%s: leaks=%d", label, leaks);
}

fn int main() {
    io::printfn("iterations=%d/worker units=ns/descriptor,ns/op", BENCH_ITERS_PER_WORKER);
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .application_name = "gpu_descriptor_churn_bench",
    };
    run_bench_suite(&desc, "automatic heap");
    return 0;
}
