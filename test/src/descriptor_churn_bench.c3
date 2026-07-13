module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::thread;
import std::time;
import std::os::env;
import std::core::mem::allocators;

// Advisory, not gating: texture-descriptor singles vs batched (N=16) vs
// sampler create/destroy round trips at 1/2/4 workers, off-frame so retires
// drain immediately. Reports ns/descriptor, scaling vs one thread, and the
// batch-vs-single ratio. DB-mode phases run only when
// GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1 (same lavapipe caveat as
// test_vk_descriptor_buffer.c3). Indicative on lavapipe.

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
        default:                         return "samplers";
    }
}

fn uint descriptors_per_op(BenchPhase phase) {
    return phase == BenchPhase.TEXTURE_BATCHED ? BENCH_BATCH_SIZE : 1;
}

fn void run_texture_single_op(gpu::Device* device, gpu::TextureHandle texture) {
    gpu::TextureIndex index = gpu::create_texture_descriptor(device, texture, null)!!;
    gpu::destroy_texture_descriptor(device, index)!!;
}

fn void run_texture_batched_op(gpu::Device* device, gpu::TextureHandle texture) {
    gpu::TextureDescriptorDesc[BENCH_BATCH_SIZE] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureIndex[BENCH_BATCH_SIZE] indices;
    gpu::create_texture_descriptors(device, descs[..], indices[..])!!;
    foreach (index : indices) gpu::destroy_texture_descriptor(device, index)!!;
}

fn void run_sampler_op(gpu::Device* device) {
    gpu::SamplerDesc desc = {};
    gpu::SamplerIndex index = gpu::create_sampler(device, &desc)!!;
    gpu::destroy_sampler(device, index)!!;
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

<* Wall-clock ns/descriptor across `workers` concurrent workers, each running
   BENCH_ITERS_PER_WORKER descriptor create+destroy round trips of `phase`
   (batched phase: BENCH_ITERS_PER_WORKER/BENCH_BATCH_SIZE calls of
   BENCH_BATCH_SIZE each) against its own pool texture. *>
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
        io::printfn("phase %s workers=%d: %.1f ns/descriptor, %.2fx scaling vs 1 thread",
            phase_name(phase), workers, ns_per_op, scaling);
    }
    return single_worker_ns;
}

fn gpu::TextureHandle[BENCH_MAX_WORKERS] make_texture_pool(gpu::Device* device) {
    gpu::TextureHandle[BENCH_MAX_WORKERS] textures;
    foreach (&handle : textures) {
        gpu::TextureDesc desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled, .storage },
        };
        *handle = gpu::create_texture(device, &desc)!!;
    }
    return textures;
}

fn void run_bench_suite(gpu::DeviceDesc* desc, ZString label) {
    gpu::Device device = gpu::create_device(desc)!!;
    defer gpu::destroy_device(&device)!!;
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

fn bool db_e2e_enabled() {
    if (try value = env::tget_var("GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E")) {
        return value == "1";
    }
    return false;
}

fn int main() {
    io::printfn("iterations=%d/worker units=ns/descriptor", BENCH_ITERS_PER_WORKER);
    gpu::DeviceDesc indexing_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_descriptor_churn_bench",
    };
    run_bench_suite(&indexing_desc, "indexing");

    if (db_e2e_enabled()) {
        gpu::DeviceDesc buffer_desc = {
            .backend              = gpu::BackendKind.VULKAN,
            .descriptor_heap_mode = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
            .application_name     = "gpu_descriptor_churn_bench_db",
        };
        run_bench_suite(&buffer_desc, "buffer");
    }

    return 0;
}
