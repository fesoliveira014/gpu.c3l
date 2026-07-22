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
const uint BENCH_LIFECYCLE_ITERS = 64;
const uint BENCH_SWAPCHAIN_CHECK_ITERS = 10_000;
const uint[3] BENCH_DESCRIPTOR_HIGH_WATER = { 16, 4_096, 65_536 };
const uint[4] BENCH_SAMPLER_OCCUPANCY = { 8, 64, 1_024, 65_536 };

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
    gpu::SamplerIndex index = gpu::intern_sampler(device, &desc)!!;
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

fn gpu::TextureHandle make_bench_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .access = { .graphics, .compute, .transfer },
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void run_texture_destroy_high_water(
    gpu::Device* device,
    vk::VkDeviceState* state,
    uint high_water,
) {
    gpu::TextureHandle[BENCH_LIFECYCLE_ITERS] textures;
    foreach (&texture : textures) *texture = make_bench_texture(device);
    uint texture_next_new_before =
        state.strict_heap.descriptors.texture_next_new;
    defer state.strict_heap.descriptors.texture_next_new =
        texture_next_new_before;
    state.strict_heap.descriptors.texture_next_new = high_water;
    vk::reset_texture_view_accounting(state);

    std::time::Clock start = std::time::clock::now();
    foreach (texture : textures) gpu::destroy_texture(device, texture)!!;
    long elapsed_ns = (long)start.to_now();

    vk::TextureViewAccountingCounters counters =
        vk::texture_view_accounting(state);
    assert(counters.cell_lookups == 0,
        "texture destruction must not inspect descriptor cells");
    assert(counters.ownership_work_units == BENCH_LIFECYCLE_ITERS,
        "texture destruction must charge one ownership decision per texture");
    io::printfn(
        "texture destroy highwater=%d: %.1f ns/destroy work_units=%d",
        high_water,
        (double)elapsed_ns / BENCH_LIFECYCLE_ITERS,
        counters.ownership_work_units,
    );
}

fn void run_swapchain_ownership_high_water(
    vk::VkDeviceState* state,
    gpu::TextureHandle[] textures,
    uint high_water,
) {
    vk::SwapchainSlot slot = { .image_count = (uint)textures.len };
    foreach (i, texture : textures) slot.textures[i] = texture;
    uint texture_next_new_before =
        state.strict_heap.descriptors.texture_next_new;
    defer state.strict_heap.descriptors.texture_next_new =
        texture_next_new_before;
    state.strict_heap.descriptors.texture_next_new = high_water;
    vk::reset_texture_view_accounting(state);

    long elapsed_ns;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        std::time::Clock start = std::time::clock::now();
        for (uint i = 0; i < BENCH_SWAPCHAIN_CHECK_ITERS; i++) {
            assert(!vk::swapchain_images_referenced_locked(state, &slot)!!,
                "unreferenced benchmark textures must remain releasable");
        }
        elapsed_ns = (long)start.to_now();
    }

    vk::TextureViewAccountingCounters counters =
        vk::texture_view_accounting(state);
    ulong expected_work =
        (ulong)BENCH_SWAPCHAIN_CHECK_ITERS * (ulong)textures.len;
    assert(counters.ownership_work_units == expected_work,
        "swapchain ownership work must equal checks times wrapped images");
    io::printfn(
        "swapchain ownership highwater=%d images=%d: %.1f ns/check work_units=%d",
        high_water,
        textures.len,
        (double)elapsed_ns / BENCH_SWAPCHAIN_CHECK_ITERS,
        counters.ownership_work_units,
    );
}

fn void run_sampler_lookup_probe(
    vk::VkDeviceState* state,
    uint occupancy,
) {
    vk::SamplerLookupProbeResult result =
        vk::probe_sampler_lookup(state, occupancy);
    assert(result.occupancy == occupancy
            && result.bucket_count >= occupancy * 2
            && (result.bucket_count & (result.bucket_count - 1)) == 0
            && result.probes >= 1
            && result.probes <= 8,
        "sampler lookup probe must report bounded indexed work");
    io::printfn(
        "sampler lookup occupancy=%d bucket_count=%d probes=%d elapsed_ns=%d",
        result.occupancy,
        result.bucket_count,
        result.probes,
        result.elapsed_ns,
    );
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

    foreach (occupancy : BENCH_SAMPLER_OCCUPANCY) {
        run_sampler_lookup_probe(state, occupancy);
    }

    foreach (high_water : BENCH_DESCRIPTOR_HIGH_WATER) {
        run_texture_destroy_high_water(&device, state, high_water);
        run_swapchain_ownership_high_water(
            state,
            textures[..],
            high_water,
        );
    }

    foreach (handle : textures) (void)gpu::destroy_texture(&device, handle);
    uint leaks = vk::report_resource_leaks(state);
    io::printfn("%s: leaks=%d", label, leaks);
}

fn int main() {
    io::printfn("iterations=%d/worker units=ns/descriptor,ns/op", BENCH_ITERS_PER_WORKER);
    gpu::RuntimeDesc desc = {
        .backend               = gpu::BackendKind.VULKAN,
        .texture_heap_capacity = 65_536,
        .application_name      = "gpu_descriptor_churn_bench",
    };
    run_bench_suite(&desc, "automatic heap");
    return 0;
}
