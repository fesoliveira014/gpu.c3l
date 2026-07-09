module gpu_bench;

import gpu;
import gpu::vk;
import std::io;
import std::thread;
import std::time;
import std::core::mem;
import std::core::mem::allocators;

// Advisory, not gating: Phase A times parallel Tier-C cmd_upload_buffer
// across per-thread recording contexts at payload x staging-arena-size
// combinations, some of which force the dedicated-fallback path hoisted
// out of transfer_mutex. Phase B times the blocking upload_buffer_data
// helper across thread counts; expect a near-flat line — it is queue-bound
// (record-mutex + graphics-queue idle dominate), not lock-bound, so the hoist
// buys it nothing. Lavapipe caveats: host mallocs understate real
// dedicated-create cost, and its instant queue completion understates how
// queue-bound phase B really is on a discrete GPU.

const uint[3] UPLOAD_BENCH_THREAD_COUNTS = { 1, 2, 4 };
const usz[3] UPLOAD_BENCH_PAYLOAD_SIZES = { 4 * 1024, 256 * 1024, 4 * 1024 * 1024 };
const usz[2] UPLOAD_BENCH_ARENA_SIZES = { 8 << 20, 32 << 20 };
const uint UPLOAD_BENCH_MAX_THREADS = 4;
const usz UPLOAD_BENCH_POOL_SIZE = 4 * 1024 * 1024 + 256 * 1024;
// FRAME-tagged ranges (arena and dedicated-fallback alike) only retire at an
// end_frame boundary, so recording runs in waves with a submit + end_frame
// between them: at most WAVE_SIZE * max threads dedicated fallbacks can be
// pending at once, well under DEDICATED_RETIRE_CAP even when every call in
// the wave misses the arena outright.
const uint UPLOAD_BENCH_WAVE_SIZE = 4;
const uint UPLOAD_BENCH_WAVES = 10;

const usz UPLOAD_BENCH_HELPER_PAYLOAD = 256 * 1024;
const uint UPLOAD_BENCH_HELPER_ITERS_PER_THREAD = 40;

fn gpu::Device make_bench_device(usz staging_size) {
    gpu::DeviceDesc desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .frames_in_flight   = 2,
        .staging_arena_size = staging_size,
        .application_name   = "gpu_upload_throughput_bench",
    };
    return gpu::create_device(&desc)!!;
}

struct UploadPhaseAArgs {
    gpu::Device*                device;
    gpu::RecordingContextHandle ctx;
    gpu::BufferHandle           dst;
    usz                         payload_size;
    uint                        iters;
    gpu::CommandList            list;
}

fn int upload_phase_a_worker(void* arg) {
    UploadPhaseAArgs* args = (UploadPhaseAArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, UPLOAD_BENCH_POOL_SIZE) {
        char[] payload = mem::talloc_array(char, (sz)args.payload_size);
        foreach (&b : payload) *b = (char)0xAB;
        args.list = gpu::begin_commands(args.device, gpu::QueueKind.GRAPHICS, args.ctx)!!;
        for (uint i = 0; i < args.iters; i++) {
            gpu::cmd_upload_buffer(&args.list, args.dst, 0, payload)!!;
        }
        gpu::end_commands(args.device, &args.list)!!;
    };
    return 0;
}

<* Wall-clock uploads/s and fallback-counter delta for one thread-count x
   payload x arena-size combination: each wave has every thread record in
   parallel into its own context, then the main thread submits every list
   together, waits, and closes the frame — the boundary that actually
   retires FRAME-tagged arena ranges and dedicated buffers alike. *>
fn void run_phase_a_combo(uint threads, usz payload_size, usz arena_size) {
    gpu::Device device = make_bench_device(arena_size);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::RecordingContextHandle[UPLOAD_BENCH_MAX_THREADS] contexts;
    gpu::BufferHandle[UPLOAD_BENCH_MAX_THREADS] dsts;
    for (uint i = 0; i < threads; i++) {
        contexts[i] = gpu::create_recording_context(&device)!!;
        gpu::BufferDesc desc = {
            .size = payload_size, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        dsts[i] = gpu::create_buffer(&device, &desc)!!;
    }

    ulong fallbacks_before = vk::dedicated_fallback_count(state);
    std::time::Clock start = std::time::clock::now();

    for (uint w = 0; w < UPLOAD_BENCH_WAVES; w++) {
        gpu::begin_frame(&device)!!;
        UploadPhaseAArgs[UPLOAD_BENCH_MAX_THREADS] args;
        Thread[UPLOAD_BENCH_MAX_THREADS] threads_arr;
        for (uint i = 0; i < threads; i++) {
            args[i] = { .device = &device, .ctx = contexts[i], .dst = dsts[i], .payload_size = payload_size, .iters = UPLOAD_BENCH_WAVE_SIZE };
        }
        for (uint i = 0; i < threads; i++) threads_arr[i].create(&upload_phase_a_worker, &args[i])!!;
        for (uint i = 0; i < threads; i++) threads_arr[i].join()!!;

        gpu::CommandList[UPLOAD_BENCH_MAX_THREADS] lists;
        for (uint i = 0; i < threads; i++) lists[i] = args[i].list;
        gpu::SubmitDesc submit = { .command_lists = lists[:threads] };
        gpu::submit(&device, &submit)!!;
        gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::end_frame(&device)!!;
    }
    long elapsed_ns = (long)start.to_now();

    ulong fallbacks_after = vk::dedicated_fallback_count(state);
    uint total_ops = threads * UPLOAD_BENCH_WAVES * UPLOAD_BENCH_WAVE_SIZE;
    double uploads_per_sec = (double)total_ops / ((double)elapsed_ns / 1.0e9);
    io::printfn(
        "phase A threads=%d payload=%dK arena=%dK: %.0f uploads/s, fallbacks=%d",
        threads, payload_size / 1024, arena_size / 1024, uploads_per_sec, fallbacks_after - fallbacks_before,
    );

    for (uint i = 0; i < threads; i++) {
        (void)gpu::destroy_buffer(&device, dsts[i]);
        (void)gpu::destroy_recording_context(&device, contexts[i]);
    }
}

fn void run_phase_a() {
    io::printfn("-- phase A: parallel Tier-C cmd_upload_buffer across contexts --");
    foreach (arena_size : UPLOAD_BENCH_ARENA_SIZES) {
        foreach (payload_size : UPLOAD_BENCH_PAYLOAD_SIZES) {
            foreach (threads : UPLOAD_BENCH_THREAD_COUNTS) {
                run_phase_a_combo(threads, payload_size, arena_size);
            }
        }
    }
}

struct UploadPhaseBArgs {
    gpu::Device*       device;
    gpu::BufferHandle  dst;
    uint               iters;
}

fn int upload_phase_b_worker(void* arg) {
    UploadPhaseBArgs* args = (UploadPhaseBArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, UPLOAD_BENCH_POOL_SIZE) {
        char[] payload = mem::talloc_array(char, (sz)UPLOAD_BENCH_HELPER_PAYLOAD);
        foreach (&b : payload) *b = (char)0xCD;
        for (uint i = 0; i < args.iters; i++) {
            gpu::upload_buffer_data(
                device:      args.device,
                dst:         args.dst,
                offset:      0,
                data:        payload,
                next_stage:  gpu::Stage.TRANSFER,
                next_hazard: gpu::Hazard.TRANSFER_READ,
            )!!;
        }
    };
    return 0;
}

<* Blocking upload_buffer_data at N threads. On real drivers the queue idle
   dominates and the line flattens; lavapipe's instant queue lets it scale,
   so read this phase with the printed caveat. *>
fn void run_phase_b() {
    io::printfn("-- phase B: blocking upload_buffer_data (queue-bound; expect a near-flat uploads/s line across thread counts) --");
    gpu::Device device = make_bench_device(0);
    defer gpu::destroy_device(&device)!!;

    gpu::BufferHandle[UPLOAD_BENCH_MAX_THREADS] dsts;
    for (uint i = 0; i < UPLOAD_BENCH_MAX_THREADS; i++) {
        gpu::BufferDesc desc = {
            .size = UPLOAD_BENCH_HELPER_PAYLOAD, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        dsts[i] = gpu::create_buffer(&device, &desc)!!;
    }
    defer {
        for (uint i = 0; i < UPLOAD_BENCH_MAX_THREADS; i++) (void)gpu::destroy_buffer(&device, dsts[i]);
    }

    foreach (threads : UPLOAD_BENCH_THREAD_COUNTS) {
        UploadPhaseBArgs[UPLOAD_BENCH_MAX_THREADS] args;
        Thread[UPLOAD_BENCH_MAX_THREADS] threads_arr;
        for (uint i = 0; i < threads; i++) {
            args[i] = { .device = &device, .dst = dsts[i], .iters = UPLOAD_BENCH_HELPER_ITERS_PER_THREAD };
        }
        std::time::Clock start = std::time::clock::now();
        for (uint i = 0; i < threads; i++) threads_arr[i].create(&upload_phase_b_worker, &args[i])!!;
        for (uint i = 0; i < threads; i++) threads_arr[i].join()!!;
        long elapsed_ns = (long)start.to_now();

        uint total_ops = threads * UPLOAD_BENCH_HELPER_ITERS_PER_THREAD;
        double uploads_per_sec = (double)total_ops / ((double)elapsed_ns / 1.0e9);
        io::printfn("phase B threads=%d payload=%dK: %.0f uploads/s",
            threads, UPLOAD_BENCH_HELPER_PAYLOAD / 1024, uploads_per_sec);
    }
}

fn int main() {
    io::printfn("gpu_upload_throughput_bench: lavapipe caveats -- host mallocs understate dedicated-create cost; instant queue completion understates how queue-bound phase B really is");
    run_phase_a();
    run_phase_b();
    return 0;
}
