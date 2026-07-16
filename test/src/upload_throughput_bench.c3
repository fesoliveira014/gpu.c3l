module gpu_bench;

import gpu @public;
import gpu::vk @public;
import vk;
import std::io;
import std::thread;
import std::time;
import std::core::mem;
import std::core::mem::allocators;

// Advisory, not gating: Phase A observes parallel Tier-C recording under
// payload and staging-arena pressure. Phase B observes blocking-helper
// throughput across worker counts. Results describe the selected adapter and
// queue topology; timing and scaling are observations, never verdicts.

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

struct UploadBenchObservation {
    double uploads_per_second;
    ulong  dedicated_fallback_count;
    usz    staging_arena_size;
}

struct UploadPhaseAArgs {
    gpu::Device*                device;
    gpu::RecordingContextHandle ctx;
    gpu::BufferHandle           dst;
    usz                         payload_size;
    uint                        iters;
    gpu::CommandList            list;
}

struct UploadPhaseBArgs {
    gpu::Device*      device;
    gpu::BufferHandle dst;
    uint              iters;
}

fn gpu::Device make_bench_device(usz staging_size) {
    gpu::DeviceDesc desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .frames_in_flight   = 2,
        .staging_arena_size = staging_size,
        .application_name   = "gpu_upload_throughput_bench",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn void print_upload_bench_context(gpu::vk::VkDeviceState* state) {
    vk::PhysicalDeviceDriverProperties driver = vk::physical_device_driver_properties();
    vk::PhysicalDeviceProperties2 properties = vk::physical_device_properties2().set_next(&driver);
    vk::get_physical_device_properties2(state.physical_device, &properties);

    ZString device_name = (ZString)&properties.properties.device_name[0];
    ZString driver_name = (ZString)&driver.driver_name[0];
    bool transfer_distinct = state.transfer_queue != state.graphics_queue;

    io::printfn(
        "adapter: name=\"%s\" type=%s driver=\"%s\" driver_id=%d",
        device_name,
        upload_device_type_name(properties.properties.device_type),
        driver_name,
        (uint)driver.driver_id,
    );
    io::printfn("heap: mode=%s", upload_heap_mode_name(state.strict_heap.mode));
    io::printfn(
        "queues: graphics=%d:0 compute=%d:%d compute_distinct=%s transfer=%d:0 transfer_distinct=%s",
        state.graphics_family,
        state.compute_family,
        state.compute_queue_index,
        state.compute_queue_distinct ? "true" : "false",
        state.transfer_family,
        transfer_distinct ? "true" : "false",
    );
    io::printfn(
        "arenas: frame_bytes=%d persistent_bytes=%d staging_bytes=%d readback_bytes=%d",
        state.frame_arenas[0].size,
        state.persistent_arena.size,
        state.staging_arena.size,
        state.readback_arena.size,
    );
    if (upload_is_lavapipe_driver(driver.driver_id)) {
        io::printfn("note: Mesa llvmpipe/lavapipe selected; measurements describe this software adapter only");
    }
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
        gpu::end_commands(&args.list)!!;
    };
    return 0;
}

<* Wall-clock uploads/s and fallback-counter delta for one thread-count x
   payload x arena-size combination: each wave has every thread record in
   parallel into its own context, then the main thread submits every list
   together, waits, and closes the frame — the boundary that actually
   retires FRAME-tagged arena ranges and dedicated buffers alike. *>
fn UploadBenchObservation run_phase_a_combo(
    uint threads,
    usz payload_size,
    usz arena_size,
    bool* context_printed
) {
    gpu::Device device = make_bench_device(arena_size);
    defer destroy_bench_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    if (!*context_printed) {
        print_upload_bench_context(state);
        *context_printed = true;
    }

    gpu::RecordingContextHandle[UPLOAD_BENCH_MAX_THREADS] contexts;
    gpu::BufferHandle[UPLOAD_BENCH_MAX_THREADS] dsts;
    for (uint i = 0; i < threads; i++) {
        contexts[i] = gpu::create_recording_context(&device)!!;
        gpu::BufferDesc desc = {
            .size = payload_size, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE, .access = { .graphics, .compute, .transfer },
        };
        dsts[i] = gpu::create_buffer(&device, &desc)!!;
    }

    ulong fallbacks_before = gpu::vk::dedicated_fallback_count(state);
    std::time::Clock start = std::time::clock::now();

    for (uint w = 0; w < UPLOAD_BENCH_WAVES; w++) {
        gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
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
        wait_bench_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::end_frame(&frame_token)!!;
    }
    long elapsed_ns = (long)start.to_now();

    ulong fallbacks_after = gpu::vk::dedicated_fallback_count(state);
    uint total_ops = threads * UPLOAD_BENCH_WAVES * UPLOAD_BENCH_WAVE_SIZE;
    double uploads_per_sec = (double)total_ops / ((double)elapsed_ns / 1.0e9);
    ulong fallback_count = fallbacks_after - fallbacks_before;

    for (uint i = 0; i < threads; i++) {
        (void)gpu::destroy_buffer(&device, dsts[i]);
        (void)gpu::destroy_recording_context(&device, contexts[i]);
    }
    return {
        .uploads_per_second       = uploads_per_sec,
        .dedicated_fallback_count = fallback_count,
        .staging_arena_size       = state.staging_arena.size,
    };
}

fn void run_phase_a() {
    io::printfn("-- phase A hypothesis: observe parallel Tier-C recording under staging pressure --");
    bool context_printed = false;
    foreach (arena_size : UPLOAD_BENCH_ARENA_SIZES) {
        foreach (payload_size : UPLOAD_BENCH_PAYLOAD_SIZES) {
            double one_worker_rate = 0.0;
            foreach (threads : UPLOAD_BENCH_THREAD_COUNTS) {
                UploadBenchObservation observation = run_phase_a_combo(
                    threads:         threads,
                    payload_size:    payload_size,
                    arena_size:      arena_size,
                    context_printed: &context_printed,
                );
                if (threads == 1) one_worker_rate = observation.uploads_per_second;
                io::printfn(
                    "phase=A workers=%d payload_kib=%d staging_bytes=%d uploads_per_sec=%.0f scaling_vs_1_worker=%.2f dedicated_fallback_count=%d dedicated_fallback_used=%s",
                    threads,
                    payload_size / 1024,
                    observation.staging_arena_size,
                    observation.uploads_per_second,
                    upload_scaling_vs_one(observation.uploads_per_second, one_worker_rate),
                    observation.dedicated_fallback_count,
                    upload_dedicated_fallback_used(observation.dedicated_fallback_count) ? "true" : "false",
                );
            }
        }
    }
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

<* Observe blocking upload_buffer_data across worker counts; serialization
   and queue costs vary by adapter and topology. *>
fn void run_phase_b() {
    io::printfn("-- phase B hypothesis: observe blocking-helper throughput; serialization and queue costs vary by adapter/topology --");
    gpu::Device device = make_bench_device(0);
    defer destroy_bench_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::BufferHandle[UPLOAD_BENCH_MAX_THREADS] dsts;
    for (uint i = 0; i < UPLOAD_BENCH_MAX_THREADS; i++) {
        gpu::BufferDesc desc = {
            .size = UPLOAD_BENCH_HELPER_PAYLOAD, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE, .access = { .graphics, .compute, .transfer },
        };
        dsts[i] = gpu::create_buffer(&device, &desc)!!;
    }
    defer {
        for (uint i = 0; i < UPLOAD_BENCH_MAX_THREADS; i++) (void)gpu::destroy_buffer(&device, dsts[i]);
    }

    double one_worker_rate = 0.0;
    foreach (threads : UPLOAD_BENCH_THREAD_COUNTS) {
        UploadPhaseBArgs[UPLOAD_BENCH_MAX_THREADS] args;
        Thread[UPLOAD_BENCH_MAX_THREADS] threads_arr;
        for (uint i = 0; i < threads; i++) {
            args[i] = { .device = &device, .dst = dsts[i], .iters = UPLOAD_BENCH_HELPER_ITERS_PER_THREAD };
        }
        ulong fallbacks_before = gpu::vk::dedicated_fallback_count(state);
        std::time::Clock start = std::time::clock::now();
        for (uint i = 0; i < threads; i++) threads_arr[i].create(&upload_phase_b_worker, &args[i])!!;
        for (uint i = 0; i < threads; i++) threads_arr[i].join()!!;
        long elapsed_ns = (long)start.to_now();

        uint total_ops = threads * UPLOAD_BENCH_HELPER_ITERS_PER_THREAD;
        double uploads_per_sec = (double)total_ops / ((double)elapsed_ns / 1.0e9);
        if (threads == 1) one_worker_rate = uploads_per_sec;
        ulong fallback_count = gpu::vk::dedicated_fallback_count(state) - fallbacks_before;
        io::printfn(
            "phase=B workers=%d payload_kib=%d staging_bytes=%d uploads_per_sec=%.0f scaling_vs_1_worker=%.2f dedicated_fallback_count=%d dedicated_fallback_used=%s",
            threads,
            UPLOAD_BENCH_HELPER_PAYLOAD / 1024,
            state.staging_arena.size,
            uploads_per_sec,
            upload_scaling_vs_one(uploads_per_sec, one_worker_rate),
            fallback_count,
            upload_dedicated_fallback_used(fallback_count) ? "true" : "false",
        );
    }
}

fn int main() {
    io::printfn("iterations=%d/worker/combination units=uploads/s", UPLOAD_BENCH_HELPER_ITERS_PER_THREAD);
    io::printfn(
        "config: workers=[%d,%d,%d] phase_a_payload_kib=[%d,%d,%d] phase_a_staging_mib=[%d,%d] waves=%d uploads_per_wave=%d phase_b_payload_kib=%d phase_b_iters_per_worker=%d",
        UPLOAD_BENCH_THREAD_COUNTS[0],
        UPLOAD_BENCH_THREAD_COUNTS[1],
        UPLOAD_BENCH_THREAD_COUNTS[2],
        UPLOAD_BENCH_PAYLOAD_SIZES[0] / 1024,
        UPLOAD_BENCH_PAYLOAD_SIZES[1] / 1024,
        UPLOAD_BENCH_PAYLOAD_SIZES[2] / 1024,
        UPLOAD_BENCH_ARENA_SIZES[0] >> 20,
        UPLOAD_BENCH_ARENA_SIZES[1] >> 20,
        UPLOAD_BENCH_WAVES,
        UPLOAD_BENCH_WAVE_SIZE,
        UPLOAD_BENCH_HELPER_PAYLOAD / 1024,
        UPLOAD_BENCH_HELPER_ITERS_PER_THREAD,
    );
    run_phase_a();
    run_phase_b();
    return 0;
}
