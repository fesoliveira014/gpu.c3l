module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::thread;
import std::time;
import std::atomic;
import std::atomic::types;
import std::core::mem::allocators;

// Advisory reset timing. The recording case includes the frame wait.

const uint BENCH_WORKERS = 15;
const uint BENCH_IDLE_FRAMES = 2_000;
const uint BENCH_RECORDING_FRAMES = 2_000;
const usz BENCH_FILL_STRIDE = 64;
const usz BENCH_WORKER_POOL_SIZE = 64 * 1024;

struct ResetWorkerArgs {
    gpu::Device*               device;
    gpu::BufferHandle          target;
    uint                       index;
    Atomic{uint}*              epoch;
    Atomic{uint}*              ready;
    Atomic{uint}*              done;
    Atomic{bool}*              stop;
    gpu::ExecutableCommandList executable;
}

fn int reset_worker(void* raw_args) {
    ResetWorkerArgs* args = (ResetWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, BENCH_WORKER_POOL_SIZE) {
        uint observed_epoch = 0;
        (void)args.ready.add(1, AtomicOrdering.RELEASE);
        while (!args.stop.load(AtomicOrdering.ACQUIRE)) {
            uint epoch = args.epoch.load(AtomicOrdering.ACQUIRE);
            if (epoch == observed_epoch) continue;
            observed_epoch = epoch;

            gpu::CommandList commands = gpu::begin_commands(
                gpu::get_queue(args.device, gpu::QueueKind.GRAPHICS)!!,
            )!!;
            gpu::cmd_fill_buffer(
                commands: &commands,
                buffer:   args.target,
                offset:   args.index * BENCH_FILL_STRIDE,
                size:     BENCH_FILL_STRIDE,
                value:    args.index,
            )!!;
            args.executable = gpu::end_commands(&commands)!!;
            (void)args.done.add(1, AtomicOrdering.RELEASE);
        }
    };
    return 0;
}

fn void? record_and_submit_all(
    gpu::Device* device,
    ResetWorkerArgs[BENCH_WORKERS]* args,
    Atomic{uint}* epoch,
    Atomic{uint}* done,
) {
    done.store(0, AtomicOrdering.RELEASE);
    (void)epoch.add(1, AtomicOrdering.RELEASE);
    while (done.load(AtomicOrdering.ACQUIRE) < BENCH_WORKERS) {}

    gpu::ExecutableCommandList[BENCH_WORKERS] lists;
    for (uint i = 0; i < BENCH_WORKERS; i++) {
        lists[i] = (*args)[i].executable;
    }
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    (void)submit_bench_queue(device, gpu::QueueKind.GRAPHICS, &submit)!;
}

fn int main() {
    io::printfn(
        "iterations=idle=%d;recording=%d units=ns/begin_frame",
        BENCH_IDLE_FRAMES,
        BENCH_RECORDING_FRAMES,
    );
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .frames_in_flight  = 2,
        .application_name  = "gpu_context_reset_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_bench_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = BENCH_WORKERS * BENCH_FILL_STRIDE,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle target = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, target)!!;

    Atomic{uint} epoch;
    Atomic{uint} ready;
    Atomic{uint} done;
    Atomic{bool} stop;
    ResetWorkerArgs[BENCH_WORKERS] args;
    Thread[BENCH_WORKERS] workers;
    for (uint i = 0; i < BENCH_WORKERS; i++) {
        args[i] = {
            .device = &device,
            .target = target,
            .index  = i,
            .epoch  = &epoch,
            .ready  = &ready,
            .done   = &done,
            .stop   = &stop,
        };
        workers[i].create(&reset_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < BENCH_WORKERS) {}
    defer {
        stop.store(true, AtomicOrdering.RELEASE);
        foreach (&worker : workers) (void)worker.join();
    }

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;
    frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;

    long idle_elapsed_ns = 0;
    uint idle_reset_total = 0;
    uint idle_skipped_total = 0;
    for (uint i = 0; i < BENCH_IDLE_FRAMES; i++) {
        std::time::Clock mark = std::time::clock::now();
        frame_token = gpu::begin_frame(&device)!!;
        idle_elapsed_ns += (long)mark.to_now();
        vk::PoolResetStats stats = vk::pool_reset_stats(state);
        idle_reset_total += stats.reset;
        idle_skipped_total += stats.skipped;
        gpu::end_frame(&frame_token)!!;
    }
    double idle_ns_per_frame = (double)idle_elapsed_ns / (double)BENCH_IDLE_FRAMES;
    io::printfn(
        "phase 1 (idle, %d frames, %d idle worker pools): %.1f ns/begin_frame, resets=%d skips=%d",
        BENCH_IDLE_FRAMES,
        BENCH_WORKERS,
        idle_ns_per_frame,
        idle_reset_total,
        idle_skipped_total,
    );

    // Dirty every worker pool for the slot where the timed loop starts.
    frame_token = gpu::begin_frame(&device)!!;
    record_and_submit_all(&device, &args, &epoch, &done)!!;
    gpu::end_frame(&frame_token)!!;

    long rec_elapsed_ns = 0;
    uint rec_reset_total = 0;
    uint rec_skipped_total = 0;
    for (uint i = 0; i < BENCH_RECORDING_FRAMES; i++) {
        std::time::Clock mark = std::time::clock::now();
        frame_token = gpu::begin_frame(&device)!!;
        rec_elapsed_ns += (long)mark.to_now();
        vk::PoolResetStats stats = vk::pool_reset_stats(state);
        rec_reset_total += stats.reset;
        rec_skipped_total += stats.skipped;

        record_and_submit_all(&device, &args, &epoch, &done)!!;
        gpu::end_frame(&frame_token)!!;
    }
    double rec_ns_per_frame = (double)rec_elapsed_ns / (double)BENCH_RECORDING_FRAMES;
    io::printfn(
        "phase 2 (all-recording, %d frames, %d worker pools): %.1f ns/begin_frame, resets=%d skips=%d",
        BENCH_RECORDING_FRAMES,
        BENCH_WORKERS,
        rec_ns_per_frame,
        rec_reset_total,
        rec_skipped_total,
    );

    wait_bench_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    return 0;
}