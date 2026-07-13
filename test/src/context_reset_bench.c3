module gpu_bench;

import gpu::compat @public;
import gpu::compat::vk @public;
import std::io;
import std::time;

// Advisory, not gating: 15 user contexts; phase 1 idle frames, phase 2
// all-recording frames. Phase 2's ns/begin_frame is dominated by the real
// frame-timeline wait, not the reset loop — compare the reset/skip counters,
// not the wall time. Indicative on lavapipe.

const uint BENCH_USER_CONTEXTS = 15;
const uint BENCH_IDLE_FRAMES = 2_000;
const uint BENCH_RECORDING_FRAMES = 2_000;
const usz BENCH_FILL_STRIDE = 64;

<* One no-op fill per context, submitted together so every context's pool
   is dirtied exactly once per call. *>
fn void? record_and_submit_all(gpu::compat::Device* device, gpu::compat::RecordingContextHandle[BENCH_USER_CONTEXTS]* contexts, gpu::compat::BufferHandle target) {
    gpu::compat::CommandList[BENCH_USER_CONTEXTS] lists;
    for (uint c = 0; c < BENCH_USER_CONTEXTS; c++) {
        lists[c] = gpu::compat::begin_commands(device, gpu::compat::QueueKind.GRAPHICS, (*contexts)[c])!;
        gpu::compat::cmd_fill_buffer(&lists[c], target, c * BENCH_FILL_STRIDE, BENCH_FILL_STRIDE, c)!;
        gpu::compat::end_commands(&lists[c])!;
    }
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    return gpu::compat::submit(device, &submit);
}

fn int main() {
    io::printfn(
        "iterations=idle=%d;recording=%d units=ns/begin_frame",
        BENCH_IDLE_FRAMES,
        BENCH_RECORDING_FRAMES,
    );
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .frames_in_flight  = 2,
        .application_name  = "gpu_context_reset_bench",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::RecordingContextHandle[BENCH_USER_CONTEXTS] contexts;
    foreach (&ctx : contexts) *ctx = gpu::compat::create_recording_context(&device)!!;
    defer {
        foreach (ctx : contexts) (void)gpu::compat::destroy_recording_context(&device, ctx);
    }

    gpu::compat::BufferDesc buf_desc = {
        .size        = BENCH_USER_CONTEXTS * BENCH_FILL_STRIDE,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
    };
    gpu::compat::BufferHandle target = gpu::compat::create_buffer(&device, &buf_desc)!!;
    defer gpu::compat::destroy_buffer(&device, target)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    // Warm-up: let both frame slots pass once so phase timing starts clean.
    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;
    frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;

    long idle_elapsed_ns = 0;
    uint idle_reset_total = 0;
    uint idle_skipped_total = 0;
    for (uint i = 0; i < BENCH_IDLE_FRAMES; i++) {
        std::time::Clock mark = std::time::clock::now();
        frame_token = gpu::compat::begin_frame(&device)!!;
        idle_elapsed_ns += (long)mark.to_now();
        vk::PoolResetStats stats = vk::pool_reset_stats(state);
        idle_reset_total += stats.reset;
        idle_skipped_total += stats.skipped;
        gpu::compat::end_frame(&frame_token)!!;
    }
    double idle_ns_per_frame = (double)idle_elapsed_ns / (double)BENCH_IDLE_FRAMES;
    io::printfn("phase 1 (idle, %d frames, %d idle user contexts): %.1f ns/begin_frame, resets=%d skips=%d",
        BENCH_IDLE_FRAMES, BENCH_USER_CONTEXTS, idle_ns_per_frame, idle_reset_total, idle_skipped_total);

    // Prime every context's pool dirty for the slot the timed loop starts on.
    frame_token = gpu::compat::begin_frame(&device)!!;
    record_and_submit_all(&device, &contexts, target)!!;
    gpu::compat::end_frame(&frame_token)!!;

    long rec_elapsed_ns = 0;
    uint rec_reset_total = 0;
    uint rec_skipped_total = 0;
    for (uint i = 0; i < BENCH_RECORDING_FRAMES; i++) {
        std::time::Clock mark = std::time::clock::now();
        frame_token = gpu::compat::begin_frame(&device)!!;
        rec_elapsed_ns += (long)mark.to_now();
        vk::PoolResetStats stats = vk::pool_reset_stats(state);
        rec_reset_total += stats.reset;
        rec_skipped_total += stats.skipped;

        record_and_submit_all(&device, &contexts, target)!!;
        gpu::compat::end_frame(&frame_token)!!;
    }
    double rec_ns_per_frame = (double)rec_elapsed_ns / (double)BENCH_RECORDING_FRAMES;
    io::printfn("phase 2 (all-recording, %d frames, %d contexts): %.1f ns/begin_frame, resets=%d skips=%d",
        BENCH_RECORDING_FRAMES, BENCH_USER_CONTEXTS, rec_ns_per_frame, rec_reset_total, rec_skipped_total);

    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    return 0;
}
