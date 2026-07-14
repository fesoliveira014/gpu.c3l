module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::time;

// Advisory, not gating: ns/end_frame + submit accounting across four
// queue-topology phases; the printed topology line flags lavapipe's collapse
// to the graphics-only path.

const uint BENCH_FRAME_COUNT = 2_000;

struct BenchPhase {
    String           name;
    gpu::QueueKind[] kinds;
}

fn void run_phase(gpu::Device* device, BenchPhase* phase) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
    vk::SubmitStats before = vk::submit_stats(state);
    std::time::Clock start = std::time::clock::now();

    for (uint frame = 0; frame < BENCH_FRAME_COUNT; frame++) {
        gpu::FrameToken frame_token = gpu::begin_frame(device)!!;
        foreach (kind : phase.kinds) {
            gpu::CommandList cmd = gpu::begin_commands(device, kind)!!;
            gpu::end_commands(&cmd)!!;
            gpu::CommandList[1] one = { cmd };
            gpu::SubmitDesc submit = { .command_lists = one[..] };
            gpu::submit(device, &submit)!!;
        }
        gpu::end_frame(&frame_token)!!;
    }

    long elapsed_ns = (long)start.to_now();
    vk::SubmitStats after = vk::submit_stats(state);
    double ns_per_end_frame = (double)elapsed_ns / (double)BENCH_FRAME_COUNT;
    double queue_submits_per_frame = (double)(after.queue_submits - before.queue_submits) / (double)BENCH_FRAME_COUNT;
    double empty_submits_per_frame = (double)(after.empty_signal_submits - before.empty_signal_submits) / (double)BENCH_FRAME_COUNT;
    io::printfn(
        "%s: %.1f ns/end_frame, %.3f queue submits/frame, %.3f empty submits/frame",
        phase.name, ns_per_end_frame, queue_submits_per_frame, empty_submits_per_frame,
    );
}

fn int main() {
    io::printfn("iterations=%d/phase units=ns/end_frame,submits/frame", BENCH_FRAME_COUNT);
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .frames_in_flight  = 2,
        .application_name  = "gpu_frame_signal_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    io::printfn("queue topology: async_compute=%s", gpu::get_device_caps(&device)!!.async_compute ? "true" : "false");
    if (!gpu::get_device_caps(&device)!!.async_compute) {
        io::printfn("distinct compute/transfer queues unavailable on this adapter: every phase below aliases onto the graphics queue, same as phase 1");
    }

    gpu::QueueKind[1] graphics_only     = { gpu::QueueKind.GRAPHICS };
    gpu::QueueKind[2] graphics_compute  = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE };
    gpu::QueueKind[2] graphics_transfer = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.TRANSFER };
    gpu::QueueKind[3] all_three         = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };

    BenchPhase[4] phases = {
        { .name = "phase 1 (graphics only)",                 .kinds = graphics_only[..] },
        { .name = "phase 2 (graphics + compute)",             .kinds = graphics_compute[..] },
        { .name = "phase 3 (graphics + transfer)",            .kinds = graphics_transfer[..] },
        { .name = "phase 4 (graphics + compute + transfer)",  .kinds = all_three[..] },
    };
    foreach (&phase : phases) run_phase(&device, phase);

    return 0;
}
