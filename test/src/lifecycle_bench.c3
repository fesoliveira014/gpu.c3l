module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::time;

const uint SUBMISSION_ITERATIONS = 256;
const uint COMPLETION_POLL_ITERATIONS = 100_000;
const uint DESTRUCTION_ITERATIONS = 300;
const uint LIFECYCLE_REPETITIONS = 5;

struct SubmissionFixture {
    gpu::ExecutableCommandList[1] commands;
}

fn double lifecycle_median(double[LIFECYCLE_REPETITIONS] values) {
    double[LIFECYCLE_REPETITIONS] sorted = values;
    for (uint i = 1; i < LIFECYCLE_REPETITIONS; i++) {
        double key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    return sorted[LIFECYCLE_REPETITIONS / 2];
}

fn gpu::ExecutableCommandList record_empty(gpu::Queue queue) {
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    return gpu::end_commands(&commands)!!;
}

fn double measure_submissions(
    gpu::Queue queue,
    gpu::CompletionPoint* last,
) {
    SubmissionFixture[SUBMISSION_ITERATIONS] fixtures;
    for (uint i = 0; i < SUBMISSION_ITERATIONS; i++) {
        fixtures[i].commands[0] = record_empty(queue);
    }

    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < SUBMISSION_ITERATIONS; i++) {
        gpu::SubmitDesc submit = {
            .command_lists = fixtures[i].commands[..],
        };
        *last = gpu::submit(queue, &submit)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::wait_completion(*last)!!;
    return (double)elapsed_ns / (double)SUBMISSION_ITERATIONS;
}

fn double measure_completed_polls(gpu::CompletionPoint point) {
    bool all_complete = true;
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMPLETION_POLL_ITERATIONS; i++) {
        all_complete = gpu::poll_completion(point)!! && all_complete;
    }
    long elapsed_ns = (long)start.to_now();
    assert(all_complete, "completed point remains complete");
    return (double)elapsed_ns / (double)COMPLETION_POLL_ITERATIONS;
}

fn gpu::TextureHandle create_bench_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn double measure_texture_destruction(
    gpu::Device* device,
    ulong* wait_calls,
    ulong* deferred_releases,
) {
    gpu::TextureHandle[DESTRUCTION_ITERATIONS] textures;
    foreach (&texture : textures) {
        *texture = create_bench_texture(device);
    }

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(*device);
    vk::SubmitStats submit_before = vk::submit_stats(state);
    vk::CommandBufferStats commands_before = vk::command_buffer_stats(state);
    std::time::Clock start = std::time::clock::now();
    foreach (texture : textures) {
        gpu::destroy_texture(device, texture)!!;
    }
    long elapsed_ns = (long)start.to_now();
    vk::SubmitStats submit_after = vk::submit_stats(state);
    vk::CommandBufferStats commands_after = vk::command_buffer_stats(state);
    *wait_calls += submit_after.completion_wait_calls
        - submit_before.completion_wait_calls;
    if (commands_after.pending > commands_before.pending) {
        *deferred_releases +=
            (ulong)(commands_after.pending - commands_before.pending);
    }
    return (double)elapsed_ns / (double)DESTRUCTION_ITERATIONS;
}

fn int main() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_lifecycle_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::CompletionPoint warm_completion;
    (void)measure_submissions(queue, &warm_completion);
    (void)measure_completed_polls(warm_completion);
    ulong warm_destruction_waits;
    ulong warm_deferred_releases;
    (void)measure_texture_destruction(
        &device,
        &warm_destruction_waits,
        &warm_deferred_releases,
    );

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::SubmitStats submit_before = vk::submit_stats(state);

    double[LIFECYCLE_REPETITIONS] submission_results;
    double[LIFECYCLE_REPETITIONS] poll_results;
    double[LIFECYCLE_REPETITIONS] destruction_results;
    gpu::CompletionPoint completion;
    ulong destruction_waits;
    ulong deferred_releases;
    for (uint i = 0; i < LIFECYCLE_REPETITIONS; i++) {
        submission_results[i] = measure_submissions(queue, &completion);
        poll_results[i] = measure_completed_polls(completion);
        destruction_results[i] = measure_texture_destruction(
            &device,
            &destruction_waits,
            &deferred_releases,
        );
    }
    vk::SubmitStats submit_after = vk::submit_stats(state);
    ulong point_publications = submit_after.completion_point_publications
        - submit_before.completion_point_publications;
    ulong point_allocations = submit_after.completion_point_allocations
        - submit_before.completion_point_allocations;
    assert(point_publications
        == (ulong)SUBMISSION_ITERATIONS * LIFECYCLE_REPETITIONS,
        "every measured submit publishes one completion point");
    assert(point_allocations == 0,
        "scalar completion publication performs no table allocation");
    assert(destruction_waits == 0,
        "immediate destruction performs no device wait");
    assert(deferred_releases == 0,
        "immediate destruction enqueues no deferred release");

    io::printfn(
        "submission: iterations=%d repetitions=%d median=%.1f ns/submit",
        SUBMISSION_ITERATIONS,
        LIFECYCLE_REPETITIONS,
        lifecycle_median(submission_results),
    );
    io::printfn(
        "completion poll: iterations=%d repetitions=%d median=%.1f ns/poll",
        COMPLETION_POLL_ITERATIONS,
        LIFECYCLE_REPETITIONS,
        lifecycle_median(poll_results),
    );
    io::printfn(
        "texture destroy: iterations=%d repetitions=%d median=%.1f ns/destroy",
        DESTRUCTION_ITERATIONS,
        LIFECYCLE_REPETITIONS,
        lifecycle_median(destruction_results),
    );
    io::printfn(
        "invariants: point_allocations=%d destruction_waits=%d deferred_releases=%d",
        point_allocations,
        destruction_waits,
        deferred_releases,
    );
    return 0;
}
