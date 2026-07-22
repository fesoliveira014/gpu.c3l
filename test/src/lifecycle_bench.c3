module gpu_bench;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
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

fn gpu::ExecutableCommandList record_empty(gpu::CommandAllocator* allocator) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    return gpu::end_commands(&commands)!!;
}

fn double measure_submissions(
    gpu::CommandAllocator* allocator,
    gpu::Queue queue,
    gpu::CompletionPoint* last,
) {
    SubmissionFixture[SUBMISSION_ITERATIONS] fixtures;
    for (uint i = 0; i < SUBMISSION_ITERATIONS; i++) {
        fixtures[i].commands[0] = record_empty(allocator);
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

fn double measure_completed_polls(
    gpu::CompletionPoint point,
    vk::VkDeviceState* state,
    ulong* native_queries,
    ulong* retirement_locks,
) {
    vk::reset_completion_work_counters(state);
    bool all_complete = true;
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMPLETION_POLL_ITERATIONS; i++) {
        all_complete = gpu::poll_completion(point)!! && all_complete;
    }
    long elapsed_ns = (long)start.to_now();
    vk::CompletionWorkCounters work = vk::completion_work_counters(state);
    *native_queries += work.native_counter_queries;
    *retirement_locks += work.retirement_lock_acquisitions;
    assert(all_complete, "completed point remains complete");
    return (double)elapsed_ns / (double)COMPLETION_POLL_ITERATIONS;
}

fn gpu::TextureHandle create_bench_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
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
    ulong* completion_queries,
    ulong* completion_waits,
) {
    gpu::TextureHandle[DESTRUCTION_ITERATIONS] textures;
    foreach (&texture : textures) {
        *texture = create_bench_texture(device);
    }

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(*device);
    vk::reset_completion_work_counters(state);
    std::time::Clock start = std::time::clock::now();
    foreach (texture : textures) {
        gpu::destroy_texture(device, texture)!!;
    }
    long elapsed_ns = (long)start.to_now();
    vk::CompletionWorkCounters completion =
        vk::completion_work_counters(state);
    *completion_queries += completion.native_counter_queries;
    *completion_waits += completion.native_wait_calls;
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
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = SUBMISSION_ITERATIONS,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);

    gpu::CompletionPoint warm_completion;
    (void)measure_submissions(&allocator, queue, &warm_completion);
    ulong warm_native_queries;
    ulong warm_retirement_locks;
    (void)measure_completed_polls(
        point:            warm_completion,
        state:            state,
        native_queries:   &warm_native_queries,
        retirement_locks: &warm_retirement_locks,
    );
    ulong warm_destruction_queries;
    ulong warm_completion_waits;
    (void)measure_texture_destruction(
        device:            &device,
        completion_queries: &warm_destruction_queries,
        completion_waits:   &warm_completion_waits,
    );

    vk::SubmitStats submit_before = vk::submit_stats(state);

    double[LIFECYCLE_REPETITIONS] submission_results;
    double[LIFECYCLE_REPETITIONS] poll_results;
    double[LIFECYCLE_REPETITIONS] destruction_results;
    gpu::CompletionPoint completion;
    ulong destruction_queries;
    ulong destruction_completion_waits;
    ulong cached_poll_native_queries;
    ulong cached_poll_retirement_locks;
    for (uint i = 0; i < LIFECYCLE_REPETITIONS; i++) {
        submission_results[i] = measure_submissions(
            &allocator,
            queue,
            &completion,
        );
        poll_results[i] = measure_completed_polls(
            point:            completion,
            state:            state,
            native_queries:   &cached_poll_native_queries,
            retirement_locks: &cached_poll_retirement_locks,
        );
        destruction_results[i] = measure_texture_destruction(
            device:             &device,
            completion_queries: &destruction_queries,
            completion_waits:   &destruction_completion_waits,
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
    assert(destruction_queries == 0
        && destruction_completion_waits == 0,
        "immediate destruction performs no completion query or wait");
    assert(cached_poll_native_queries == 0,
        "cached completed polls perform no native counter queries");
    assert(cached_poll_retirement_locks == 0,
        "cached completed polls acquire no retirement lock");

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
        "invariants: point_allocations=%d destruction_queries=%d destruction_completion_waits=%d cached_poll_queries=%d retirement_locks=%d",
        point_allocations,
        destruction_queries,
        destruction_completion_waits,
        cached_poll_native_queries,
        cached_poll_retirement_locks,
    );
    return 0;
}
