module gpu_bench;

import gpu @public;
import std::io;
import std::time;

const uint SUBMISSION_ITERATIONS = 256;
const uint COMPLETION_POLL_ITERATIONS = 100_000;
const uint DESTRUCTION_ITERATIONS = 300;
const uint LIFECYCLE_REPETITIONS = 5;
const uint MAX_SUBMIT_BATCH_SIZE = 1_024;
const uint SUBMIT_BATCH_ITERATIONS = 5;
const uint[5] SUBMIT_BATCH_SIZES = { 1, 8, 32, 128, 1_024 };

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

fn void record_submit_batch(
    gpu::CommandAllocator* allocator,
    gpu::ExecutableCommandList[] out_lists,
) {
    foreach (&out_list : out_lists) {
        *out_list = record_empty(allocator);
    }
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

fn double measure_submit_batch(
    gpu::CommandAllocator* allocator,
    gpu::Queue queue,
    uint batch_size,
) {
    long elapsed_ns = 0;
    for (uint i = 0; i < SUBMIT_BATCH_ITERATIONS; i++) {
        gpu::ExecutableCommandList[MAX_SUBMIT_BATCH_SIZE] lists;
        record_submit_batch(allocator, lists[:batch_size]);
        gpu::SubmitDesc submit = {
            .command_lists = lists[:batch_size],
        };

        std::time::Clock start = std::time::clock::now();
        gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
        elapsed_ns += (long)start.to_now();

        foreach (list : lists[:batch_size]) {
            assert(!list.is_valid(),
                "successful submission consumes every executable token");
        }
        gpu::wait_completion(completion)!!;
    }
    return (double)elapsed_ns / (double)SUBMIT_BATCH_ITERATIONS;
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
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn double measure_texture_destruction(gpu::Device* device) {
    gpu::TextureHandle[DESTRUCTION_ITERATIONS] textures;
    foreach (&texture : textures) {
        *texture = create_bench_texture(device);
    }

    std::time::Clock start = std::time::clock::now();
    foreach (texture : textures) {
        gpu::destroy_texture(device, texture)!!;
    }
    long elapsed_ns = (long)start.to_now();
    return (double)elapsed_ns / (double)DESTRUCTION_ITERATIONS;
}

fn int main() {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_lifecycle_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = MAX_SUBMIT_BATCH_SIZE,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    gpu::CompletionPoint warm_completion;
    (void)measure_submissions(&allocator, queue, &warm_completion);
    (void)measure_completed_polls(warm_completion);
    (void)measure_texture_destruction(&device);

    double[LIFECYCLE_REPETITIONS] submission_results;
    double[LIFECYCLE_REPETITIONS] poll_results;
    double[LIFECYCLE_REPETITIONS] destruction_results;
    gpu::CompletionPoint completion;
    for (uint i = 0; i < LIFECYCLE_REPETITIONS; i++) {
        submission_results[i] = measure_submissions(
            &allocator,
            queue,
            &completion,
        );
        poll_results[i] = measure_completed_polls(completion);
        destruction_results[i] = measure_texture_destruction(&device);
    }

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
    foreach (batch_size : SUBMIT_BATCH_SIZES) {
        double ns_per_submit = measure_submit_batch(
            &allocator,
            queue,
            batch_size,
        );
        io::printfn(
            "submit batch: size=%d iterations=%d %.1f ns/submit %.3f ns/list",
            batch_size,
            SUBMIT_BATCH_ITERATIONS,
            ns_per_submit,
            ns_per_submit / (double)batch_size,
        );
    }
    return 0;
}
