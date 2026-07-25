module gpu_bench;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::io;
import std::time;

const uint MAX_SUBMIT_BATCH_SIZE = 1_024;
const uint SUBMIT_BATCH_ITERATIONS = 5;
const uint[5] SUBMIT_BATCH_SIZES = { 1, 8, 32, 128, 1_024 };

fn void record_submit_batch(
    gpu::CommandAllocator* allocator,
    gpu::ExecutableCommandList[] out_lists,
) {
    foreach (&out_list : out_lists) {
        gpu::CommandList commands = gpu::begin_commands(allocator)!!;
        *out_list = gpu::end_commands(&commands)!!;
    }
}

fn void measure_submit_batch(
    gpu::CommandAllocator* allocator,
    gpu::Queue queue,
    vk::VkDeviceState* state,
    uint batch_size,
) {
    long elapsed_ns = 0;
    vk::SubmitWorkCounters work;
    ulong total_host_allocations;
    for (uint i = 0; i < SUBMIT_BATCH_ITERATIONS; i++) {
        gpu::ExecutableCommandList[MAX_SUBMIT_BATCH_SIZE] lists;
        record_submit_batch(allocator, lists[:batch_size]);
        vk::reset_submit_work_counters(state);
        gpu::SubmitDesc submit = {
            .command_lists = lists[:batch_size],
        };
        vk::RecordingHostAllocatorSnapshot allocator_before =
            vk::recording_host_allocator_snapshot(state);

        std::time::Clock start = std::time::clock::now();
        gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
        elapsed_ns += (long)start.to_now();

        work = vk::submit_work_counters(state);
        vk::RecordingHostAllocatorSnapshot allocator_after =
            vk::recording_host_allocator_snapshot(state);
        ulong host_allocations =
            allocator_after.allocations - allocator_before.allocations
            + allocator_after.resizes - allocator_before.resizes;
        total_host_allocations += host_allocations;
        assert(work.authoritative_token_resolutions == batch_size
                && work.duplicate_token_visits == batch_size
                && work.epoch_reset_cells == 0
                && work.command_mutex_acquisitions == 1
                && work.queue_submission_mutex_acquisitions == 1
                && work.rollback_mutex_acquisitions == 0
                && work.native_submissions == 1,
            "ordinary submit work must match the authoritative transaction");
        assert(host_allocations == 0,
            "warm submission must not allocate host storage");
        foreach (list : lists[:batch_size]) {
            assert(!list.is_valid(),
                "successful submission consumes every executable token");
        }
        gpu::wait_completion(completion)!!;
        assert(vk::queue_completion_state(state, queue.id)!!
                .submitted_count == 0,
            "completion wait drains the submitted-command list");
    }
    double ns_per_submit = (double)elapsed_ns / SUBMIT_BATCH_ITERATIONS;
    double ns_per_list = ns_per_submit / batch_size;
    io::printfn(
        "submit batch size=%d ns/submit=%.1f ns/list=%.3f "
        "resolutions=%d duplicate_visits=%d "
        "epoch_reset_cells=%d command_mutex=%d "
        "queue_submission_mutex=%d "
        "rollback_mutex=%d native_submissions=%d host_allocations=%d",
        batch_size,
        ns_per_submit,
        ns_per_list,
        work.authoritative_token_resolutions,
        work.duplicate_token_visits,
        work.epoch_reset_cells,
        work.command_mutex_acquisitions,
        work.queue_submission_mutex_acquisitions,
        work.rollback_mutex_acquisitions,
        work.native_submissions,
        total_host_allocations,
    );
}

fn int main() {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_submit_batch_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = MAX_SUBMIT_BATCH_SIZE,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;

    io::printfn(
        "iterations=%d batch_sizes=1,8,32,128,1024 units=ns/submit,ns/list",
        SUBMIT_BATCH_ITERATIONS,
    );
    foreach (batch_size : SUBMIT_BATCH_SIZES) {
        measure_submit_batch(
            &allocator,
            queue,
            state,
            batch_size,
        );
    }
    gpu::destroy_command_allocator(&allocator)!!;
    io::printfn("submit batch leaks=%d", vk::report_resource_leaks(state));
    return 0;
}
