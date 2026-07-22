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
    ulong total_token_visits = 0;
    ulong total_epoch_reset_cells = 0;
    for (uint i = 0; i < SUBMIT_BATCH_ITERATIONS; i++) {
        gpu::ExecutableCommandList[MAX_SUBMIT_BATCH_SIZE] lists;
        record_submit_batch(allocator, lists[:batch_size]);
        vk::reset_submit_duplicate_work_counters(state);
        gpu::SubmitDesc submit = {
            .command_lists = lists[:batch_size],
        };

        std::time::Clock start = std::time::clock::now();
        gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
        elapsed_ns += (long)start.to_now();

        vk::SubmitDuplicateWorkCounters work =
            vk::submit_duplicate_work_counters(state);
        assert(work.token_visits == batch_size,
            "submit duplicate work must visit each command record once");
        assert(work.epoch_reset_cells == 0,
            "ordinary submit batches must not perform epoch rollover work");
        total_token_visits += work.token_visits;
        total_epoch_reset_cells += work.epoch_reset_cells;
        gpu::wait_completion(completion)!!;
    }
    assert(total_token_visits
            == (ulong)batch_size * SUBMIT_BATCH_ITERATIONS,
        "repeated submits preserve one duplicate visit per command record");
    assert(total_epoch_reset_cells == 0,
        "repeated ordinary submits perform no epoch rollover work");
    io::printfn(
        "submit batch size=%d: %.1f ns/submit token_visits=%d epoch_reset_cells=%d",
        batch_size,
        (double)elapsed_ns / SUBMIT_BATCH_ITERATIONS,
        batch_size,
        0,
    );
}

fn int main() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
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
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);

    io::printfn(
        "iterations=%d batch_sizes=1,8,32,128,1024 units=ns/submit",
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
