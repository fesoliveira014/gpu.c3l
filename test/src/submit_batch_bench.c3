module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::time;

const uint MAX_SUBMIT_BATCH_SIZE = 1_024;
const uint[5] SUBMIT_BATCH_SIZES = { 1, 8, 32, 128, 1_024 };

fn void record_submit_batch(
    gpu::Queue queue,
    gpu::ExecutableCommandList[] out_lists,
) {
    foreach (&out_list : out_lists) {
        gpu::CommandList commands = gpu::begin_commands(queue)!!;
        *out_list = gpu::end_commands(&commands)!!;
    }
}

fn void measure_submit_batch(
    gpu::Queue queue,
    vk::VkDeviceState* state,
    uint batch_size,
) {
    gpu::ExecutableCommandList[MAX_SUBMIT_BATCH_SIZE] lists;
    record_submit_batch(queue, lists[:batch_size]);
    vk::reset_submit_duplicate_work_counters(state);
    gpu::SubmitDesc submit = {
        .command_lists = lists[:batch_size],
    };

    std::time::Clock start = std::time::clock::now();
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    long elapsed_ns = (long)start.to_now();

    vk::SubmitDuplicateWorkCounters work =
        vk::submit_duplicate_work_counters(state);
    assert(work.token_visits == batch_size,
        "submit duplicate work must visit each command record once");
    assert(work.epoch_reset_cells == 0,
        "ordinary submit batches must not perform epoch rollover work");
    io::printfn(
        "submit batch size=%d: %.1f ns/submit token_visits=%d epoch_reset_cells=%d",
        batch_size,
        (double)elapsed_ns,
        work.token_visits,
        work.epoch_reset_cells,
    );
    gpu::wait_completion(completion)!!;
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
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);

    io::printn("iterations=batch_sizes=1,8,32,128,1024 units=ns/submit");
    foreach (batch_size : SUBMIT_BATCH_SIZES) {
        measure_submit_batch(queue, state, batch_size);
    }
    io::printfn("submit batch leaks=%d", vk::report_resource_leaks(state));
    return 0;
}
