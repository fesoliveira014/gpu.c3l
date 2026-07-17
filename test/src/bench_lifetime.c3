module gpu_bench;

import gpu @public;

fn void? discard_bench_executable(gpu::ExecutableCommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_executable_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn gpu::CompletionPoint? submit_bench_queue(
    gpu::Device* device,
    gpu::QueueKind kind,
    gpu::SubmitDesc* desc,
) {
    return gpu::submit(gpu::get_queue(device, kind)!, desc);
}

fn void? wait_bench_queue(gpu::Device* device, gpu::QueueKind queue) {
    gpu::Queue target = gpu::get_queue(device, queue)!;
    gpu::CommandList commands = gpu::begin_commands(target)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(target, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? destroy_bench_device(gpu::Device* device) {
    if (device == null || !device.is_valid()) return;
    gpu::QueueCounts? counts_result = gpu::get_queue_counts(device);
    if (catch err = counts_result) {
        if (err == gpu::DEVICE_LOST) {
            gpu::destroy_device(device)!;
            return;
        }
        return err~;
    }
    gpu::QueueCounts counts = counts_result;
    if (counts.graphics > 0) {
        wait_bench_queue(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        wait_bench_queue(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        wait_bench_queue(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
}
