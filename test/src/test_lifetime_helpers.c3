module gpu_test;

import gpu @public;

fn void? discard_test_commands(gpu::CommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn void? wait_test_queue(gpu::Device* device, gpu::QueueKind queue) {
    gpu::CommandList[1] lists;
    lists[0] = gpu::begin_commands(device, queue)!;
    defer (void)gpu::discard_commands(&lists[0]);
    gpu::end_commands(&lists[0])!;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(device, &submit)!;
    gpu::wait_completion(completion, ulong::max)!;
}

fn void? destroy_test_device(gpu::Device* device) {
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
        wait_test_queue(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        wait_test_queue(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        wait_test_queue(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
}
