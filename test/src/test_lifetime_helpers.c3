module gpu_test;

import gpu @public;

fn void? discard_test_commands(gpu::CommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
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
        gpu::wait_queue_idle(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        gpu::wait_queue_idle(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        gpu::wait_queue_idle(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
}
