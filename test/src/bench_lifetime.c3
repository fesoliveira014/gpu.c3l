module gpu_bench;

import gpu @public;

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