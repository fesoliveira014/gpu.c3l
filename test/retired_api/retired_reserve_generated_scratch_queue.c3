module retired_reserve_generated_queue;

import gpu;

fn void? use_retired_reserve_generated_scratch(
    gpu::Queue queue,
    gpu::GeneratedScratchDesc* desc,
) {
    gpu::reserve_generated_scratch(queue, desc)!;
}
