module retired_release_generated_queue;

import gpu;

fn void? use_retired_release_generated_scratch(
    gpu::Queue queue,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
) {
    gpu::release_generated_scratch(queue, pipeline, kind)!;
}
