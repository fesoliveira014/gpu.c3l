module retired_alloc_persistent_span;
import gpu;
fn void? use_retired_alloc_persistent_span(gpu::Device* device) {
    gpu::alloc_persistent_span(device, null)!;
}
