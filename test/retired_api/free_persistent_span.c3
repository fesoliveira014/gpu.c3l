module retired_free_persistent_span;

import gpu;

fn void? use_retired_free_persistent_span(gpu::Device* device) {
    gpu::free_persistent_span(device, {})!;
}
