module retired_alloc_frame_span;

import gpu;

fn void? use_retired_alloc_frame_span() {
    (void)gpu::alloc_frame_span(null, 1, 1)!;
}