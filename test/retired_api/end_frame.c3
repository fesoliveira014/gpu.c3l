module retired_end_frame;

import gpu;

fn void? use_retired_end_frame() {
    gpu::end_frame(null)!;
}