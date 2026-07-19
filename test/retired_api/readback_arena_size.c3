module retired_readback_arena_size;

import gpu;

fn gpu::DeviceDesc use_retired_readback_arena_size() {
    return { .readback_arena_size = 1 };
}
