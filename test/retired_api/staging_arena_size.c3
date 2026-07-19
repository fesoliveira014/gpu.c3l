module retired_staging_arena_size;

import gpu;

fn gpu::DeviceDesc use_retired_staging_arena_size() {
    return { .staging_arena_size = 1 };
}
