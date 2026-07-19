module retired_memory_kind;

import gpu;

fn void use_retired_memory_kind() {
    gpu::MemoryKind kind = {};
    (void)kind;
}