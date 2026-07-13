module gpu::vk @private;

import gpu @public;

const uint HANDLE_GENERATION_MAX @private = 0xFF_FFFF;

fn uint next_handle_generation(uint generation) @private {
    return generation >= HANDLE_GENERATION_MAX ? 1 : generation + 1;
}

fn bool is_pow2(usz value) @private {
    return value != 0 && (value & (value - 1)) == 0;
}

fn uint? resolve_descriptor_capacity(
    uint requested,
    uint default_capacity,
) @private {
    uint capacity = requested == 0 ? default_capacity : requested;
    if (capacity > gpu::MAX_DESCRIPTOR_SLOTS) return gpu::INVALID_ARGUMENT~;
    return capacity;
}
