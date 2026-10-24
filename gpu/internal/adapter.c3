module gpu::internal @private;

import gpu;


fn RuntimeData*? require_adapter(Adapter* adapter) @private {
    if (adapter == null) abort("gpu: null Adapter");
    if (!adapter.is_valid()) abort("gpu: zero Adapter");
    if (adapter.reserved != 0 || adapter.runtime_index >= MAX_RUNTIMES) {
        abort("gpu: malformed Adapter");
    }
    Runtime runtime = {
        .index      = adapter.runtime_index,
        .generation = adapter.runtime_generation,
    };
    RuntimeData* data = resolve_runtime(&runtime);
    if (data == null) abort("gpu: stale Adapter");
    if (adapter.adapter_index >= data.adapter_count) {
        abort("gpu: malformed Adapter");
    }
    return data;
}
