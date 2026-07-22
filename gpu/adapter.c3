module gpu;



fn RuntimeData* resolve_adapter(Adapter* adapter) @private {
    if (adapter == null || !adapter.is_valid()) return null;
    if (adapter.reserved != 0) return null;
    Runtime runtime = {
        .index      = adapter.runtime_index,
        .generation = adapter.runtime_generation,
    };
    RuntimeData* data = resolve_runtime(&runtime);
    if (data == null || adapter.adapter_index >= data.adapter_count) return null;
    return data;
}
