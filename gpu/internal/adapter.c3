module gpu::internal @private;

import gpu;


fn RuntimeData*? require_adapter(Adapter* adapter) @private {
    if (adapter == null) return reject_handle(HandleKind.ADAPTER, HandleCause.NULL_POINTER)~;
    if (!adapter.is_valid()) return reject_handle(HandleKind.ADAPTER, HandleCause.ZERO)~;
    if (adapter.reserved != 0 || adapter.runtime_index >= MAX_RUNTIMES) {
        return reject_handle(HandleKind.ADAPTER, HandleCause.MALFORMED)~;
    }
    Runtime runtime = {
        .index      = adapter.runtime_index,
        .generation = adapter.runtime_generation,
    };
    RuntimeData* data = resolve_runtime(&runtime);
    if (data == null) return reject_handle(HandleKind.ADAPTER, HandleCause.STALE)~;
    if (adapter.adapter_index >= data.adapter_count) {
        return reject_handle(HandleKind.ADAPTER, HandleCause.MALFORMED)~;
    }
    return data;
}

fn RuntimeData* resolve_adapter(Adapter* adapter) @private {
    return require_adapter(adapter) ?? null;
}
