module gpu;

enum AdapterClass : int {
    OTHER,
    INTEGRATED,
    DISCRETE,
    VIRTUAL,
    SOFTWARE,
}

bitstruct Adapter : ulong {
    ushort adapter_index      : 0..15;
    ushort runtime_index      : 16..31;
    uint   runtime_generation : 32..55;
    uint   reserved           : 56..63;
}

const Adapter ADAPTER_INVALID = {};

struct AdapterMemoryInfo {
    ulong device_local_bytes;
    ulong host_visible_bytes;
}

struct AdapterQueueInfo {
    uint graphics_count;
    uint compute_count;
    uint transfer_count;
}

struct AdapterLimits {
    uint  max_texture_dimension_2d;
    uint  max_texture_array_layers;
    uint  max_color_attachments;
    uint  max_push_constant_size;
    Vec3u max_compute_work_group_count;
    uint  max_draw_indirect_count;
}

<* Cached semantic data. name is a borrowed read-only view until runtime destruction. *>
struct AdapterInfo {
    String            name;
    uint              vendor_id;
    uint              device_id;
    AdapterClass      device_class;
    AdapterMemoryInfo memory;
    AdapterQueueInfo  queues;
    bool              strict_supported;
    AdapterLimits     limits;
}

struct BackendVersion {
    uint major;
    uint minor;
    uint patch;
}

<* Backend diagnostics. Strings are borrowed read-only views until runtime destruction. *>
struct AdapterDiagnostics {
    String         backend_name;
    BackendVersion backend_version;
    String         driver_name;
    String         driver_info;
    uint           driver_version;
}

<* Allocation-free view of one runtime's borrowed adapters. *>
struct AdapterList {
    Runtime runtime;
    uint    count;
}

<* True when the token is not the invalid zero value. *>
fn bool Adapter.is_valid(self) => (ulong)self != 0;

<* Return one borrowed adapter from this runtime-owned view. *>
fn Adapter? AdapterList.get(&self, uint index) {
    RuntimeData* data = resolve_runtime(&self.runtime);
    if (data == null) return INVALID_HANDLE~;
    if (index >= self.count || index >= data.adapter_count || index > ushort::max) {
        return INVALID_ARGUMENT~;
    }
    return (Adapter){
        .adapter_index      = (ushort)index,
        .runtime_index      = (ushort)self.runtime.index,
        .runtime_generation = self.runtime.generation,
    };
}

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

<* Return an allocation-free view of the runtime's borrowed adapters. *>
fn AdapterList? enumerate_adapters(Runtime* runtime) {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    return {
        .runtime = *runtime,
        .count   = data.adapter_count,
    };
}

<* Query backend-neutral information for a borrowed adapter. *>
fn AdapterInfo? get_adapter_info(Adapter* adapter) {
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_info(
        data.backend_state,
        adapter.adapter_index,
    );
}

<* Query backend and driver identity for diagnostics and logging. *>
fn AdapterDiagnostics? get_adapter_diagnostics(Adapter* adapter) {
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_diagnostics(
        data.backend_state,
        adapter.adapter_index,
    );
}
