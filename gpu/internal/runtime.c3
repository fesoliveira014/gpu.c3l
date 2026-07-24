module gpu::internal @private;

import gpu;
import std::atomic;
import std::atomic::types;
import std::core::mem;

alias DestroyRuntimeBackendFn @private = fn void (void*);
alias GetAdapterInfoBackendFn @private = fn AdapterInfo? (void*, uint);
alias GetAdapterDiagnosticsBackendFn @private = fn AdapterDiagnostics? (void*, uint);
alias CreateWin32SurfaceBackendFn @private = fn void*? (void*, void*, void*);
alias CreateWaylandSurfaceBackendFn @private = fn void*? (void*, void*, void*);
alias CreateX11SurfaceBackendFn @private = fn void*? (void*, void*, ulong);
alias DestroySurfaceBackendFn @private = fn void (void*, void*);
alias SupportsPresentationBackendFn @private = fn bool? (void*, uint, void*);
alias SupportsQueueRequirementsBackendFn @private = fn bool? (void*, uint, QueueRequirements*, void*);

const uint MAX_RUNTIMES @private = 16;
const uint MAX_ADAPTERS_PER_RUNTIME @private = (uint)ushort::max + 1;
const uint MAX_RUNTIME_GENERATION @private = 0x00FF_FFFF;


struct RuntimeBackendVTable @private {
    DestroyRuntimeBackendFn        destroy_runtime;
    GetAdapterInfoBackendFn        get_adapter_info;
    GetAdapterDiagnosticsBackendFn get_adapter_diagnostics;
    CreateWin32SurfaceBackendFn     create_win32_surface;
    CreateWaylandSurfaceBackendFn   create_wayland_surface;
    CreateX11SurfaceBackendFn       create_x11_surface;
    DestroySurfaceBackendFn         destroy_surface;
    SupportsPresentationBackendFn   supports_presentation;
    SupportsQueueRequirementsBackendFn supports_queue_requirements;
}

struct RuntimeData @private {
    uint                  generation;
    BackendKind           backend;
    RuntimeBackendVTable* vtable;
    void*                 backend_state;
    uint                  adapter_count;
    uint                  live_surface_count;
    Atomic{uint}          live_device_count;
    bool                  used;
}

RuntimeData[MAX_RUNTIMES] runtime_slots @private;


fn bool contract_validation_is_known(ContractValidation policy) @private {
    switch (policy) {
        case ContractValidation.TRUSTED:
        case ContractValidation.OBJECT_BOUNDARIES:
        case ContractValidation.FULL:
            return true;
        default:
            return false;
    }
}

fn void? validate_contract_validation(ContractValidation policy) @private {
    if (!contract_validation_is_known(policy)) return gpu::INVALID_ARGUMENT~;
}

fn RuntimeData* resolve_runtime(Runtime* runtime) @private {
    if (runtime == null || !runtime.is_valid()) return null;
    if (runtime.reserved != 0) return null;
    if (runtime.index >= MAX_RUNTIMES) return null;
    RuntimeData* data = &runtime_slots[runtime.index];
    if (!data.used || data.generation != runtime.generation) return null;
    return data;
}

fn Runtime? register_runtime(
    BackendKind backend,
    RuntimeBackendVTable* vtable,
    void* backend_state,
    uint adapter_count,
) @private {
    if (vtable == null || backend_state == null) return gpu::INVALID_ARGUMENT~;
    if (adapter_count > MAX_ADAPTERS_PER_RUNTIME) return gpu::SLOT_TABLE_FULL~;
    for (uint i = 0; i < MAX_RUNTIMES; i++) {
        RuntimeData* data = &runtime_slots[i];
        if (data.used || data.generation == MAX_RUNTIME_GENERATION) continue;
        data.generation++;
        data.backend = backend;
        data.vtable = vtable;
        data.backend_state = backend_state;
        data.adapter_count = adapter_count;
        data.live_surface_count = 0;
        data.live_device_count.store(0, AtomicOrdering.RELAXED);
        data.used = true;
        return (Runtime){
            .index      = i,
            .generation = data.generation,
        };
    }
    return gpu::SLOT_TABLE_FULL~;
}

fn void* runtime_backend_state(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    return data == null ? null : data.backend_state;
}

fn void? runtime_retain_surface(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return gpu::INVALID_HANDLE~;
    if (data.live_surface_count == uint::max) return gpu::INVALID_RESOURCE_STATE~;
    data.live_surface_count++;
}

fn void? runtime_release_surface(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return gpu::INVALID_HANDLE~;
    if (data.live_surface_count == 0) return gpu::INVALID_RESOURCE_STATE~;
    data.live_surface_count--;
}

fn void? runtime_retain_device(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return gpu::INVALID_HANDLE~;
    uint count = data.live_device_count.load(AtomicOrdering.RELAXED);
    while (true) {
        if (count == uint::max) return gpu::INVALID_RESOURCE_STATE~;
        uint seen = mem::compare_exchange(
            &data.live_device_count.data,
            count,
            count + 1,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.RELAXED,
        );
        if (seen == count) return;
        count = seen;
    }
}

fn void? runtime_release_device(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return gpu::INVALID_HANDLE~;
    uint count = data.live_device_count.load(AtomicOrdering.RELAXED);
    while (true) {
        if (count == 0) return gpu::INVALID_RESOURCE_STATE~;
        uint seen = mem::compare_exchange(
            &data.live_device_count.data,
            count,
            count - 1,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.RELAXED,
        );
        if (seen == count) return;
        count = seen;
    }
}

fn void? validate_runtime_desc(RuntimeDesc* desc) @private {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    validate_contract_validation(desc.contract_validation)!;
    $if $feature(DIRECT_COMMAND_TOKENS):
        if (desc.contract_validation == ContractValidation.FULL) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
    $endif
    if (desc.texture_heap_capacity > gpu::MAX_SHADER_HEAP_CAPACITY
        || desc.sampler_heap_capacity > gpu::MAX_SHADER_HEAP_CAPACITY
        || desc.texture_capacity > gpu::MAX_SHADER_HEAP_CAPACITY
        || desc.pipeline_capacity > gpu::MAX_SHADER_HEAP_CAPACITY) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.pipeline_cache_data.len > 0
        && desc.pipeline_cache_data.ptr == null) return gpu::INVALID_ARGUMENT~;
}
