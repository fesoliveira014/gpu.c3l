module gpu;

import gpu::vk @public;
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

const uint MAX_RUNTIMES @private = 16;
const uint MAX_ADAPTERS_PER_RUNTIME @private = (uint)ushort::max + 1;
const uint MAX_RUNTIME_GENERATION @private = 0x00FF_FFFF;

bitstruct Runtime : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

const Runtime RUNTIME_INVALID = {};

<* Runtime discovery configuration; zero-initialized fields select defaults. *>
struct RuntimeDesc {
    BackendKind          backend;
    bool                 enable_validation;
    ZString              application_name;
    DebugMessageCallback debug_callback;
    void*                debug_user_data;
}

struct RuntimeBackendVTable @private {
    DestroyRuntimeBackendFn        destroy_runtime;
    GetAdapterInfoBackendFn        get_adapter_info;
    GetAdapterDiagnosticsBackendFn get_adapter_diagnostics;
    CreateWin32SurfaceBackendFn     create_win32_surface;
    CreateWaylandSurfaceBackendFn   create_wayland_surface;
    CreateX11SurfaceBackendFn       create_x11_surface;
    DestroySurfaceBackendFn         destroy_surface;
    SupportsPresentationBackendFn   supports_presentation;
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

<* True when the token is not the invalid zero value. *>
fn bool Runtime.is_valid(self) => (ulong)self != 0;

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
    if (vtable == null || backend_state == null) return INVALID_ARGUMENT~;
    if (adapter_count > MAX_ADAPTERS_PER_RUNTIME) return SLOT_TABLE_FULL~;
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
    return SLOT_TABLE_FULL~;
}

fn void* runtime_backend_state(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    return data == null ? null : data.backend_state;
}

fn void? runtime_retain_surface(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count == uint::max) return INVALID_RESOURCE_STATE~;
    data.live_surface_count++;
}

fn void? runtime_release_surface(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count == 0) return INVALID_RESOURCE_STATE~;
    data.live_surface_count--;
}

fn void? runtime_retain_device(Runtime* runtime) @private {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    uint count = data.live_device_count.load(AtomicOrdering.RELAXED);
    while (true) {
        if (count == uint::max) return INVALID_RESOURCE_STATE~;
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
    if (data == null) return INVALID_HANDLE~;
    uint count = data.live_device_count.load(AtomicOrdering.RELAXED);
    while (true) {
        if (count == 0) return INVALID_RESOURCE_STATE~;
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

<* Initialize backend discovery. Importing gpu performs no initialization. *>
fn Runtime? create_runtime(RuntimeDesc* desc) {
    if (desc == null) return INVALID_ARGUMENT~;
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::vk::create_vk_runtime(desc);
        default:
            DebugMessage message = {
                .severity       = DebugMessageSeverity.ERROR,
                .category       = { .public_contract },
                .operation      = "create_runtime",
                .has_fault      = true,
                .public_fault   = UNSUPPORTED_BACKEND,
                .rejected_field = "RuntimeDesc.backend",
                .invariant      = "backend must be supported",
                .backend_text   = "unsupported GPU backend",
            };
            dispatch_debug_message(desc.debug_callback, desc.debug_user_data, &message);
            return UNSUPPORTED_BACKEND~;
    }
}

<* Destroy a runtime after its surfaces and devices have been destroyed. *>
fn void? destroy_runtime(Runtime* runtime) {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count > 0 || data.live_device_count.load(AtomicOrdering.ACQUIRE) > 0) {
        return RESOURCE_IN_USE~;
    }

    DestroyRuntimeBackendFn destroy_backend = data.vtable.destroy_runtime;
    void* backend_state = data.backend_state;
    data.vtable = null;
    data.backend_state = null;
    data.adapter_count = 0;
    data.used = false;
    *runtime = RUNTIME_INVALID;
    destroy_backend(backend_state);
}
