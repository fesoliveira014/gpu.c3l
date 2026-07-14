module gpu;

const uint MAX_SURFACES @private = 64;
const uint MAX_SURFACE_GENERATION @private = 0x00FF_FFFF;

<* Runtime-owned platform surface token. *>
bitstruct Surface : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

const Surface SURFACE_INVALID = {};

struct SurfaceData @private {
    uint    generation;
    Runtime runtime;
    void*   backend_surface;
    uint    live_swapchain_count;
    bool    used;
}

SurfaceData[MAX_SURFACES] surface_slots @private;

fn bool Surface.is_valid(self) => (ulong)self != 0;

fn SurfaceData* resolve_surface(Surface* surface) @private {
    if (surface == null || !surface.is_valid()) return null;
    if (surface.reserved != 0 || surface.index >= MAX_SURFACES) return null;
    SurfaceData* data = &surface_slots[surface.index];
    if (!data.used || data.generation != surface.generation) return null;
    if (resolve_runtime(&data.runtime) == null) return null;
    return data;
}

fn Surface? register_surface(Runtime* runtime, void* backend_surface) @private {
    if (resolve_runtime(runtime) == null) return INVALID_HANDLE~;
    if (backend_surface == null) return INVALID_ARGUMENT~;

    for (uint i = 0; i < MAX_SURFACES; i++) {
        SurfaceData* data = &surface_slots[i];
        if (data.used || data.generation == MAX_SURFACE_GENERATION) continue;

        runtime_retain_surface(runtime)!;
        data.generation++;
        data.runtime = *runtime;
        data.backend_surface = backend_surface;
        data.live_swapchain_count = 0;
        data.used = true;
        return {
            .index      = i,
            .generation = data.generation,
        };
    }
    return SLOT_TABLE_FULL~;
}

fn void*? surface_backend_handle(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    return data.backend_surface;
}

fn void? surface_retain_swapchain(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_swapchain_count == uint::max) return INVALID_RESOURCE_STATE~;
    data.live_swapchain_count++;
}

fn void? surface_release_swapchain(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_swapchain_count == 0) return INVALID_RESOURCE_STATE~;
    data.live_swapchain_count--;
}

<* Return whether an adapter can present to a surface from the same runtime. *>
fn bool? supports_presentation(Adapter* adapter, Surface* surface) {
    RuntimeData* adapter_runtime = resolve_adapter(adapter);
    SurfaceData* surface_data = resolve_surface(surface);
    if (adapter_runtime == null || surface_data == null) return INVALID_HANDLE~;

    RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
    if (surface_runtime != adapter_runtime) return INVALID_HANDLE~;
    if (adapter_runtime.vtable.supports_presentation == null) return false;
    return adapter_runtime.vtable.supports_presentation(
        adapter_runtime.backend_state,
        adapter.adapter_index,
        surface_data.backend_surface,
    );
}

<* Destroy a surface after all swapchains that borrow it. *>
fn void? destroy_surface(Surface* surface) {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_swapchain_count > 0) return RESOURCE_IN_USE~;

    Runtime runtime = data.runtime;
    RuntimeData* runtime_data = resolve_runtime(&runtime);
    if (runtime_data == null) return INVALID_HANDLE~;
    void* backend_surface = data.backend_surface;

    data.runtime = RUNTIME_INVALID;
    data.backend_surface = null;
    data.live_swapchain_count = 0;
    data.used = false;
    *surface = SURFACE_INVALID;

    runtime_data.vtable.destroy_surface(
        runtime_data.backend_state,
        backend_surface,
    );
    runtime_release_surface(&runtime)!!;
}
