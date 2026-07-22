module gpu::internal @private;

import gpu;

const uint MAX_SURFACES @private = 64;
const uint MAX_SURFACE_GENERATION @private = 0x00FF_FFFF;

struct SurfaceData @private {
    uint    generation;
    Runtime runtime;
    void*   backend_surface;
    uint    live_swapchain_count;
    bool    used;
}

SurfaceData[MAX_SURFACES] surface_slots @private;


fn SurfaceData* resolve_surface(Surface* surface) @private {
    if (surface == null || !surface.is_valid()) return null;
    if (surface.reserved != 0 || surface.index >= MAX_SURFACES) return null;
    SurfaceData* data = &surface_slots[surface.index];
    if (!data.used || data.generation != surface.generation) return null;
    if (resolve_runtime(&data.runtime) == null) return null;
    return data;
}

fn Surface? register_surface(Runtime* runtime, void* backend_surface) @private {
    if (resolve_runtime(runtime) == null) return gpu::INVALID_HANDLE~;
    if (backend_surface == null) return gpu::INVALID_ARGUMENT~;

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
    return gpu::SLOT_TABLE_FULL~;
}

fn void*? surface_backend_handle(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return gpu::INVALID_HANDLE~;
    return data.backend_surface;
}

fn void? surface_retain_swapchain(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return gpu::INVALID_HANDLE~;
    if (data.live_swapchain_count == uint::max) return gpu::INVALID_RESOURCE_STATE~;
    data.live_swapchain_count++;
}

fn void? surface_release_swapchain(Surface* surface) @private {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return gpu::INVALID_HANDLE~;
    if (data.live_swapchain_count == 0) return gpu::INVALID_RESOURCE_STATE~;
    data.live_swapchain_count--;
}
