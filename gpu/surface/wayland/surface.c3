module gpu::surface::wayland;

import gpu @public;
import gpu::internal @public;

<* Create a runtime-owned Wayland surface from native handles. *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    DisplayHandle display,
    SurfaceHandle surface,
) {
    void* native_display = (void*)display;
    void* native_surface = (void*)surface;
    if (native_display == null || native_surface == null) {
        return gpu::INVALID_ARGUMENT~;
    }

    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(runtime);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    if (runtime_data.vtable.create_wayland_surface == null
        || runtime_data.vtable.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    void* backend_surface = runtime_data.vtable.create_wayland_surface(
        runtime_data.backend_state,
        native_display,
        native_surface,
    )!;
    defer catch runtime_data.vtable.destroy_surface(
        runtime_data.backend_state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
