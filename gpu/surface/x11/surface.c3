module gpu::surface::x11;

import gpu @public;
import gpu::internal @public;

<* Create a runtime-owned X11 surface from native handles. *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    DisplayHandle display,
    WindowHandle window,
) {
    void* native_display = (void*)display;
    ulong native_window = (ulong)window;
    if (native_display == null || native_window == 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(runtime);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    if (runtime_data.vtable.create_x11_surface == null
        || runtime_data.vtable.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    void* backend_surface = runtime_data.vtable.create_x11_surface(
        runtime_data.state,
        native_display,
        native_window,
    )!;
    defer catch runtime_data.vtable.destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
