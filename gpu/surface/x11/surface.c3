module gpu::surface::x11;

import gpu @public;

typedef DisplayHandle = void*;
typedef WindowHandle = ulong;

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

    gpu::RuntimeData* runtime_data = gpu::resolve_runtime(runtime);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    if (runtime_data.vtable.create_x11_surface == null
        || runtime_data.vtable.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    void* backend_surface = runtime_data.vtable.create_x11_surface(
        runtime_data.backend_state,
        native_display,
        native_window,
    )!;
    defer catch runtime_data.vtable.destroy_surface(
        runtime_data.backend_state,
        backend_surface,
    );
    return gpu::register_surface(runtime, backend_surface);
}
