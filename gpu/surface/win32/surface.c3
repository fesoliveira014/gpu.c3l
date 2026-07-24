module gpu::surface::win32;

import gpu @public;
import gpu::internal @public;

<* Create a runtime-owned Win32 surface from native handles. *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    InstanceHandle instance,
    WindowHandle window,
) {
    void* native_instance = (void*)instance;
    void* native_window = (void*)window;
    if (native_instance == null || native_window == null) {
        return gpu::INVALID_ARGUMENT~;
    }

    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(runtime);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    if (runtime_data.vtable.create_win32_surface == null
        || runtime_data.vtable.destroy_surface == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    void* backend_surface = runtime_data.vtable.create_win32_surface(
        runtime_data.state,
        native_instance,
        native_window,
    )!;
    defer catch runtime_data.vtable.destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
