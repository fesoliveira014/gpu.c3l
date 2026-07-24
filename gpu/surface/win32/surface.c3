module gpu::surface::win32;

import gpu @public;
import gpu::internal @public;
import gpu::internal::vk;

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
    void* backend_surface = gpu::internal::vk::vk_create_win32_surface(
        runtime_data.state,
        native_instance,
        native_window,
    )!;
    defer catch gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
