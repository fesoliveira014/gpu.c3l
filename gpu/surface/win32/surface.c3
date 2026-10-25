module gpu::surface::win32;

import gpu @public;
import gpu::internal @public;
import gpu::internal::vk;

<* Create a runtime-owned Win32 surface from native handles.
   The runtime retains the surface; the caller destroys it with
   `gpu::destroy_surface`. The native instance and window must remain valid
   until then.
   @param runtime : "Live runtime that will own the surface."
   @param instance : "Borrowed non-null native instance handle."
   @param window : "Borrowed non-null native window handle."
   @return "A runtime-owned surface token, or a recoverable capacity, allocation, or platform fault."
   @require (void*)instance != null : "instance must not be null"
   @require (void*)window != null : "window must not be null" *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    InstanceHandle instance,
    WindowHandle window,
) {
    gpu::internal::RuntimeData* runtime_data = gpu::internal::require_runtime(runtime);
    void* backend_surface = gpu::internal::vk::vk_create_win32_surface(
        runtime_data.state,
        (void*)instance,
        (void*)window,
    )!;
    defer catch gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
