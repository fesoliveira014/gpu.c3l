module gpu::surface::x11;

import gpu @public;
import gpu::internal @public;
import gpu::internal::vk;

<* Create a runtime-owned X11 surface from native handles.
   The runtime retains the surface; the caller destroys it with
   `gpu::destroy_surface`. The native display and window must remain valid
   until then.
   @param runtime : "Live runtime that will own the surface."
   @param display : "Borrowed non-null native display handle."
   @param window : "Borrowed nonzero native window handle."
   @return "A runtime-owned surface token, or a recoverable capacity, allocation, or platform fault."
   @require (void*)display != null : "display must not be null"
   @require (ulong)window != 0 : "window must be nonzero" *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    DisplayHandle display,
    WindowHandle window,
) {
    gpu::internal::RuntimeData* runtime_data = gpu::internal::require_runtime(runtime);
    void* backend_surface = gpu::internal::vk::vk_create_x11_surface(
        runtime_data.state,
        (void*)display,
        (ulong)window,
    )!;
    defer catch gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
