module gpu::surface::wayland;

import gpu @public;
import gpu::internal @public;
import gpu::internal::vk;

<* Create a runtime-owned Wayland surface from native handles.
   The runtime retains the surface; the caller destroys it with
   `gpu::destroy_surface`. The native display and surface must remain valid
   until then.

   @param runtime : "Live runtime that will own the surface."
   @param display : "Borrowed non-null native display handle."
   @param surface : "Borrowed non-null native surface handle."
   @return "A runtime-owned surface token, or a recoverable invalid-argument, invalid-handle, capacity, allocation, or platform fault." *>
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
    void* backend_surface = gpu::internal::vk::vk_create_wayland_surface(
        runtime_data.state,
        native_display,
        native_surface,
    )!;
    defer catch gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
