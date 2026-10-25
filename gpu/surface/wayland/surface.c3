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
   @return "A runtime-owned surface token, or a recoverable capacity, allocation, or platform fault."
   @require (void*)display != null : "display must not be null"
   @require (void*)surface != null : "surface must not be null" *>
fn gpu::Surface? create_surface(
    gpu::Runtime* runtime,
    DisplayHandle display,
    SurfaceHandle surface,
) {
    gpu::internal::RuntimeData* runtime_data = gpu::internal::require_runtime(runtime);
    void* backend_surface = gpu::internal::vk::vk_create_wayland_surface(
        runtime_data.state,
        (void*)display,
        (void*)surface,
    )!;
    defer catch gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    return gpu::internal::register_surface(runtime, backend_surface);
}
