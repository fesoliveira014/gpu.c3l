module gpu;

const uint MAX_SWAPCHAINS = 8;


<* Pacing of presentation; unsupported requested modes fall back to FIFO. *>
enum PresentMode : int {
    // Vsync; always available.
    FIFO,
    // No vsync; may tear.
    IMMEDIATE,
    // Low-latency vsync; may drop frames.
    MAILBOX,
}

<* Present modes a swapchain's surface supports, from get_present_mode_support. *>
bitstruct PresentModeSupport : uint {
    bool fifo;
    bool immediate;
    bool mailbox;
}


<* Creation parameters for create_swapchain. *>
struct SwapchainDesc {
    uint        width;
    uint        height;
    Format      preferred_format;
    PresentMode present_mode;
    // 0 means backend default.
    uint        image_count;
    // Selects sRGB over UNORM encoding for the swapchain format.
    bool        srgb;
    ZString     debug_name;
}

<* Actual swapchain configuration selected by the backend. Dormant snapshots
   use UNDEFINED format, zero extent/count, and FIFO as an ignored mode
   sentinel. *>
struct SwapchainInfo {
    Format      format;
    uint        width;
    uint        height;
    uint        image_count;
    PresentMode present_mode;
    bool        dormant;
}

// The texture handle is frame-transient: resize re-wraps images and stales
// prior handles, so hold it for the current frame only.
struct AcquiredImage {
    TextureHandle texture;
    uint          index;
    bool          suboptimal;
    TextureLayout prior_layout;
}

<* Parameters for present; targets the swapchain's most recently acquired image. *>
struct PresentDesc {
    SwapchainHandle swapchain;
}

<* The surface remains owned by its runtime and must outlive the swapchain. *>
fn SwapchainHandle? create_swapchain(Device* device, Surface* surface, SwapchainDesc* desc) {
    if (surface == null || desc == null) return INVALID_ARGUMENT~;
    DeviceData* device_data = resolve_device(device);
    SurfaceData* surface_data = resolve_surface(surface);
    if (device_data == null || surface_data == null) return INVALID_HANDLE~;
    if (!device_data.runtime_retained) return INVALID_RESOURCE_STATE~;
    if (!device_data.presentation_surface.is_valid()) return UNSUPPORTED_FEATURE~;
    if ((ulong)*surface != (ulong)device_data.presentation_surface) {
        return INVALID_HANDLE~;
    }

    RuntimeData* device_runtime = resolve_runtime(&device_data.runtime);
    RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
    if (device_runtime == null || surface_runtime != device_runtime) {
        return INVALID_HANDLE~;
    }
    return device_data.vtable.create_swapchain(device, surface, desc);
}

<* Counterpart of create_swapchain. Returns INVALID_RESOURCE_STATE while an
   acquired image is pending; otherwise waits for in-flight swapchain work. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_swapchain(device, swapchain);
}

<* Recreate under the same handle at a new extent. A zero clamped extent or
   a failed rebuild leaves the swapchain dormant (acquire faults
   SWAPCHAIN_OUT_OF_DATE) until a successful resize. Prior acquired-image
   handles become stale. Returns INVALID_RESOURCE_STATE while an acquired
   image is pending. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.resize_swapchain(device, swapchain, width, height);
}

<* Actual format, extent, image count, and present mode selected for a
   swapchain. Dormant swapchains return the dormant sentinel snapshot. *>
fn SwapchainInfo? get_swapchain_info(Device* device, SwapchainHandle swapchain) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.get_swapchain_info(device, swapchain);
}

<* Acquire the next image. WAIT_TIMEOUT is transient: retry the same swapchain.
   SWAPCHAIN_OUT_OF_DATE requires resize_swapchain; SURFACE_LOST requires
   destroying the swapchain and creating another from fresh platform handles.
   INVALID_RESOURCE_STATE means an acquire is already pending. A true
   `suboptimal` image remains usable but is a soft signal to resize soon. The
   rendering submit must name the swapchain in its SubmitDesc. *>
fn AcquiredImage? acquire_next_image(Device* device, SwapchainHandle swapchain) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.acquire_next_image(device, swapchain);
}

<* Present the last acquired image. Requires the coupled GRAPHICS submit that
   consumes the acquire to have run first and the image in the PRESENT tracked
   layout (INVALID_RESOURCE_STATE otherwise). SWAPCHAIN_OUT_OF_DATE requires
   resize_swapchain; SURFACE_LOST requires destroying the swapchain and
   creating another from fresh platform handles. *>
fn void? present(Device* device, PresentDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.present(device, desc);
}

<* Present modes the swapchain's surface supports. Unsupported requested
   modes fall back to FIFO at creation; query this to pick deliberately. *>
fn PresentModeSupport? get_present_mode_support(Device* device, SwapchainHandle swapchain) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.get_present_mode_support(device, swapchain);
}
