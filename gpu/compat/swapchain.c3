module gpu::compat;

const uint MAX_SWAPCHAINS = 8;

<* Windowing system that owns the native handles in SurfaceDesc. *>
enum PlatformKind : int {
    WAYLAND,
    X11,
    WIN32,
}

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

<* Native window-system handles; meaning follows `platform`: wl_display/wl_surface
   (WAYLAND), Display*/Window (X11), HINSTANCE/HWND (WIN32). *>
struct SurfaceDesc {
    PlatformKind platform;
    void*        native_display;
    void*        native_window;
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

<* Create a swapchain against platform-native window handles. Requires a
   device created with enable_presentation.
   Ownership: release with destroy_swapchain. *>
fn SwapchainHandle? create_swapchain(Device* device, SurfaceDesc* surface, SwapchainDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (surface == null || desc == null) return INVALID_ARGUMENT~;
    return vtable.create_swapchain(device, surface, desc);
}

<* Counterpart of create_swapchain; in-flight presentation must be idle. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_swapchain(device, swapchain);
}

<* Recreate under the same handle at a new extent. A zero clamped extent or
   a failed rebuild leaves the swapchain dormant (acquire faults
   SWAPCHAIN_OUT_OF_DATE) until a successful resize. Prior acquired-image
   handles become stale. *>
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
