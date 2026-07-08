module gpu;

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

// The texture handle is frame-transient: resize re-wraps images and stales
// prior handles, so hold it for the current frame only.
struct AcquiredImage {
    TextureHandle texture;
    uint          index;
    bool          suboptimal;
}

<* Parameters for present; targets the swapchain's most recently acquired image. *>
struct PresentDesc {
    SwapchainHandle swapchain;
}

<* Create a swapchain against platform-native window handles. Requires a
   device created with enable_presentation.
   Ownership: release with destroy_swapchain. *>
fn SwapchainHandle? create_swapchain(Device* device, SurfaceDesc* surface, SwapchainDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_swapchain(device, surface, desc);
}

<* Counterpart of create_swapchain; in-flight presentation must be idle. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_swapchain(device, swapchain);
}

<* Recreate under the same handle at a new extent. A zero clamped extent or
   a failed rebuild leaves the swapchain dormant (acquire faults
   SWAPCHAIN_OUT_OF_DATE) until a successful resize. Prior acquired-image
   handles become stale. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.resize_swapchain(device, swapchain, width, height);
}

<* Acquire the next image. Faults SWAPCHAIN_OUT_OF_DATE when the swapchain
   needs a resize and INVALID_RESOURCE_STATE while a prior acquire is still
   pending; a true `suboptimal` is a soft signal to resize soon. The submit
   that renders to the image must name the swapchain in its SubmitDesc. *>
fn AcquiredImage? acquire_next_image(Device* device, SwapchainHandle swapchain) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.acquire_next_image(device, swapchain);
}

<* Present the last acquired image. Requires the coupled GRAPHICS submit that
   consumes the acquire to have run first (INVALID_RESOURCE_STATE otherwise)
   and the image in the PRESENT tracked layout (INVALID_RESOURCE_STATE).
   Faults SWAPCHAIN_OUT_OF_DATE when the surface changed. *>
fn void? present(Device* device, PresentDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.present(device, desc);
}

<* Present modes the swapchain's surface supports. Unsupported requested
   modes fall back to FIFO at creation; query this to pick deliberately. *>
fn PresentModeSupport? get_present_mode_support(Device* device, SwapchainHandle swapchain) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_present_mode_support(device, swapchain);
}
