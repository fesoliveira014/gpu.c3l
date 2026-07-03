module gpu;

const uint MAX_SWAPCHAINS = 8;

enum PlatformKind : int {
    WAYLAND,
    X11,
    WIN32,
}

enum PresentMode : int {
    FIFO,
    IMMEDIATE,
    MAILBOX,
}

struct SurfaceDesc {
    PlatformKind platform;
    void*        native_display;
    void*        native_window;
}

struct SwapchainDesc {
    uint        width;
    uint        height;
    Format      preferred_format;
    PresentMode present_mode;
    uint        image_count;
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

fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_swapchain(device, swapchain);
}

<* Recreate under the same handle at a new extent. A zero clamped extent
   leaves the swapchain dormant (acquire faults SWAPCHAIN_OUT_OF_DATE) until a
   non-zero resize. Prior acquired-image handles become stale. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.resize_swapchain(device, swapchain, width, height);
}

<* Acquire the next image. Faults SWAPCHAIN_OUT_OF_DATE when the swapchain
   needs a resize; a true `suboptimal` is a soft signal to resize soon. The
   submit that renders to the image must name the swapchain in its SubmitDesc. *>
fn AcquiredImage? acquire_next_image(Device* device, SwapchainHandle swapchain) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.acquire_next_image(device, swapchain);
}

<* Present the last acquired image. The image must be in the PRESENT layout.
   Faults SWAPCHAIN_OUT_OF_DATE when the surface changed. *>
fn void? present(Device* device, PresentDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.present(device, desc);
}
