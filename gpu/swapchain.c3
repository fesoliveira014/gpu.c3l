module gpu;

const uint MAX_SWAPCHAINS = 8;
const SwapchainReadiness SWAPCHAIN_READINESS_INVALID = {};


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

<* One-shot acquisition readiness. Submission accepts it exactly once. *>
struct SwapchainReadiness {
    Device device;
    ulong  payload;
}
$assert(SwapchainReadiness::size <= ulong::size * 2);

<* Borrowed swapchain texture, its color attachment view, and its one-shot
   acquisition identity. Resize invalidates both handles; an enqueued
   presentation consumes the value. *>
struct AcquiredImage {
    TextureHandle        texture;
    AttachmentViewHandle attachment_view;
    SwapchainReadiness   readiness;
    uint                 index;
    bool                 suboptimal;
    TextureState         prior_state;
}

fn bool SwapchainReadiness.is_valid(self) => self.device.is_valid()
    && self.payload != 0;
fn bool SwapchainReadiness.equals(
    self,
    SwapchainReadiness other,
) @operator(==) {
    return self.device == other.device && self.payload == other.payload;
}

<* The surface remains owned by its runtime and must outlive the swapchain. *>
fn SwapchainHandle? create_swapchain(Device* device, Surface* surface, SwapchainDesc* desc) {
    if (surface == null || desc == null) return INVALID_ARGUMENT~;
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    DeviceData* device_data = operation.data;
    SurfaceData* surface_data = resolve_surface(surface);
    if (surface_data == null) return INVALID_HANDLE~;
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
    SwapchainHandle swapchain = device_data.vtable.create_swapchain(device, surface, desc)!;
    retain_device_children(&operation);
    return swapchain;
}

<* Counterpart of create_swapchain. Immediate: pending acquisition,
   presentation, command, or view use faults without consuming the handle. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    operation.vtable.destroy_swapchain(device, swapchain)!;
    release_device_children(&operation);
}

<* Recreate under the same handle at a new extent. A zero clamped extent or
   a failed rebuild leaves the swapchain dormant (acquire faults
   SWAPCHAIN_OUT_OF_DATE) until a successful resize. Prior acquired-image
   handles become stale. Pending acquisition, presentation, command, or view
   use faults immediately and leaves the swapchain unchanged. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.resize_swapchain(device, swapchain, width, height);
}

<* Actual format, extent, image count, and present mode selected for a
   swapchain. Dormant swapchains return the dormant sentinel snapshot. *>
fn SwapchainInfo? get_swapchain_info(Device* device, SwapchainHandle swapchain) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_swapchain_info(device, swapchain);
}

<* Acquire a borrowed texture, its color attachment view, and one-shot
   readiness. WAIT_TIMEOUT is retryable; out-of-date requires resize and
   surface loss requires replacement. *>
fn AcquiredImage? acquire_next_image(Device* device, SwapchainHandle swapchain) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.acquire_next_image(device, swapchain);
}

<* Present and consume an acquired image after its readiness-consuming submit.
   `render_completion` must be the point returned by that submit. WAIT_TIMEOUT
   preserves the image for retry when private present completion is still busy. *>
fn void? present(
    Device* device,
    AcquiredImage* image,
    CompletionPoint render_completion,
) {
    if (image == null) return INVALID_ARGUMENT~;
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_completion_point(&operation, device, render_completion)!;
    operation.vtable.present(device, image, render_completion)!;
    *image = {};
}

<* Present modes the swapchain's surface supports. Unsupported requested
   modes fall back to FIFO at creation; query this to pick deliberately. *>
fn PresentModeSupport? get_present_mode_support(Device* device, SwapchainHandle swapchain) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_present_mode_support(device, swapchain);
}
