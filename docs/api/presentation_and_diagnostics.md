# Presentation and diagnostics

## Platform surfaces

Create a `Surface` through the module matching the active native window:

- `gpu::surface::wayland::create_surface(Runtime*, DisplayHandle,
  SurfaceHandle)`;
- `gpu::surface::x11::create_surface(Runtime*, DisplayHandle, WindowHandle)`;
  or
- `gpu::surface::win32::create_surface(Runtime*, InstanceHandle,
  WindowHandle)`.

Wayland handles are opaque pointers, X11 uses an opaque display pointer and an
unsigned window ID, and Win32 uses opaque instance/window pointers. These
aliases describe borrowed native values and do not transfer ownership.

Surface creation retains the runtime. Keep the native instance, display, and
window alive through `destroy_surface`. Surface registry mutation and support
checks are externally synchronized process-wide. `destroy_surface` returns
`RESOURCE_IN_USE` while a swapchain retains it.

## Swapchain lifecycle

`create_swapchain` uses one presentation-enabled `Device`, its exact `Surface`,
and a `SwapchainDesc` containing extent, format preference, image count,
presentation mode, and usage. It returns a `SwapchainHandle`.

`get_swapchain_info` reports the selected extent, format, image count, and
related runtime state. `get_present_mode_support` reports supported
`PresentMode` values through `PresentModeSupport`.

`resize_swapchain` recreates device-side presentation resources for a new
extent. It never waits for outstanding image use. Quiesce covering completion
points and release acquired images first; retry `RESOURCE_IN_USE` or
`DEVICE_BUSY` after progress.

`destroy_swapchain` releases its surface retain only when no image is acquired
or pending. It inserts no hidden wait.

## Acquire, submit, and present

`acquire_next_image` waits up to a caller-selected timeout and returns
`AcquiredImage`:

- the swapchain texture handle;
- its `prior_state`; and
- one-shot `SwapchainReadiness`.

The default timeout is zero. `WAIT_TIMEOUT` leaves the swapchain usable.
`SWAPCHAIN_OUT_OF_DATE` requests resize/recreation. Transition
`prior_state` to an attachment state before rendering, then transition the
image to `TextureLayout.PRESENT`.

Pass readiness in `SubmitDesc` for the first submission that consumes the
image. Successful submit consumes it. `present` consumes the acquired image
after the rendering completion point is ordered on the presentation queue.
Presentation may temporarily return `WAIT_TIMEOUT`; pump events and retry.

Acquire, resize, queries, and present are externally synchronized per
swapchain/native queue as specified. They do not access the process-wide
surface registry after creation.

## Structured diagnostics

`RuntimeDesc.debug_callback` installs a `DebugMessageCallback` with borrowed
userdata. `DebugMessage` includes severity, category, operation, optional
public fault, optional `DebugResourceRef`, rejected field, invariant, backend
text, and native validation identifiers.

`DebugMessageSeverity`, `DebugMessageCategory`, and `DebugResourceKind` allow
filtering without parsing message text.

Delivery is synchronous and may occur concurrently on arbitrary backend
threads. All pointers and strings are valid only during the callback. The
callback must:

- synchronize its own userdata;
- return promptly;
- avoid blocking backend progress; and
- never call back into `gpu.c3l`.

Callback presence controls delivery only. It does not enable contract checks,
Vulkan validation, debug names, leak tracking, or alter returned faults.

## Memory and debug reporting

`get_memory_stats` returns advisory heap usage/budget snapshots.
`build_memory_report` returns an owned C3 `String` report for application
logging; the caller releases it according to normal C3 string ownership.
Quiesce allocation mutation when an exact diagnostic snapshot matters.

`cmd_begin_label` and `cmd_end_label` add nested GPU command labels when native
debug-utils support is active. They are harmless no-ops otherwise and remain
subject to command token confinement.

Debug names are copied or retained according to each create descriptor's
source contract and are exposed only for diagnostics. Full-validation device
teardown reports leaks and partial/device-loss state but still rejects live
public children rather than destroying them silently.

## Fault behavior

- unavailable platform/presentation capability: `UNSUPPORTED_FEATURE`;
- invalid or lost native surface: `SURFACE_LOST`;
- changed surface extent or compatibility: `SWAPCHAIN_OUT_OF_DATE`;
- finite acquire/present progress timeout: `WAIT_TIMEOUT`;
- live acquisition, child, or queued use: `RESOURCE_IN_USE` or `DEVICE_BUSY`;
- fixed swapchain table exhausted: `SLOT_TABLE_FULL`; and
- lost device: `DEVICE_LOST`.

On retryable faults, public handles and unconsumed readiness remain valid as
documented by the failing operation.
