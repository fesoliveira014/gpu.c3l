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
extent and never waits for outstanding use. Follow the lifecycle order below.
If it still returns `RESOURCE_IN_USE` or `DEVICE_BUSY`, resolve an outstanding
acquisition or image reference before retrying.

`destroy_swapchain` releases its surface retain only when no image is acquired
or pending. It inserts no hidden wait.

`wait_swapchain_presentations` explicitly waits for every presentation pending
when the call begins. The device must own the live swapchain, and the caller
must externally synchronize acquire, present, resize, destroy, and other work
on that swapchain. Keep the surface and its native display or window alive and
pump platform progress while using a blocking timeout.

Use this order before a lifecycle change:

1. establish render completion with `wait_completion`;
2. call `wait_swapchain_presentations`;
3. call `resize_swapchain` or `destroy_swapchain`.

`timeout_ns == 0` performs a nonblocking query. `TIMEOUT_INFINITE` waits
without a deadline. With presentations still pending, `WAIT_TIMEOUT` preserves
the handle and pending state, emits no diagnostic, and is safe to retry.
Out-of-host/device memory, `DEVICE_BUSY`, `DEVICE_LOST`, and `BACKEND_ERROR`
also preserve the handle. The wait consumes neither the swapchain handle nor
an acquired image, waits no completion point, and retires no command or view
references. With no pending presentations it returns immediately without a
native wait or reset.

`resize_swapchain` and `destroy_swapchain` still insert no hidden wait and
continue to reject outstanding acquisitions, presentations, and image use.

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

## Debug reporting

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
