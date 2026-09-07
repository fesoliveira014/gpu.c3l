# Device context

`gpu::util::DeviceContext` groups setup and ownership of a runtime, device,
command allocator, and optional surface and swapchain. It ships in the existing
bundle. Importing `gpu::util` creates nothing; calling `create_device_context`
opts in. Its fields work with the ordinary [GPU API](../api/index.md).

## Functions and types

All symbols below are in `gpu::util`.

| Symbol | Contract |
|---|---|
| `default_device_context_desc()` | Returns a `DeviceContextDesc` with the explicit defaults below. |
| `create_device_context(DeviceContextDesc*)` | Returns `DeviceContext?`; borrows the description for the call and leaves it unchanged. |
| `destroy_device_context(DeviceContext*)` | Returns `void`; attempts every owned release, reports faults, and clears the struct. |
| `SurfaceFactory` | Callback taking `gpu::Runtime*` and `void*` user data, returning `gpu::Surface?`. |
| `AdapterSelector` | Callback taking `gpu::Runtime*`, `gpu::DeviceDesc*` requirements, and `void*` user data, returning `gpu::Adapter?`. |

`DeviceContextDesc` fields:

| Field | Purpose |
|---|---|
| `runtime` | `gpu::RuntimeDesc`: validation, capacities, diagnostics, and application identity. |
| `device` | `gpu::DeviceDesc`: queue and feature requirements; leave `surface` invalid. |
| `command_queue` | `gpu::QueueKind` for the context queue and allocator; must be a required role. |
| `command_allocator` | `gpu::CommandAllocatorDesc` for the one owned allocator. |
| `swapchain` | `gpu::SwapchainDesc`, used when a surface factory is supplied. |
| `create_surface`, `surface_user_data` | Optional `SurfaceFactory` and its borrowed user data. |
| `select_adapter`, `adapter_user_data` | Optional `AdapterSelector` and its borrowed user data. |

`DeviceContext` fields:

| Field | Ownership |
|---|---|
| `runtime` | Owned `gpu::Runtime`. |
| `surface` | Owned `gpu::Surface`, invalid for headless contexts. |
| `adapter` | `gpu::Adapter` borrowed from `runtime`. |
| `device` | Owned `gpu::Device`. |
| `queue` | `gpu::Queue` borrowed from `device`; no separate destruction. |
| `command_allocator` | Owned `gpu::CommandAllocator`. |
| `swapchain` | Owned `gpu::SwapchainHandle`, invalid for headless contexts. |
| `debug_callback`, `debug_user_data` | Saved `gpu::DebugMessageCallback` and borrowed user data for teardown diagnostics. |

Pass a context by pointer. Copies do not create independent owners: do not
destroy copies or independently destroy the context's fields. Resources and
additional command allocators you create through its device remain yours to
release. Native windows and displays remain application-owned.

## Defaults and overrides

`default_device_context_desc()` selects:

- `ContractValidation.FULL`, with native validation layers disabled;
- the public default runtime and command allocator capacities;
- graphics, compute, and transfer queue roles, allowing aliasing;
- a `GRAPHICS` queue and command allocator;
- optional device features disabled;
- FIFO presentation, `BGRA8_UNORM` preference, and the default image count;
- zero width and height, and no callbacks or user data.

Without a surface factory, creation is headless and skips the swapchain.
For presentation, supply the drawable size; zero extent follows the ordinary
swapchain rules and can produce a dormant swapchain. All nested descriptions
remain configurable. For compute-only use, set both `device.queues.required`
and `command_queue`, as below.

A zero-initialized description keeps the nested API's zero semantics, including
`TRUSTED` contract validation. Use the defaults function to request the convenience
baseline. Creation never weakens feature, queue, or capacity requirements.

## Headless use

This function submits an empty command list to demonstrate the ordinary command
lifecycle. Its caller must report a returned fault and terminate the application;
the example only reaches teardown after successful completion. For real work,
record commands before `end_commands` and release application resources after
their last use completes.

```c3
module context_headless;

import gpu;
import gpu::util;

fn void? run_headless() {
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.device.queues.required = { .compute };
    desc.command_queue = gpu::QueueKind.COMPUTE;
    util::DeviceContext context = util::create_device_context(&desc)!;

    gpu::CommandList commands = gpu::begin_commands(&context.command_allocator)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(context.queue, &submit_desc)!;
    gpu::wait_completion(completion)!;

    util::destroy_device_context(&context);
}
```

The [complete bundle consumer](https://github.com/fesoliveira014/gpu.c3l/tree/main/examples/device_context)
shows application-level error reporting. The
[manual compute example](../getting_started.md#step-1-double-an-array-on-the-gpu)
shows resource allocation, shader dispatch, and readback.

## Presentation use

Create the native window before the context and keep it alive through context
teardown. A surface factory adapts the platform's native handles. This Win32
example uses the public surface module; use the corresponding X11 or Wayland
module on those platforms.

Presentation requires the graphics role in `device.queues.required`. Keep the
default graphics context queue for the acquire, submit, and present sequence.

```c3
module context_window;

import gpu;
import gpu::util;
import gpu::surface::win32;

struct NativeWindow {
    win32::InstanceHandle instance;
    win32::WindowHandle window;
}

fn gpu::Surface? create_window_surface(gpu::Runtime* runtime, void* user_data) {
    NativeWindow* native = (NativeWindow*)user_data;
    return win32::create_surface(runtime, native.instance, native.window);
}

fn util::DeviceContext? create_window_context(NativeWindow* window, uint width, uint height) {
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &create_window_surface;
    desc.surface_user_data = window;
    desc.swapchain.width = width;
    desc.swapchain.height = height;
    return util::create_device_context(&desc);
}
```

Use `context.device`, `context.queue`, and `context.swapchain` with the ordinary
[acquire, submit, and present sequence](../api/presentation_and_diagnostics.md).
Resize, event processing, and frame scheduling remain application-controlled.

The factory runs synchronously once, after successful adapter enumeration and
before selection. It must return one fresh surface owned by the supplied runtime,
clean its own partial work on failure, create no unrelated GPU children, and
retain no bootstrap handles. Its user data is borrowed during the call. A foreign
surface is rejected without being destroyed. Empty enumeration fails before the
factory runs. Do not supply `device.surface` yourself.

## Adapter selection and creation faults

Default selection filters adapters with `supports_device_desc`, then prefers
`DISCRETE`, `INTEGRATED`, `VIRTUAL`, `OTHER`, and `SOFTWARE`, in that order.
The first enumerated adapter wins a tie. No supported candidate returns
`UNSUPPORTED_FEATURE`.

An `AdapterSelector` receives the new runtime and a private effective device
description, including the created surface when present. Select an adapter from
that runtime's adapter list. Treat requirements as borrowed input; do not modify
them, retain the pointer, create owners, or publish bootstrap state. The utility
checks adapter provenance and support before device creation. Zero, foreign,
and stale adapter results return
`INVALID_HANDLE`. Callback faults propagate unchanged. Device creation is attempted
once and remains authoritative; a creation fault does not trigger reselection.

A null description, supplied `device.surface`, user data without its callback,
or inconsistent command queue policy returns `INVALID_ARGUMENT`. Nested API
validation, allocation, capacity, and device faults also propagate.

Creation performs no submissions, acquisitions, waits, or background work. If a
stage fails, it attempts reverse cleanup of acquired objects, discards secondary
cleanup faults, and returns the original fault without publishing a context.
Treat creation failure as fatal at application level: report it and terminate.
The utility itself does not abort, exit, retry, or retain recovery state.

## Shutdown

Before calling `destroy_device_context`:

1. Stop producing work and exclude every concurrent use of the context's fields.
2. Discard recording and executable command lists that were not submitted.
3. Wait for completion points covering all submitted work on every used queue.
4. For presentation, resolve any acquired image and call
   `wait_swapchain_presentations` while continuing required platform progress.
5. Release application-owned children, including extra allocators, resources,
   pipelines, views, and allocations.

Then destroy the context, followed by its native window and display objects.
GPU completion alone does not establish presentation completion. See
[presentation shutdown](../api/presentation_and_diagnostics.md) for the platform
progress and acquired-image rules.

Teardown attempts swapchain, allocator, device, surface, and runtime destruction
in that order. It never waits or retries. Every failed release is reported through
the saved debug callback, or stderr when none is configured, and later releases
are still attempted. Diagnostics use `ERROR`, `resource_lifetime`, the failing
operation, and its fault. Callback payloads are synchronous and call-scoped;
callbacks must not reenter GPU operations. Saved callback user data must remain
valid until teardown returns, including reporting a final runtime release fault.

After all attempts, the supplied struct is cleared. A zero context is harmless.
Release failures can leave objects unreleased, but the context and its copies
must never be reused after teardown; there is no partial-context recovery.

Creation and destruction inherit the runtime and surface external synchronization
rules. Independent allocators, recording confinement, and native queue host
synchronization follow the existing [threading contract](../architecture.md#threading).
