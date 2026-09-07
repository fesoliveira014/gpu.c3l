# Utilities

`gpu::util` provides optional helpers over the public `gpu` API. They ship in
the same `gpu.c3l` bundle, require no extra dependency, and create nothing on
import. Opt in by calling the helper; use its objects with ordinary GPU calls.

- [Device context](device_context.md): create a runtime, select a supported
  adapter, create a device and command allocator, and optionally add a surface
  and swapchain. Work, native windows, and shutdown synchronization remain
  application-controlled.

See the [API reference](../api/index.md) for individual operations and
[architecture](../architecture.md) for ownership and threading rules.
