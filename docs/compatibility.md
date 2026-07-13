# Compatibility Profile

`gpu::compat` preserves the stabilized Vulkan 1.3 API that formerly lived in
`gpu`. It is an explicit profile, not a fallback for strict `gpu`.

## Use

```c3
import gpu::compat;

gpu::compat::DeviceDesc desc = {
    .backend = gpu::compat::BackendKind.VULKAN,
};
gpu::compat::Device device = gpu::compat::create_device(&desc)!;
defer gpu::compat::destroy_device(&device)!!;
```

Importing `gpu` or `gpu::compat` creates no loader, instance, device, allocator,
descriptor storage, or thread. Compatibility initialization begins only in
`gpu::compat::create_device`.

## Contract

The compatibility profile retains:

- one live device per process;
- generation-checked resource and command tokens;
- Vulkan-backed buffer and texture allocation through private VMA state;
- frame, persistent, staging, and readback arenas;
- descriptor indexing by default and opt-in descriptor buffers;
- explicit resource barriers and texture layouts;
- root-pointer draw and dispatch ABI;
- dynamic rendering, timeline semaphores, and synchronization2;
- optional platform-neutral presentation.

Behavior changes are limited to documented correctness fixes. Public
signatures contain no Vulkan or VMA types.

## Migration from the former root API

Migration is mechanical:

```text
import gpu;        -> import gpu::compat;
gpu::Type         -> gpu::compat::Type
gpu::function     -> gpu::compat::function
```

Do not add local aliases back to `gpu`. Strict and compatibility values are
intentionally non-interchangeable; the compiler rejects cross-profile calls.

The shader ABI include paths and generated GLSL names are unchanged. C3 ABI
declarations now live in `gpu::compat`.

## Requirements and limits

The current backend requires Vulkan 1.3 semantics and VMA. Vulkan 1.2 plus
extensions is not yet supported. SDL3 is a sample dependency, not a library
dependency.

See [Getting started](getting_started.md), [API](api.md),
[Limitations](limitations.md), and [Platforms and dependencies](platforms_and_dependencies.md).