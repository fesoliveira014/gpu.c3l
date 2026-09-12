# Documentation

Consumer guide to `gpu.c3l`. Docstrings in `gpu/gpu.c3`, `gpu/gpu.c3i`,
and `gpu/util/device_context.c3` are the authority for exact signatures.

## Learn

1. [Concepts](concepts.md): handles, memory, root pointers, indices,
   commands, completion, barriers, queues, presentation, validation. For
   readers new to explicit GPU APIs; skip it if you have used Vulkan or
   D3D12.
2. [Getting started](getting_started.md): install, a compute program, a
   windowed triangle.
3. [Architecture](architecture.md): the contract for objects, ownership,
   memory, commands, synchronization, threading. Read this before designing
   around the API.
4. [Shader ABI](shader_abi.md): root pointers, std430 records, heap indices,
   the schema generator. Read this before writing shaders.
5. [Cookbook](cookbook.md): recipes for uploads, readback, depth, indirect
   draws, multiple queues, threads, timestamps, resize, ray tracing.
6. [Features and limitations](features_and_limitations.md): what is and is
   not provided, required device profile, fixed limits.

## Reference

The [API index](api/index.md) lists every public symbol by domain and
explains the shared conventions and fault set. Domain pages:

- [Runtime and devices](api/runtime_and_devices.md)
- [Memory and resources](api/memory_and_resources.md)
- [Shaders and pipelines](api/shaders_and_pipelines.md)
- [Commands and rendering](api/commands_and_rendering.md)
- [Synchronization and submission](api/synchronization_and_submission.md)
- [Presentation and diagnostics](api/presentation_and_diagnostics.md)

[Utilities](util/index.md) compose the same API for optional convenience.
[Device context](util/device_context.md) groups setup and ownership of a runtime,
device, command allocator, and optional surface and swapchain.

## Troubleshoot

Develop with `RuntimeDesc.enable_vulkan_validation = true` and a
[debug callback](cookbook.md#receive-diagnostics). The callback receives native
validation messages and useful library failure details. Applications remain
responsible for valid GPU usage and ordering. Environment-specific symptoms are listed in
[features and limitations](features_and_limitations.md#known-environment-behavior).
