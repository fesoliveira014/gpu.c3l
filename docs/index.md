# Documentation

Consumer guide to `gpu.c3l`. Docstrings in `gpu/gpu.c3` and `gpu/gpu.c3i`
are the authority for exact signatures.

## Learn

1. [Getting started](getting_started.md): install, a compute program, a
   windowed triangle. Read this first.
2. [Architecture](architecture.md): objects, ownership, memory, commands,
   synchronization, threading. Read this before designing around the API.
3. [Shader ABI](shader_abi.md): root pointers, std430 records, heap indices,
   the schema generator. Read this before writing shaders.
4. [Cookbook](cookbook.md): recipes for uploads, readback, depth, indirect
   draws, multiple queues, threads, timestamps, resize, ray tracing.
5. [Features and limitations](features_and_limitations.md): what is and is
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

## Troubleshoot

Develop with `full_validation_runtime_desc()` and a
[debug callback](cookbook.md#receive-diagnostics). Ownership and call-order
faults are reported with the operation, the offending field, and the
violated invariant. Environment-specific symptoms are listed in
[features and limitations](features_and_limitations.md#known-environment-behavior).
