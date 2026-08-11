# Documentation

Use this documentation as a consumer guide to the shipped `gpu.c3l` API.
Source docstrings remain the authority for exact signatures.

## Start and learn

- [Getting started](getting_started.md) — vendor the library, run a minimal
  compute program, and build a small SDL3 triangle.
- [Architecture](architecture.md) — goals, ownership, memory, commands,
  synchronization, threading, presentation, and platform boundaries.
- [Features and limitations](features_and_limitations.md) — supported
  workloads, required capabilities, fixed limits, and known environment
  behavior.
- [Shader ABI](shader_abi.md) — root pointers, generated std430 layouts, and
  bindless heap values.
- [Cookbook](cookbook.md) — focused recipes for common operations.

## Look up an API

Start at the [public API index](api/index.md), then choose a domain:

- [Runtime and devices](api/runtime_and_devices.md)
- [Memory and resources](api/memory_and_resources.md)
- [Shaders and pipelines](api/shaders_and_pipelines.md)
- [Commands and rendering](api/commands_and_rendering.md)
- [Synchronization and submission](api/synchronization_and_submission.md)
- [Presentation and diagnostics](api/presentation_and_diagnostics.md)

## Troubleshoot

Check [features and limitations](features_and_limitations.md) for capability
requirements, returned faults, fixed capacities, and known driver or
environment symptoms. Use `ContractValidation.FULL` during development and
attach the structured debug callback when diagnosing ownership or call-order
faults.

## Contribute

- [Style](contributing/style.md)
- [Testing](contributing/testing.md)
- [Benchmarking](contributing/benchmarking.md)
- [Graphics pipeline identity](contributing/pipeline_identity.md)

Repository workflow and contributor setup live in
[`AGENTS.md`](../AGENTS.md), outside the published consumer guide.
