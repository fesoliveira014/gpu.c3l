# Strict GPU Profile Requirements

The strict GPU profile initiative stabilizes the current library, preserves it as `gpu::compat`, and then rebuilds `gpu` around the [strict GPU profile](../../strict_gpu_profile.md).

## Goal

Provide a small GPU-shaped API based on addresses, raw descriptor indices, resource-agnostic hazards, transient commands, and minimal pipeline identity without losing the current Vulkan API as an explicit compatibility path.

## Deliverables

- A coherent, tested, documented, and benchmarked baseline of the current architecture.
- The stabilized current API under `gpu::compat`, initialized only through `gpu::compat::create_device`.
- A strict `gpu` profile with distinct types and no compatibility fallback.
- Placement-first memory and caller-owned resource placement.
- Raw contiguous descriptor heaps with CPU ownership separate from shader indices.
- Resource-agnostic synchronization with no public layouts or resource lists.
- Pipelines keyed only by compiled state, with specialization and dynamic state elsewhere.
- GPU-generated per-command root data where the strict capability is available.
- Strict rasterization, presentation, tests, samples, documentation, and benchmarks.

## Non-goals

- Vulkan 1.2 support in the strict profile.
- Silent strict-to-compatibility fallback.
- Shared device, resource, command, or pipeline types between profiles.
- A renderer, render graph, material system, asset system, or windowing dependency.
- User-managed descriptor sets or queue-family ownership in strict `gpu`.
- Preserving current source compatibility in strict `gpu`.

## User-visible behavior

- Importing `gpu` may expose `gpu::compat` through C3 recursive imports but performs no runtime initialization.
- `gpu::create_device` requires the strict semantic profile and fails deterministically when it is unavailable.
- `gpu::compat::create_device` explicitly initializes the compatibility profile.
- The compatibility profile preserves stabilized current behavior, except for documented correctness fixes.
- Strict buffers are address ranges; strict textures use caller-owned placement.
- Strict shaders consume GPU addresses and raw descriptor indices.
- Strict barriers express execution and memory hazards without naming resources.
- Strict command records are one-shot.
- Optional strict capabilities never change the meaning of an existing operation.

## Acceptance checks

- The current architecture passes its documented test and sample matrix before the profile split.
- Benchmark baselines exist for allocation, descriptors, recording, submission, pipelines, barriers, and indirect work.
- Strict and compatibility imports, builds, device creation, and smoke tests are independent.
- Compatibility state does not exist until explicit compatibility device creation.
- Strict values cannot be passed to compatibility APIs or the reverse.
- Generated strict API documentation contains no Vulkan types, image layouts, descriptor modes, queue families, backend dispatch, or backend state.
- Strict device creation never selects compatibility behavior.
- Shader ABI checks pin address, index, descriptor-range, and root-record layouts.
- All public failure paths return documented C3 faults.
- The strict sample path is canonical when the initiative is complete.
