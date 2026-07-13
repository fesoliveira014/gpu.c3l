# Strict GPU Profile

This document defines the target architecture for `gpu`, following [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api). The strict profile is GPU-shaped: addresses, raw descriptor indices, resource-agnostic hazards, transient commands, and minimal pipeline identity. Vulkan 1.2 compatibility remains secondary, but work starts by stabilizing the current architecture so its behavior can be preserved deliberately.

Implementation documents: [requirements](specs/strict-gpu-profile/requirements.md), [design](specs/strict-gpu-profile/design.md), and [tasks](specs/strict-gpu-profile/tasks.md).

## Profile boundary

- `gpu` contains the strict profile.
- `gpu::compat` contains the compatibility profile.
- Importing `gpu` may expose `gpu::compat`, but must not initialize it.
- `gpu::create_device` creates only a strict device.
- `gpu::compat::create_device` explicitly initializes compatibility state.
- Strict and compatibility devices, resources, commands, and pipelines are distinct types.
- Values are shared only when their meaning is identical.
- Neither profile may silently fall back to the other.

The strict API must not expose Vulkan handles, extensions, layouts, descriptor modes, queue families, or backend selection.

## Target model

### Memory

- Allocations own memory and expose size, alignment, properties, GPU address, and an optional CPU mapping.
- Buffers are address ranges, not public binding objects.
- Texture requirements are queried before creation. Textures use caller-provided placement and do not own their allocation.
- Arenas, upload rings, readback, and deferred release build on the placement primitives.
- Live allocations with exposed addresses do not move. Relocation requires explicit reference repair.

### Shader data and descriptors

- Draw and dispatch root data is a GPU address or addressable root record.
- Descriptor heaps allocate contiguous ranges.
- Shader-visible indices have a fixed ABI width and contain no generation data.
- CPU ownership and stale-handle detection use separate tokens.
- Resource and sampler heaps are separate when required.
- Index reuse waits for every referencing submission to complete.

The strict profile has one descriptor model. Descriptor-set and descriptor-indexing choices remain in `gpu::compat`.

### Synchronization

- Barriers contain source and destination stages plus memory hazards.
- Barriers contain no resource handles, image layouts, or queue-family data.
- Copy, clear, rasterization, presentation, and host operations provide the resource intent the backend needs.
- Render-pass boundaries imply no synchronization.
- Backend compression and presentation state remain private.

The hazard vocabulary must cover shader data, indirect arguments, descriptors, color, depth, transfers, and host access.

### Commands and pipelines

- Command objects are one-shot and invalid after submission or abandonment.
- Draw and dispatch commands bind root addresses directly.
- Indirect work can receive per-command root data without a CPU loop.
- Unsupported GPU-generated root data is an explicit semantic capability, not an emulated API path.
- Pipeline identity contains only state that affects compiled GPU code.
- Frequently changed raster, depth, stencil, blend, topology, and attachment state is dynamic.
- Shader specialization replaces avoidable pipeline permutations.

### Rasterization and presentation

Textures remain explicit where the CPU must identify attachments, copies, or presentation images. Their public state is limited to stable GPU semantics such as dimensions, format, usage, and placement.

Presentation consumes strict textures without changing the memory, descriptor, synchronization, or pipeline model.

## Strict Vulkan backend

The backend implements semantic requirements. Extension names and fallback paths remain private.

| Requirement | Preferred implementation |
|---|---|
| Address-based buffer commands and global hazards | [`VK_KHR_device_address_commands`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_device_address_commands.html) |
| Raw descriptor heaps | [`VK_EXT_descriptor_heap`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_descriptor_heap.html) |
| Layout-free image use | [`VK_KHR_unified_image_layouts`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_unified_image_layouts.html) |
| Texture requirements before creation | Vulkan 1.3 maintenance functionality, including [`vkGetDeviceImageMemoryRequirements`](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceImageMemoryRequirements.html) |
| Small pipeline keys | [`VK_EXT_extended_dynamic_state3`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_extended_dynamic_state3.html) and related dynamic state |
| GPU-generated root data | [`VK_EXT_device_generated_commands`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_device_generated_commands.html) |

Strict device creation probes every required semantic and returns a deterministic fault when one is missing. Optional semantics use capability flags and never alter existing operation meanings.

The compatibility backend may use Vulkan 1.2 descriptor indexing, explicit buffer and image objects, image layouts, and conventional indirect draw identifiers. These choices must not constrain strict types.

## Work sequence

Do not split the profiles while current contracts are still moving. First make the existing API coherent, correct, documented, and measurable. Then move that stable model into `gpu::compat` with minimal semantic change. Only after the compatibility baseline is isolated should `gpu` be reshaped into the strict profile.

This order separates existing correctness work from namespace migration and prevents compatibility requirements from shaping the strict API.

## Adoption order

### 1. Stabilize the current architecture

- Reconcile the public API, architecture documents, backend behavior, tests, and samples.
- Resolve correctness gaps in device ownership, resource lifetime, command state, synchronization, descriptors, and swapchains.
- Remove contradictory, dead, or unnecessarily backend-shaped public contracts.
- Make the existing test matrix validation-clean and keep the canonical samples working.
- Record allocation, descriptor, recording, submission, and pipeline baselines.
- Avoid profile moves and strict ABI changes during this step.

Exit when the current API is a coherent, tested, documented, and measurable baseline.

### 2. Establish `gpu::compat`

- Move the stabilized current API into `gpu::compat` with minimal behavioral change.
- Give the compatibility profile distinct device, resource, command, and pipeline types.
- Keep imports free of runtime side effects and require explicit compatibility device creation.
- Preserve the working Vulkan 1.3 path before adding Vulkan 1.2 coverage.
- Add migration, compile, smoke, and sample tests for the compatibility surface.

Exit when existing consumers have a tested migration path and compatibility behavior is isolated from strict design.

### 3. Lock the strict profile boundary

- Introduce the strict `gpu` type family without aliases to compatibility objects.
- Share values only when their semantics are identical.
- Replace backend identity and descriptor modes with strict semantic capabilities.
- Separate required semantics from optional acceleration.
- Keep the device opaque; expose immutable limits and capabilities.
- Define faults for unsupported hardware, invalid placement, exhaustion, and lifetime errors.

Exit when both profiles initialize independently, strict values cannot cross into compatibility APIs, and strict paths require no Vulkan knowledge.

### 4. Make placement primary

- Add allocation and address-range primitives.
- Add texture requirement queries and placed texture creation.
- Rebuild arenas, upload, and readback on these primitives.
- Define relocation rules for escaped addresses.

Exit when resource ownership requires no hidden allocation.

### 5. Adopt raw descriptor heaps

- Separate shader indices from CPU ownership tokens.
- Add contiguous resource and sampler range allocation.
- Define writes, deferred reuse, exhaustion, and stale-token behavior.
- Update the shader ABI generator and includes.

Exit when strict shaders use one fixed descriptor ABI.

### 6. Use resource-agnostic hazards

- Define the complete stage and access vocabulary.
- Replace resource barriers and image transitions.
- Remove public layout and queue-ownership concepts.
- Validate indirect, descriptor, depth, color, transfer, and host hazards.

Exit when synchronization never names a resource.

### 7. Minimize pipeline identity

- Classify each field as compiled, specialized, or dynamic.
- Remove dynamic and specialized state from cache keys.
- Add command operations for dynamic state.
- Measure pipeline count, cache hit rate, and creation latency.

Exit when pipeline identity represents compiled GPU code.

### 8. Complete GPU-driven work

- Define direct and indirect root-record layouts.
- Add address-based indirect draw and dispatch.
- Implement GPU-generated per-command root data where supported.
- Test multi-draw, multi-dispatch, and GPU-produced arguments.

Exit when generated work needs neither a CPU loop nor a compatibility draw identifier.

### 9. Rebuild rasterization and presentation

- Create renderable textures over explicit placement.
- Keep attachment selection in commands and hazards outside render-pass boundaries.
- Present strict textures without exposing swapchain image state.
- Verify resize, recreation, and device-loss cleanup.

Exit when compute, rasterization, and presentation share one memory and hazard model.

### 10. Migrate validation and examples

- Split samples into strict and compatibility entry points.
- Make strict compute, rasterization, indirect, upload, readback, and presentation samples canonical.
- Keep Vulkan 1.2 compatibility smoke samples.
- Benchmark allocation, descriptors, recording, submission, pipelines, and indirect workloads.
- Scan generated strict API documentation for backend leakage.

Exit when strict usage is the default learning path and regressions are measurable.

### 11. Remove transitional strict APIs

- Delete strict aliases for compatibility resources, barriers, layouts, and modes.
- Remove obsolete shader ABI forms after migration.
- Re-run public symbol, documentation, ABI, and lifetime checks.

Exit when `gpu` can be used without learning the compatibility model.

## Completion criteria

- The pre-split architecture has a recorded passing test, sample, documentation, and benchmark baseline.
- The stabilized current behavior is available through `gpu::compat`, except for explicitly documented corrections.
- Importing `gpu` performs no runtime initialization.
- Compatibility state exists only after `gpu::compat::create_device`.
- Strict device creation never selects compatibility behavior.
- Strict documentation exposes no Vulkan types, layouts, descriptor modes, queue families, or backend state.
- Shader buffer references are addresses.
- Shader descriptors use raw contiguous indices; ownership metadata remains CPU-only.
- Barriers contain stages and hazards but no resources.
- Command objects enforce one-shot lifetime.
- Pipeline keys exclude dynamic and specialized state.
- Indirect root-data support has no semantic fallback.
- Strict and compatibility smoke tests build independently.
- ABI tests validate layouts, index width, address width, and root records.
- Benchmarks cover the usage patterns the architecture is designed to make cheap.
