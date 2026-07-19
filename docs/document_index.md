# Documentation

Start with:

- [Getting started](getting_started.md) for toolchain setup and a minimal compute program.
- [Cookbook](cookbook.md) for focused usage patterns.
- [Limitations](limitations.md) for unsupported features, limits, and driver quirks.
- [Samples](samples.md) for runnable examples in `gpu.c3l-samples`.

## Reference

| Document | Scope |
|---|---|
| [Architecture](architecture.md) | Modules, objects, caller-owned lifetimes, commands, descriptors, and swapchains. |
| [Strict GPU architecture](strict_gpu_profile.md) | Target runtime, device, memory, command, and compatibility extension model. |
| [Public API](api.md) | Public types, functions, faults, and examples. |
| [Memory](memory.md) | Owning allocations, non-owning spans, mapping/address queries, visibility, and transfers. |
| [Shader ABI](shader_abi.md) | Root pointers, generated layouts, descriptor indices, and reflection. |
| [Threading](threading.md) | Entry-point tiers, lock order, command recording, and completion-driven reuse. |
| [Vulkan backend](vulkan_backend.md) | Vulkan 1.3 requirements and backend implementation. |
| [Platforms and dependencies](platforms_and_dependencies.md) | Supported targets, native dependencies, and setup. |
| [Performance](performance.md) | Reproducible benchmark method, baseline, and usage guidance. |
| [Testing](testing.md) | Test targets, validation policy, CI, and release checks. |
| [Style](style.md) | C3 naming, formatting, errors, comments, and dependency hygiene. |

## Planning

- [Strict GPU architecture requirements](specs/strict-gpu-profile/requirements.md) for scope and acceptance criteria.
- [Strict GPU architecture design](specs/strict-gpu-profile/design.md) for module boundaries and implementation order.
- [Strict GPU architecture tasks](specs/strict-gpu-profile/tasks.md) for implementation status and verification commands.

## Maintenance

- Source doc comments define public contracts; `docs/api.md` explains usage.
- Keep Vulkan and VMA implementation details out of public signatures and consumer examples.
- Update the relevant topic document with every contract or behavior change.
- Describe current behavior, not project-management history.
- Keep examples executable and link to a complete sample when a recipe would otherwise become long.
