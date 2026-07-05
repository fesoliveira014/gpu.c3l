# gpu.c3l

[![ci](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml/badge.svg)](https://github.com/fesoliveira014/gpu.c3l/actions/workflows/ci.yml)

A C3 GPU programming library (module `gpu`, Vulkan 1.3 backend). Samples live
in [gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples).

## Quick start (linux-x64)

```sh
git clone --recursive https://github.com/fesoliveira014/gpu.c3l
cd gpu.c3l
./scripts/gen_abi.sh --check && ./scripts/build_shaders.sh
c3c test unit --path test        # pure CPU
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json c3c test vk_bootstrap --path test
```

c3c 0.8.0 and `glslc` on PATH; any Vulkan 1.3 ICD to run (lavapipe works
headless). Windows setup: `docs/platforms_and_dependencies.md` §8.

# Documentation artifact set

This folder contains the generated documentation set for the `gpu.c3l` C3 library architecture.

## Files

- `master.md` — master architecture and document map.
- `docs/document_index.md` — reading order and maintenance rules.
- `docs/architecture.md` — architecture, modules, object model, resources.
- `docs/api.md` — public API shape and usage examples.
- `docs/memory.md` — VMA-backed memory model and arenas.
- `docs/shader_abi.md` — root pointer ABI and shared layout rules.
- `docs/vulkan_backend.md` — Vulkan backend implementation plan.
- `docs/testing.md` — test matrix and verification rules.
- `docs/style.md` — C3 style and project conventions.
- `docs/milestones.md` — milestone plan with deliverables and acceptance criteria.
- `docs/platforms_and_dependencies.md` — dependency, platform, and package setup.
- `docs/samples.md` — samples design (sources live in the `gpu.c3l-samples` repository).
