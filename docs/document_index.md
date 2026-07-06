# gpu.c3l Documentation Index

This document lists the project documentation set, the intended reading order, and the maintenance rules for each document.

## Reading order

0. `docs/getting_started.md` — consumer walkthrough: fresh machine to a
   running compute program (executed verbatim by the CI docs job).
1. `master.md` — high-level architecture and document map.
2. `docs/architecture.md` — full system architecture.
3. `docs/api.md` — public API and examples.
4. `docs/memory.md` — VMA-backed memory architecture.
5. `docs/shader_abi.md` — shader ABI and generated layout rules.
6. `docs/vulkan_backend.md` — Vulkan backend implementation details.
7. `docs/testing.md` — test matrix and verification rules.
8. `docs/style.md` — coding conventions.
9. `docs/milestones.md` — implementation plan.
10. `docs/platforms_and_dependencies.md` — dependency and platform setup.
11. `docs/samples.md` — sample and windowed test plan.

## Document responsibilities

| Document | Maintained when |
|---|---|
| `master.md` | Any major design direction changes. |
| `docs/getting_started.md` | Toolchain, vendoring flow, or the minimal program's API surface changes — the CI docs job fails when it drifts. |
| `docs/architecture.md` | Layers, modules, object model, or resource lifetime rules change. |
| `docs/api.md` | Public types, descriptors, functions, or examples change. |
| `docs/memory.md` | Memory kind policy, VMA usage, arenas, or allocation behavior changes. |
| `docs/shader_abi.md` | Root pointer ABI, shared structs, layout rules, or shader conventions change. |
| `docs/vulkan_backend.md` | Vulkan feature requirements, descriptor implementation, synchronization, or swapchain policy changes. |
| `docs/testing.md` | Test targets, validation expectations, CI, or sample verification changes. |
| `docs/style.md` | Naming, formatting, module, error-handling, or documentation conventions change. |
| `docs/milestones.md` | Work breakdown, acceptance criteria, or release scope changes. |
| `docs/platforms_and_dependencies.md` | Dependency names, binding versions, manifests, or platform support change. |
| `docs/samples.md` | Samples are added, removed, renamed, or promoted to release gates. |

## Stability tiers

| Tier | Meaning | Examples |
|---|---|---|
| Required | Public API and backend behavior depend on it. | Root pointer ABI, explicit barriers, VMA-backed memory. |
| Preferred | Default implementation path, but may have fallback. | Descriptor buffer fast path, SDL3 samples. |
| Deferred | Documented but not first-release blocking. | Defragmentation, Slang support, multi-backend support. |

## Documentation rules

- Public API requirements belong in `docs/api.md` and doc comments in source.
- Backend behavior belongs in `docs/vulkan_backend.md`, not in public API docs unless it affects users.
- Memory lifetime and mapping rules belong in `docs/memory.md`.
- Shader layout rules belong in `docs/shader_abi.md`.
- Milestone tags belong in docs and planning only, not in code identifiers, file names, tests, or comments.
- The master document should summarize; topic documents should specify.

## Artifact packaging

A release documentation bundle should include:

```text
master.md
docs/document_index.md
docs/architecture.md
docs/api.md
docs/memory.md
docs/shader_abi.md
docs/vulkan_backend.md
docs/testing.md
docs/style.md
docs/milestones.md
docs/platforms_and_dependencies.md
docs/samples.md
```
