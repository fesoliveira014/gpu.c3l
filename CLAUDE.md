# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`gpu.c3l` — a C3 GPU programming library. Public module `gpu`; Vulkan 1.3 backend module `gpu::vk`. Targets **C3 0.8.0** (pre-1.0; syntax changes between releases — use the `c3-expert` skill / bundled C3 docs, don't rely on memory).

## Project state — read docs before writing code

Architecture-first. The design docs are complete; almost no source exists yet (`gpu.c3i` stub + `manifest.json` only). **Before writing or changing source, read `master.md` and `docs/document_index.md`**, then the relevant doc:

- `docs/architecture.md` — module split, handle/ABI model
- `docs/api.md` — public API shape
- `docs/memory.md` — VMA allocator, frame/persistent arenas
- `docs/shader_abi.md` — root-pointer ABI, std430 struct layout
- `docs/vulkan_backend.md` — backend strategy
- `docs/style.md` — full code style guide (source of truth)
- `docs/testing.md`, `docs/milestones.md`

## Code style (see `docs/style.md` for the rest)

- **Naming**: `snake_case` for vars, fields, params, **and functions**; `PascalCase` for structs/enums/typedefs; `SCREAMING_SNAKE_CASE` for constants and enum values; files `snake_case.c3`.
- **Errors**: C3 optionals/faults (`create_buffer(...) -> BufferHandle?`), never exceptions or null. Use named faults (`INVALID_HANDLE`, `ARENA_FULL`, …).
- **Lifecycle**: free functions `create_x()` / `destroy_x()` for project-owned resources, not `X.create` / `X.destroy`.
- **Definition order in a file**: typedefs → aliases → constants → enums/bitstructs → structs → struct methods → free functions.
- **Formatting**: K&R braces; calls with 4+ args use named arguments, one per line, trailing comma. No auto-formatter — hand-format to the style guide.
- No milestone names in identifiers, file names, or test names.

## Architecture rules

- Public API is **GPU-shaped, not Vulkan-shaped**: keep `vk::` and `vma::` types out of public signatures.
- Use strongly-typed handles, not raw `int`/`uint`/`ulong`.

## Build & deps

- Built with the `c3c` compiler (C3 0.8.0). `manifest.json` provides `gpu`; native libs live under `linked-libs/<target>/`. No build/test scripts exist yet.
- Backend needs a Vulkan 1.3 loader + VMA. SDL3 is for windowed samples/tests only, not the core library.
- Primary dev target: `linux-x64` (then `windows-x64`).
