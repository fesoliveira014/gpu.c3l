# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

`gpu.c3l` — a C3 GPU programming library. Public module `gpu`; Vulkan 1.3 backend module `gpu::vk`. Targets **C3 0.8.0** (pre-1.0; syntax changes between releases — use the C3 skills below / bundled C3 docs, don't rely on memory).

## C3 skills — use when planning, reviewing, or implementing

C3 is pre-1.0; invoke the bundled skills instead of relying on memory:

- **`c3-expert`** — language reference: syntax, semantics, the `c3c` 0.8.0 compiler, version differences. Any C3 question, build error, or code you write/review. Verify snippets with `c3c` (on PATH; version pinned in `.Codex/c3-skill.json`).
- **`c3-style`** — house conventions & idiom: naming, definition order, optional/fault error handling, anti-patterns to flag. Use when writing, reviewing, or refactoring `.c3`/`.c3i`.
- **`c3-bindings`** — `.c3l` binding conventions (`extern fn`, `@cname`, opaque-vs-exposed structs, backend sub-modules). Use when touching the `vk`/`vma`/`sdl3` bindings or wrapping any C library.

For exact syntax/version questions, `c3-expert` wins; for idiom, `c3-style`; for binding authoring, `c3-bindings`.

## Project documentation

The implementation lives under `gpu/`; the Vulkan backend is under `gpu/vk/`. Before changing a contract, read `docs/document_index.md` and the relevant topic document:

- `docs/architecture.md` — module split, handle/ABI model
- `docs/api.md` — public API shape
- `docs/memory.md` — VMA allocator, frame/persistent arenas
- `docs/shader_abi.md` — root-pointer ABI, std430 struct layout
- `docs/vulkan_backend.md` — backend strategy
- `docs/style.md` — full code style guide (source of truth)
- `docs/testing.md` — verification commands and coverage

## Code style (see `docs/style.md` for the rest)

- **Naming**: `snake_case` for vars, fields, params, **and functions**; `PascalCase` for structs/enums/typedefs; `SCREAMING_SNAKE_CASE` for constants and enum values; files `snake_case.c3`.
- **Errors**: C3 optionals/faults (`create_buffer(...) -> BufferHandle?`), never exceptions or null. Use named faults (`INVALID_HANDLE`, `ARENA_FULL`, …).
- **Lifecycle**: free functions `create_x()` / `destroy_x()` for project-owned resources, not `X.create` / `X.destroy`.
- **Definition order in a file**: typedefs → aliases → constants → enums/bitstructs → structs → struct methods → free functions.
- **Formatting**: K&R braces; calls with 4+ args use named arguments, one per line, trailing comma. No auto-formatter — hand-format to the style guide.
- **Comments**: keep to a minimum. Code should be self-documenting. Acceptable on non-trivial logic and a one-or-two-sentence doc on a method; avoid superfluous or over-explanatory comments that restate the code.
- No development labels in identifiers, file names, or test names.

## Architecture rules

- Public API is **GPU-shaped, not Vulkan-shaped**: keep `vk::` and `vma::` types out of public signatures.
- Use strongly-typed handles, not raw `int`/`uint`/`ulong`.

## Build & deps

- Built with `c3c` 0.8.0. `manifest.json` provides `gpu`; native libraries are resolved from dependency manifests.
- Backend needs a Vulkan 1.3 loader + VMA. SDL3 is used only by the separate `gpu.c3l-samples` repository, not this library.
- Initialize dependencies with `git submodule update --init --recursive`.
- Run the core checks from the repository root:

  ```sh
  c3c test unit --path test/cpu
  c3c test shader_abi --path test/cpu
  c3c build smoke --path test
  ```

- Vulkan targets also require the VMA static library described in `docs/platforms_and_dependencies.md`.
