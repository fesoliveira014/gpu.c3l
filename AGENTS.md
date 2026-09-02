# Contributor instructions

`gpu.c3l` is a C3 GPU programming library. The public module is `gpu`; the
private Vulkan 1.3 backend is `gpu::internal::vk`. The project targets
**C3 0.8.3**.

## Before changing code

- Confirm `c3c --version` reports 0.8.3. C3 is pre-1.0; use the bundled
  `c3-expert` skill and local compiler instead of remembered syntax.
- Use `c3-style` when writing or reviewing `.c3`/`.c3i`.
- Use `c3-bindings` when touching `vk`, `vma`, `spvreflect`, or SDL3 bindings.
- Initialize dependencies with `git submodule update --init --recursive`.
- Read [docs/index.md](docs/index.md), the relevant
  [public API page](docs/api/index.md), and the source docstrings before
  changing a public contract.

## Architecture rules

- Keep the public API GPU-shaped. No `vk::` or `vma::` type may appear in a
  public signature.
- Public resources use strongly typed handles, not raw integers.
- Imports are runtime-inert; create calls are transactional; destruction adds
  no hidden wait.
- Preserve explicit ownership, texture state, synchronization, and
  completion-based reuse.
- `GpuAddress`, `TextureIndex`, and `SamplerIndex` are raw shader values, not
  ownership tokens.
- Respect the threading tiers and lock ordering documented in
  [architecture.md](docs/architecture.md).

## C3 style

The full source of truth is [docs/contributing/style.md](docs/contributing/style.md).

- `snake_case` for variables, fields, parameters, functions, and filenames;
  `PascalCase` for types; `SCREAMING_SNAKE_CASE` for constants and enum values.
- Free functions `create_x`/`destroy_x` own project resources.
- Use C3 optionals and named faults, never exceptions, null, or sentinel return
  values for public failures.
- Definition order: typedefs, aliases, constants, enums/bitstructs, structs,
  struct methods, free functions.
- K&R braces. Calls with four or more arguments use named arguments, one per
  line, with a trailing comma.
- Keep comments minimal; public docstrings state contracts rather than restate
  code.
- Do not put development labels in identifiers, filenames, or test names.

## Documentation

Published `README.md` and `docs/` content is for consumers. Keep it
current-state, task-oriented, and backend-neutral. Development decisions,
implementation plans, migration narration, and contributor workflow belong in
OpenSpec changes, source history, or this file.

When the API changes:

- update the owning page under `docs/api/` and the symbol map;
- update getting-started/cookbook examples that use it;
- preserve ownership, fault, concurrency, and call-order details; and
- run the documentation link and symbol-coverage checks described in
  [docs/contributing/testing.md](docs/contributing/testing.md).

## Build and verification

Native libraries are resolved from dependency manifests. The backend requires a
Vulkan 1.3 loader and the vendored VMA static library. SDL3 belongs to the
separate samples repository, not this library.

Run from the repository root:

```sh
python3 scripts/gen_abi.py --check
python3 scripts/build_shaders.py
c3c test unit --path test/cpu
c3c test shader_abi --path test/cpu
c3c build smoke --path test
c3c run smoke --path test
```

See [testing.md](docs/contributing/testing.md) for the full matrix and
[benchmarking.md](docs/contributing/benchmarking.md) for measurement practice.
