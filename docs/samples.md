# gpu.c3l Samples

Samples live in their own repository:
[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples). It
vendors this library as a pinned submodule (`lib/gpu.c3l`) and consumes it
exactly as an external project would — the genuine consumer path.

## Baseline

The samples repository contains 18 samples and `shared_selftest`. Its root
README is the authoritative target index, capability matrix, and smoke-command
reference. CI checks generated ABI files, builds shaders, runs all nine
headless/helper targets, and runs all ten windowed targets under xvfb/lavapipe
with bounded frame counts.

## Conventions

- Windowed samples own SDL3: they depend on package `sdl3` and `import sdl`;
  headless samples touch neither.
- Samples are consumers of the public `gpu` API only — no `vk::` or `vma::`
  in sample code.
- Each sample owns its shaders (`<name>/shaders/`) and includes the
  library's published ABI includes.
- Screenshots go through the shared capture path (`shared/screenshot.c3` +
  `shared/png.c3`).

## Ownership

`gpu.c3l-samples/project.json` owns the target list. The root README owns
capability and smoke details; per-sample READMEs own behavior and optional
flags.
