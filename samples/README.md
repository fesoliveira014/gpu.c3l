# gpu.c3l samples

Standalone consumers of `gpu.c3l`. Each sample owns its shaders under
`<sample>/shaders/`.

## Build order

Shaders compile to SPIR-V first (the `.spv` is `$embed`ed into the sample at
compile time and is gitignored — the `.glsl` is the source of truth):

```sh
sh scripts/build_shaders.sh            # from the repo root; requires glslc on PATH
```

Then build a sample target from this directory:

```sh
cd samples
c3c build root_pointer_compute
```

## Running

Headless samples need a Vulkan 1.3 loader. On a machine without a hardware ICD,
pin lavapipe:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json ./build/root_pointer_compute
```

## Samples

- `root_pointer_compute` — headless. Proves the root-pointer shader ABI: a
  compute shader reaches its input/output buffers through GPU addresses carried
  in a `ComputeRoot` struct pushed as a single 64-bit root pointer, with no
  descriptor sets. Prints `PASS`/`FAIL` and returns a matching exit code.
  Verification is by explicit fault-return plus Vulkan validation.

## Note: c3c 0.8.0 codegen SIGSEGV

M9 hit a c3c 0.8.0 codegen crash (SIGSEGV, no diagnostic, no object). The slot
tables were monomorphized into concrete per-type tables (no generic `gpu::slots`
module, matching c3vq), which made this sample and the `vk_root_pointer` /
`vk_command` test targets build reliably with safety ON. However `test
vk_bootstrap` still crashes, so the generic module is not the sole cause — the
trigger is an accumulation/interaction in the `gpu::vk` module's codegen that
resisted reduction to a minimal standalone case. Full investigation notes and
the reliable reproduction are in `scripts/c3c_bug_repro/REPRO.md`.
