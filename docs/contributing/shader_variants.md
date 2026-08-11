# Shader variants

`gpu.c3l` has no shader-variant pressure. The repository expresses variation as
distinct shaders and as generated scalar constants, and the two mechanisms that
would otherwise benefit from SPIR-V specialization constants — workgroup size
and geometry-type branching — are respectively already generated at ABI
generation time and not expressible as scalars. `ShaderDesc` carries `spirv`,
`entry_point`, and `debug_name`; no specialization field is needed.

## Scope and counting rule

A shader has more than one variant only when one source is built more than once
to produce modules that differ. `scripts/build_shaders.py` compiles one output
per input with a fixed command — `glslc -fshader-stage=<stage>
--target-env=vulkan1.3 -I include/shaders` — and assembles `.spvasm` one-to-one
through `spirv-as`. There is no `-D`, no `-O`, and no permutation matrix.
Every variant count below is therefore 1.

Two consequences follow:

- Every `#ifdef`, `#ifndef`, and `#define` in `include/shaders`,
  `test/shaders`, and `examples/getting_started/shaders` is an include guard or
  a function-like macro. No preprocessor symbol selects behavior.
- Repeated references to one `.spv` in C3 are reuse of the same module, not
  rebuilds. `root_pointer.comp.spv` is referenced 13 times and
  `offscreen.vert.spv` 9 times, always with identical behavior.

There is no specialization machinery to inventory: no `constant_id`, no
`local_size_x_id`, no `OpSpecConstant`, and no `vk::SpecializationInfo` use
anywhere outside the `vk` bindings themselves.

## How variation is expressed today

- **Generated scalar constants.** A `.abi` schema under `test/abi` declares a
  constant once; `tools/gen_shader_abi` emits it as a GLSL `const uint` and as
  a matching C3 `const uint`. Host and shader cannot disagree.
- **Separate entry points.** `ShaderDesc.entry_point` selects one entry from a
  multi-entry module.
- **Separately compiled binaries.** One source, one module, one pipeline.
- **Root data.** Runtime values travel through the fixed root push ABI.

## Inventory

Rows cover every family where variance plausibly exists; the remainder is one
counted line. Sizes are measured artifacts, not estimates.

| Shader and entry point | Variants | Reason | Current implementation | Binary and pipeline cost | Candidate scalar constants | Runtime-data alternative |
| --- | --- | --- | --- | --- | --- | --- |
| `heap_sample.comp`, `heap_write.comp`, `overlap.comp`, `root_pointer.comp`; `main` | 1 each | Workgroup size is a shared scalar: `HEAP_TILE = 8u`, `OVERLAP_WORKGROUP = 64u`, `ROOT_POINTER_WORKGROUP = 64u` | Generated `const uint` in `test/shaders/generated/*_abi.glsl`, mirrored by `test/src/bindless_abi.c3`, `overlap_abi.c3`, `root_pointer_abi.c3` | 4 modules, 12,848 bytes; one pipeline per creation site | `local_size_x_id` and friends — the canonical use case | Not applicable: generation pins one value on both sides, and dispatch group counts already come from root data |
| `heap_write.comp` vs `heap_volume_write.comp`; `heap_sample.comp` vs `heap_volume_sample.comp`; `main` | 1 each | 2D versus 3D texture access | Separate sources calling `store_storage_texture`/`sample_texture_2d` versus `store_storage_texture_3d`/`sample_texture_3d`, over distinct root structs | 2 additional modules, 13,496 bytes | None — the difference is descriptor type and image instruction, not a scalar value | Not applicable |
| `ray_query_triangle.comp` vs `ray_query_aabb.comp`; `main` | 1 each | Triangle versus procedural AABB intersection | Separate sources; the AABB shader adds an intersection function and a candidate-confirmation loop around `rayQueryProceedEXT` | 2 modules, 10,172 bytes | None sufficient — ray flags and cull mask do differ, but an added intersection function and different committed-intersection handling remain | Root data could select a branch, but the shaders also differ in committed-intersection handling |
| `graphics_size_8.vert`, `graphics_second_offset_4.vert`, `graphics_reversed_members.vert`; `main` | 1 each | Push-constant block layout: member count and `OpMemberDecorate ... Offset` | Three hand-authored `.spvasm` fixtures assembled one-to-one | 3 modules, 1,064 bytes | None — specialization cannot alter a push-constant ABI layout | Not applicable; these exist so reflection rejects them |
| `multi_entry_root.comp`; `good`, `bad_push`, `bad_descriptor` | 1 module | Three distinct interfaces in one module | `ShaderDesc.entry_point` selects the entry at pipeline creation | 1 module, 852 bytes; one pipeline per selected entry | None | The separate-entry-point mechanism is already the alternative, and it is in use |
| Remainder | 1 each | Each is a distinct shader, not a rebuild of another | One source, one module, one entry point | 52 modules, 82,412 bytes | None | Not applicable |

The remainder includes the reflection-rejection fixtures — `bad_binding`,
`bad_ray_query_binding`, `bad_ray_query_set`, `bad_ray_query_type`,
`bad_sampled_3d_binding`, `bad_set`, `bad_storage_3d_binding`,
`bad_unknown_heap_binding`, `ray_bad_root.rgen`, `no_push`, `oversized_push`,
`root_wrong_offset.comp`, and `root_multiple_blocks.comp`. They are inputs that
must fault, not variants of the shaders they resemble.

## Measurements

Toolchain: `glslc` from shaderc 2023.8 (glslang 14.0.0, SPIRV-Tools 2023.6),
`spirv-as` from SPIRV-Tools v2025.1, both targeting `vulkan1.3` with
`-I include/shaders`, no defines and no optimization flags. Compiler versions
and flags are the only settings that affect the artifacts; no device is
involved in producing them.

| Measurement | Value |
| --- | --- |
| GLSL entry-point sources | 57 (56 in `test/shaders`, 1 in `examples/getting_started/shaders`) |
| SPIR-V assembly sources | 7 |
| Include-only headers, never compiled directly | 12 (5 under `include/shaders`, 7 under `test/shaders/generated`) |
| SPIR-V artifacts produced | 64 |
| Total artifact size | 120,844 bytes |
| Largest artifact | `heap_volume_write.comp.spv`, 7,700 bytes |
| Smallest artifact | `graphics_size_8.vert.spv`, 304 bytes |
| Full shader build, wall clock | about 5 s |
| Static pipeline-creation call sites | 107 (62 compute, 43 graphics, 2 ray tracing) |

No module is a duplicate of another. Total shader payload is under 120 KiB and
a full rebuild takes seconds, so duplicated binaries are not affecting build
time, package size, or maintenance.

## Cost of a production feature

Specialization values would have to become part of shader and pipeline identity,
and that is where the work lands.

- `PipelineKey` in `gpu/internal/vk/pipeline_cache.c3` is a fixed-size packed
  struct guarded by `$assert PipelineKey::size == 60`, hashed as a flat byte
  view. Variable-length specialization data cannot live in it; the key would
  need owned bytes and a separate equality path, as
  `OwnedRayTracingPipelineKey` already does for ray tracing.
- `ShaderIdentityInput` and `ShaderStoreEntry` in `gpu/internal/vk/shader.c3`
  deduplicate by SPIR-V hash, entry point, and role. Specialization would have
  to be normalized so caller ordering does not change identity.
- Reflection would need to enumerate specialization IDs and types and fault on
  unknown IDs or type mismatches, alongside the existing root-ABI checks.
- `ShaderDesc` would gain a new authoring concept, with new diagnostics.

## Not measured

Cold pipeline-creation time and generated-code quality on production hardware.
Both require a real driver; a CPU rasterizer would not settle either question.
Neither is needed for the decision, because no candidate variant family exists
to compare.

## Recommendation

Retain the current model. Do not add scalar specialization constants to the
public API, and do not open a follow-up implementation issue.

Revisit when a consuming project presents a shader family whose members differ
only by scalar values, are built from one source more than once, and whose
duplication measurably costs build time, package size, or pipeline count.
