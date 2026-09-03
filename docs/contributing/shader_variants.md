# Shader variants

**Question.** Does the repository need SPIR-V specialization constants in
`ShaderDesc`?

**Decision.** No. Variation is expressed as separate shaders and generated
scalar constants.

## Counting rule

A shader has more than one variant only when one source is built more than
once into differing modules. `scripts/build_shaders.py` compiles each input
once with a fixed command (`glslc --target-env=vulkan1.3 -I include/shaders`,
no `-D`, no `-O`) and assembles each `.spvasm` once. Every count is
therefore 1.

Consequences:

- every `#ifdef` and `#define` under `include/shaders`, `test/shaders`, and
  `examples/getting_started/shaders` is an include guard or a function-like
  macro;
- repeated `$embed` of one `.spv` is reuse, not a rebuild;
- there is no `constant_id`, `local_size_x_id`, `OpSpecConstant`, or
  `SpecializationInfo` use outside the `vk` bindings.

## How variation is expressed

| Need | Mechanism |
|---|---|
| shared scalar such as workgroup size | `const` in a `.abi` schema, emitted to GLSL and C3 by the generator |
| different interface in one module | `ShaderDesc.entry_point` |
| different behavior | separate source, separate module |
| runtime values | root data |

## Inventory

| Family | Reason for the pair | Could a scalar constant replace it? |
|---|---|---|
| `heap_sample`, `heap_write`, `overlap`, `root_pointer` (compute) | workgroup size | Yes in principle, but the generated constant already pins one value on both sides and group counts come from root data. |
| `heap_write` vs `heap_volume_write`, `heap_sample` vs `heap_volume_sample` | 2D vs 3D image access | No: different descriptor type and image instruction. |
| `ray_query_triangle` vs `ray_query_aabb` | triangle vs procedural intersection | No: the AABB shader adds an intersection function and a confirmation loop. |
| `graphics_size_8`, `graphics_second_offset_4`, `graphics_reversed_members` (`.spvasm`) | push-block layout fixtures | No: specialization cannot change a push-constant layout. |
| `multi_entry_root.comp` with `good`, `bad_push`, `bad_descriptor` | three interfaces in one module | Entry points already do this. |
| remainder (52 modules) | distinct shaders | Not applicable. |

Reflection-rejection fixtures (`bad_*`, `no_push`, `oversized_push`,
`root_wrong_offset`, `root_multiple_blocks`, `ray_bad_root`) are inputs that
must fault, not variants.

## Size

Measured with glslc from shaderc 2023.8 and SPIRV-Tools 2025.1; sizes will
differ under another toolchain, counts will not.

| Measurement | Value |
|---|---:|
| GLSL entry-point sources | 57 |
| SPIR-V assembly sources | 7 |
| include-only headers | 12 |
| SPIR-V artifacts | 64 |
| total artifact bytes | 120,844 |
| full shader build | about 5 s |
| static pipeline-creation call sites | 107 |

Duplicated binaries are not affecting build time, package size, or
maintenance.

## Cost of adding specialization

- `PipelineKey` is a fixed 60-byte packed struct hashed as bytes. Variable
  specialization data needs owned bytes and a separate equality path, as
  ray-tracing keys already have.
- Shader deduplication keys on SPIR-V hash, entry point, and role;
  specialization values would need normalization so caller order does not
  change identity.
- Reflection would need to enumerate specialization ids and types and fault
  on mismatches.
- `ShaderDesc` gains a new authoring concept and new diagnostics.

Not measured: cold pipeline-creation time and code quality on hardware.
Neither is needed, because no candidate variant family exists.

## Revisit when

A consuming project has a shader family whose members differ only by scalar
values, are built from one source more than once, and whose duplication
measurably costs build time, package size, or pipeline count.
