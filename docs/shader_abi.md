# Shader ABI

The shader ABI is the wire contract shared by C3 data, SPIR-V entry points, and
the backend. `gpu.c3l` uses root pointers for generic data and one device-wide
bindless heap for textures and samplers.

## Root push contract

A compute command pushes one unsigned 64-bit value:

```glsl
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;
```

A graphics command pushes two values in this exact order:

```glsl
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;
```

The graphics block is 16 bytes, with members at offsets 0 and 8. Every selected
vertex or fragment entry point that declares push constants declares the
complete block, even if it reads only one member.

Zero is a valid command root. The library does not dereference it; a shader
must avoid dereferencing zero unless the application deliberately relies on
device robustness behavior.

## Addressable std430 data

`GpuAddress` is an unsigned 64-bit address into a live addressable allocation.
Root structs and all data reached through them use std430-compatible layout.

Use these layout rules:

- 4-byte alignment for `uint`, `int`, and `float`;
- 8-byte alignment for `u64` and `GpuAddress`;
- explicit padding where C3's packed member layout differs from std430;
- `vec2` and `vec4` for shared vectors; avoid `vec3`;
- identical field order and array stride on both sides; and
- compile-time C3 size and offset assertions for shared records.

Represent matrices as vector columns. The schema language intentionally has no
matrix or fixed-array type.

A typical schema root is:

```text
root DoublerRoot {
    GpuAddress input_gpu;
    GpuAddress output_gpu;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
```

The shader uses a buffer-reference block emitted by the generator and casts the
pushed address to that reference. Application code obtains the root address
with `get_span_address`.

## Texture and sampler values

`TextureIndex` and `SamplerIndex` are 32-bit shader-visible heap values.
Generated GLSL exposes the heap convention and helper accessors.

`TextureView` is the owner-bearing CPU value. Its index is valid until the view
is destroyed, after which the slot may be reused immediately. An interned
sampler index remains valid until device destruction. Neither raw value stores
a device owner or generation; do not persist it past its owner or move it
between devices.

The backend owns the descriptor set/layout implementation. Shaders consume the
published helper contract; applications do not create or bind descriptor sets.

## Ray-query values and binding 5

`AccelerationStructureIndex` is a 32-bit shader-visible TLAS heap value. Its
owner-bearing CPU twin is `AccelerationStructureView`; destroying that view
makes the index recyclable. The index contains slot plus one, so zero is
invalid. Keep the view and TLAS alive through every query.

Ray-query shaders explicitly include `include/shaders/ray_query.glsl`. The
include requires `GL_EXT_ray_query`, declares the unbounded set-0 binding-5
acceleration-structure array, and provides
`GPU_ACCELERATION_STRUCTURE(index)` and `GPU_RAY_QUERY_INITIALIZE(...)`.
Ordinary shaders and generated ABI includes do not enable the extension or
declare binding 5.

The generated `AccelerationStructureInstance` is exactly 64 bytes: three
row-major transform rows at offsets 0, 16, and 32; packed custom-index/mask and
record-offset/flags words at offsets 48 and 52; and an ordinary 64-bit
`GpuAddress` at offset 56. CPU code normally uses
`make_acceleration_structure_instance`; GPU-authored records may use
`gpu_make_acceleration_structure_instance` after keeping fields within their
24-bit/8-bit contracts. The record offset is always zero because this API has
no shader binding tables.

Triangle candidates may commit through normal ray-query traversal. An AABB is
only a broad-phase candidate: compute the true procedural intersection inside
the `rayQueryProceedEXT` loop and call
`GPU_RAY_QUERY_CONFIRM_AABB(query, t)` only when accepted. Leaving a candidate
unconfirmed rejects it.

## Graphics and indirect work

Direct graphics commands provide independent vertex and fragment roots. Do not
pass a raw pointer through stage outputs; give each stage its root directly.

Shared-root indirect multi-draw uses the same root pair for every draw.
Per-draw data is commonly indexed with `gl_DrawID` from a table referenced by
the root.

The generated ABI includes byte-identical twins for:

- `DrawIndirectCommand` (16 bytes);
- `DrawIndexedIndirectCommand` (20 bytes); and
- `DispatchIndirectCommand` (12 bytes); and
- `TraceRaysIndirectCommand` (12 bytes: `width`, `height`, `depth`); and
- `TraceRaysIndirectCommand2` (104 bytes: ray-generation address/size; miss,
  hit, and callable address/size/stride triples; `width`, `height`, `depth`,
  and `_pad0`).

A compute shader may write `TraceRaysIndirectCommand` through the generated
GLSL declaration. `cmd_trace_rays_indirect` consumes those exact bytes without
translation or host inspection; its direct root and SBT are not part of the
record.

A compute shader may also write the generated `TraceRaysIndirectCommand2`
declaration. `cmd_trace_rays_indirect2` consumes its complete 104-byte packet
without translation or host inspection: the root is still pushed directly, but
the packet supplies all SBT regions and dimensions. The producer must write
valid addresses, SBT layout, nonzero in-limit dimensions, and zero-valued empty
regions, then make the packet visible with an explicit compute-to-indirect
barrier. Keep the owners of every raw SBT address and root-reachable target live
through completion; no command records or creates those owners implicitly.

Capability-gated generated work stores roots and arguments together in
`GeneratedDrawRecord`, `GeneratedDrawIndexedRecord`, or
`GeneratedDispatchRecord`. The shared-root indirect path remains available
when generated work is unsupported.

## Schema generator

Shared layouts are defined under `abi/` and generated by
`tools/gen_shader_abi`. The schema supports:

```text
const TYPE name = value;
type Name : scalar;
struct Name { fields }
root Name { fields }
push Name { fields }
extern struct Name { fields }
```

Field types are `uint`, `int`, `float`, `u64`, `vec2`, `vec4`, a semantic type
such as `GpuAddress`, `TextureIndex`, or `SamplerIndex`, or a previously
declared struct. `extern struct` declares a GLSL twin for a public C3 record
that already exists.

Generate or check all committed outputs with:

```sh
python3 scripts/gen_abi.py
python3 scripts/gen_abi.py --check
```

Library generation updates:

- the marked public ABI block in `gpu/gpu.c3i`;
- private reflection metadata used by pipeline validation; and
- `include/shaders/generated/shader_abi.glsl`.

The generator rejects implicit layout drift and reports the explicit `_padN`
fields needed to make C3 packing match std430. Generated C3 declarations
include compile-time size and member-offset assertions. Generated files are
committed so consumers do not run the generator to build the library.

## Source and shader ownership

The generator owns every mirrored data layout. Hand-written shader code owns
only:

- access-qualified array-view buffer-reference wrappers; and
- the push-constant binding block containing generated fields in schema order.

Do not hand-copy a shared struct into C3 and GLSL. Add it to a schema instead.
GLSL names are emitted verbatim, so schema fields must avoid GLSL keywords.

## Pipeline reflection validation

Pipeline creation validates only the selected SPIR-V entry point. A selected
entry may declare no push block, or one block beginning at offset zero that
exactly matches the compute or graphics root-push contract.

Validation checks:

- entry-point existence and execution model;
- push-block count, size, members, order, offsets, scalar width, signedness,
  and integer shape;
- the device-wide texture/sampler heap convention;
- the opted-in acceleration-structure heap at set 0 binding 5;
- explicit stage interface locations; and
- absence of unexpected descriptor sets.

Nested structs, vectors, matrices, arrays, booleans, and physical-reference
members are not accepted in the root push block even if their total byte size
matches. Put only the generated flat unsigned address fields in that block and
place structured data behind the root address.

Failures return `SHADER_INVALID` before a native pipeline is published.
Reflection names are diagnostic only and are not part of the ABI.
