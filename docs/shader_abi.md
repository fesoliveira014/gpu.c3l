# Shader ABI

The shader ABI is the byte contract between C3 code, SPIR-V, and the
backend. It has three parts:

1. A **root pointer** pushed to every dispatch, draw, or trace.
2. **std430 records** reached through that pointer.
3. **Heap indices** for textures, samplers, and acceleration structures.

There are no descriptor sets to build. The backend binds one global heap;
shaders index into it with values stored in root data.

## Root push

Every pipeline layout has one push-constant range of `ROOT_PUSH_CAPACITY`
(128) bytes. The root-address header occupies the start; an optional inline
payload starts at `INLINE_ROOT_OFFSET` (16) and holds at most
`INLINE_ROOT_CAPACITY` (112) bytes. Shaders that read push constants declare
the header for their pipeline kind first, exactly, and any payload members
at or after offset 16.

Compute and ray tracing header, 8 bytes:

```glsl
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;
```

Graphics header, 16 bytes:

```glsl
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;
```

The C3 side passes the addresses directly:

```c3
gpu::cmd_dispatch(&commands, root_address, { 64, 1, 1 })!;
gpu::cmd_draw(
    commands:       &commands,
    vertex_root:    vertex_root,
    fragment_root:  fragment_root,
    vertex_count:   3,
    instance_count: 1,
)!;
```

Zero is a valid root. The library never dereferences it. A shader that
receives zero must branch before reading through it.

### Inline payload

A direct command may carry a payload in the same push. `cmd_dispatch`,
`cmd_dispatch_indirect`, every `cmd_draw*`, and every `cmd_trace_rays*` take
a trailing `char[] inline_root` (default empty). `@inline_root(&value)`
borrows a value as bytes for the call; the command copies them immediately.
The length must be a multiple of 4 and at most `INLINE_ROOT_CAPACITY`, else
`INVALID_ARGUMENT`. Under `ContractValidation.FULL` a payload longer than the
bound pipeline's reflected push block reports a `public_contract` diagnostic
on `inline_root`.

```glsl
layout(push_constant) uniform Push {
    uint64_t root_gpu;
    layout(offset = 16) vec4 tint;
    uint material;
} pc;
```

```c3
struct TintRoot {
    Vec4f tint;
    uint  material;
}

TintRoot tint = { .tint = { 1.0f, 0.5f, 0.5f, 1.0f }, .material = 3 };
gpu::cmd_dispatch(
    commands:    &commands,
    root:        root_address,
    groups:      { 64, 1, 1 },
    inline_root: gpu::@inline_root(&tint),
)!;
```

Payload offsets in the block are `INLINE_ROOT_OFFSET` plus the C3 struct
offsets, so a std430 struct laid out from 0 mirrors the block from 16.
Generated work updates the header only; payload bytes are unspecified after
`cmd_dispatch_generated` or `cmd_draw_generated`, and every direct command
pushes its own payload again.

## Root records

A root is a std430 struct at a `GpuAddress`. The shader casts the pushed
address to a buffer-reference block:

```glsl
layout(buffer_reference, std430, buffer_reference_align = 8) buffer ComputeRoot {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

void main() {
    ComputeRoot root = ComputeRoot(pc.root_gpu);
    ...
}
```

The matching C3 struct must have identical offsets:

```c3
struct ComputeRoot {
    gpu::GpuAddress input_gpu;   // offset 0
    gpu::GpuAddress output_gpu;  // offset 8
    uint            count;       // offset 16
    uint            _pad0;
    uint            _pad1;
    uint            _pad2;       // size 32
}
```

Layout rules:

| Type | Size | Alignment |
|---|---:|---:|
| `uint`, `int`, `float`, `TextureIndex`, `SamplerIndex` | 4 | 4 |
| `u64`, `GpuAddress` | 8 | 8 |
| `vec2` | 8 | 8 |
| `vec4` | 16 | 16 |

Avoid `vec3`. Represent a matrix as `vec4` columns. Pad explicitly where C3
packing and std430 differ. Do not hand-write both sides; use the
[generator](#schema-generator).

## Data behind the root

A root usually holds addresses of larger arrays. Declare those blocks in the
shader. `include/shaders/buffer_reference.glsl` supplies four macros, each
expanding to one std430 block with a single member (`value` for a record,
`values[]` for an array):

```glsl
#include "buffer_reference.glsl"

GPU_DECLARE_READONLY_REF(CameraRef, Camera);          // { Camera value; }
GPU_DECLARE_READONLY_ARRAY_REF(MaterialTable, Material); // { Material values[]; }
GPU_DECLARE_WRITEONLY_ARRAY_REF(OutputTable, vec4);

Camera camera     = CameraRef(root.camera_gpu).value;
Material material = MaterialTable(root.materials_gpu).values[index];
OutputTable(root.output_gpu).values[index] = shaded;
```

A reference is an address and nothing more. It has no length, no bounds
check, no ownership. Pass counts as ordinary root fields and test them in
the shader:

```text
root MaterialRoot {
    GpuAddress materials_gpu;
    uint       material_count;
    uint       _pad0;
}
```

```c3
gpu::MappedGpuSpan mapped = gpu::mapped_gpu_span(&device, material_span)!;
root.materials_gpu  = mapped.address;
root.material_count = material_count;
gpu::flush_mapped_span(&device, root_span)!;
```

Alignment: the macros use the extension default of 16-byte reference
alignment. A whole allocation is 16-aligned, but a `checked_subspan` is not.
For a single record at a smaller offset, declare the block yourself with
`buffer_reference_align`. Array accesses use the element's own alignment and
are unaffected.

## Textures and samplers

`include/shaders/descriptor_heap.glsl` declares the global heap and helper
functions:

| Binding (set 0) | Contents |
|---:|---|
| 0 | sampled 2D textures |
| 1 | storage 2D images |
| 2 | samplers (also viewed as shadow samplers) |
| 3 | sampled 3D textures |
| 4 | storage 3D images |
| 5 | acceleration structures (opt-in) |
| 6 | sampled cube textures |

Store a `TextureIndex` and `SamplerIndex` in root data and sample with the
helpers:

```glsl
#include "descriptor_heap.glsl"

// fragment stage: implicit LOD
vec4 color = sample_texture_2d_implicit(root.albedo, root.sampler, uv);

// compute stage: explicit LOD 0
vec4 color = sample_texture_2d(root.albedo, root.sampler, uv);

// storage image
vec4 v = load_storage_texture(root.image, coord);
store_storage_texture(root.image, coord, v * 2.0);

// depth compare; sampler must have compare_enable
float lit = sample_shadow_2d(root.shadow_map, root.shadow_sampler, vec3(uv, depth));
```

3D variants are `sample_texture_3d`, `sample_texture_3d_implicit`,
`load_storage_texture_3d`, and `store_storage_texture_3d`. Cube views are
sampled by direction with `sample_texture_cube` (LOD 0),
`sample_texture_cube_lod` (explicit LOD), and `sample_texture_cube_implicit`;
the hardware selects the face and filters across face edges. To index the
heap arrays directly use `GPU_HEAP_SLOT(index)`: a live index is the slot
plus one, and zero is invalid.

On the C3 side:

```c3
gpu::TextureView view = gpu::create_texture_view(&device, texture, null)!;
root.albedo  = view.index;                              // TextureIndex
root.sampler = gpu::intern_sampler(&device, &sampler_desc)!; // SamplerIndex
```

The indices carry no ownership. Keep the `TextureView` alive while any
shader can read its index; a destroyed view's slot is reused at once. A
sampler index lives until the device is destroyed.

## Acceleration structures

Ray-query shaders include `ray_query.glsl`; ray-tracing pipeline shaders
include `ray_tracing.glsl`. Both declare binding 5 and the helpers below.
Ordinary shaders do not enable the extensions.

```glsl
#include "ray_query.glsl"

rayQueryEXT query;
GPU_RAY_QUERY_INITIALIZE(
    query, root.tlas_index, gl_RayFlagsNoneEXT, 0xffu,
    origin, 0.0, direction, 1000.0);
while (rayQueryProceedEXT(query)) {
    if (GPU_RAY_QUERY_CANDIDATE_IS_AABB(query)) {
        float t;
        if (intersect_procedural(origin, direction, t)) {
            GPU_RAY_QUERY_CONFIRM_AABB(query, t);
        }
    }
}
```

`GPU_ACCELERATION_STRUCTURE(index)` yields the `accelerationStructureEXT`
for `traceRayEXT`. An `AccelerationStructureIndex` comes from
`AccelerationStructureView.index`; zero is invalid.

`AccelerationStructureInstance` is the 64-byte TLAS instance record: three
row-major `vec4` transform rows, packed custom index and mask, packed record
offset and flags, and a BLAS `GpuAddress`. CPU code packs it with
`make_acceleration_structure_instance`; shaders use
`gpu_make_acceleration_structure_instance`.

## Indirect records

The generated ABI contains C3 and GLSL twins of every GPU-written argument
record:

| Record | Bytes | Consumed by |
|---|---:|---|
| `DrawIndirectCommand` | 16 | `cmd_draw_indirect` |
| `DrawIndexedIndirectCommand` | 20 | `cmd_draw_indexed_indirect`, `..._count` |
| `DispatchIndirectCommand` | 12 | `cmd_dispatch_indirect` |
| `TraceRaysIndirectCommand` | 12 | `cmd_trace_rays_indirect` |
| `TraceRaysIndirectCommand2` | 104 | `cmd_trace_rays_indirect2` |
| `AccelerationStructureIndirectBuildRange` | 16 | `cmd_build_acceleration_structure_indirect` |
| `GeneratedDrawRecord` | 32 | `cmd_draw_generated` |
| `GeneratedDrawIndexedRecord` | 40 | `cmd_draw_indexed_generated` |
| `GeneratedDispatchRecord` | 24 | `cmd_dispatch_generated` |

Shared-root indirect draws pass one vertex and fragment root to every draw;
index per-draw data with `gl_DrawID`. Generated records carry their own
roots.

## Schema generator

Write shared layouts once in a `.abi` file and generate both sides:

```text
abi my_app;

const uint TILE = 64;

root ComputeRoot {
    GpuAddress input_gpu;
    GpuAddress output_gpu;
    uint       count;
    uint       _pad0;
    uint       _pad1;
    uint       _pad2;
}

struct Material {
    vec4 base_color;
    TextureIndex albedo;
    SamplerIndex sampler;
    uint _pad0;
    uint _pad1;
}
```

Declarations: `const`, `type Name : scalar`, `struct`, `root`, `push`,
`push compute` / `push graphics`, and `extern struct` (GLSL twin of an
existing C3 record). Field types: `uint`,
`int`, `float`, `u64`, `vec2`, `vec4`, `GpuAddress`, `TextureIndex`,
`SamplerIndex`, `AccelerationStructureIndex`, or an earlier struct. No
matrices, no fixed arrays.

Build and run the generator:

```sh
c3c build gen_shader_abi --path lib/gpu.c3l/tools/gen_shader_abi
lib/gpu.c3l/tools/gen_shader_abi/build/gen_shader_abi \
  --module my_app \
  --c3-out src/shader_abi.c3 \
  --glsl-out shaders/generated/my_app_abi.glsl \
  abi/my_app.abi
```

Add `--check` in CI to fail on drift. The generator rejects implicit padding
and names the `_padN` fields to add. Generated C3 carries size and offset
assertions; generated GLSL emits `root` types as
`buffer_reference` blocks and `struct` types as plain structs.

`push compute Name { ... }` and `push graphics Name { ... }` declare an
inline payload. The C3 side is the payload struct, laid out from 0 and
asserted against `INLINE_ROOT_CAPACITY`. The GLSL side is the whole
`layout(push_constant)` block named `pc`: the header for the role, then the
fields from `layout(offset = 16)`. Put a role-qualified `push` in a schema
of its own, since every shader that includes the generated GLSL receives the
block. A bare `push` keeps the library header meaning.

Shaders include the library ABI and then the application ABI:

```glsl
#include "generated/shader_abi.glsl"
#include "generated/my_app_abi.glsl"
```

GLSL names are emitted verbatim. Do not use GLSL keywords as field names.

## Pipeline validation

Pipeline creation reflects the selected entry point and rejects with
`SHADER_INVALID` when:

- the push block is present but does not start with the exact compute or
  graphics header (offsets, 64-bit unsigned scalars), has a member between
  the header and `INLINE_ROOT_OFFSET`, or is larger than
  `ROOT_PUSH_CAPACITY`;
- a descriptor set other than set 0 is declared, or set 0 does not match the
  heap convention;
- the entry point or execution model is missing.

Binding 5 on a device without ray features returns `UNSUPPORTED_FEATURE`.

Header members must be flat unsigned 64-bit addresses. Payload members at
or after `INLINE_ROOT_OFFSET` are not shape-checked; the application owns
their layout through the schema generator or by hand.
