# gpu.c3l Shader ABI

## 1. Purpose

The shader ABI is the contract between C3 CPU code, shader code, and the Vulkan backend. `gpu.c3l` uses a root-pointer ABI: each draw or dispatch receives one or more GPU addresses that point at structured data in GPU-visible memory.

The ABI should minimize public API binding concepts. User code should not bind per-material descriptor sets or construct backend-specific pipeline layouts.

## 2. Core ABI

```text
push constants:
    root_gpu_address : uint64

root data:
    std430-compatible structs in addressable GPU memory

buffer references:
    GpuAddress values

texture references:
    TextureIndex values

sampler references:
    SamplerIndex values
```

## 3. Dispatch ABI

Compute dispatch command:

```text
cmd_dispatch(command_list, root_gpu, groups)
```

Shader receives:

```text
uint64 root_gpu_address
```

Root data example:

```text
ComputeRoot
    input  : GpuAddress
    output : GpuAddress
    count  : uint
```

Shader flow:

```text
root = pointer_from_address(root_gpu_address)
input = pointer_from_address(root.input)
output = pointer_from_address(root.output)
output[i] = transform(input[i])
```

## 4. Graphics ABI

Graphics draw command:

```text
cmd_draw_indexed(command_list, vertex_root, fragment_root, index_span, index_count, instance_count)
```

Vertex shader receives `vertex_root`. Fragment shader receives `fragment_root`.

Do not pass raw pointer values through vertex outputs to fragment shaders. Some backends and validation paths treat cross-stage pointer passing poorly. Pass each stage's root pointer directly.

### Indirect multi-draw convention

An indirect multi-draw shares one `vertex_root`/`fragment_root` pair across
every draw it issues. Per-draw variation flows through `gl_DrawID`
(`shaderDrawParameters` is a required device feature): the root points at a
per-draw table, and the shader indexes it.

```text
Root { table_gpu, ... }
record = Table(root.table_gpu).items[gl_DrawID]
```

Argument structs are part of the ABI and match Vulkan byte-for-byte —
`DrawIndirectCommand` (16 B), `DrawIndexedIndirectCommand` (20 B),
`DispatchIndirectCommand` (12 B) — declared on the C3 side in `gpu/command.c3`
(size-asserted) and on the GLSL side as `extern struct` twins emitted into
`include/shaders/generated/shader_abi.glsl` with identical field names. Compute
shaders write them std430-tight; no padding exists in any of the three.

Capability-gated generated work uses records that keep each work item's roots
and arguments together:

```text
GeneratedDrawRecord
    GpuAddress vertex_root_gpu          // offset 0
    GpuAddress fragment_root_gpu        // offset 8
    DrawIndirectCommand arguments       // offset 16, stride 32

GeneratedDrawIndexedRecord
    GpuAddress vertex_root_gpu          // offset 0
    GpuAddress fragment_root_gpu        // offset 8
    DrawIndexedIndirectCommand arguments // offset 16
    uint _pad0                          // offset 36, stride 40

GeneratedDispatchRecord
    GpuAddress root_gpu                 // offset 0
    DispatchIndirectCommand arguments   // offset 8
    uint _pad0                          // offset 20, stride 24
```

These structs are generated as C3 and GLSL twins. A GPU producer can compact
or reorder the complete work record without maintaining a parallel root table.
`DeviceCaps.generated_work` is true only when all three layouts are supported;
the shared-root indirect convention remains portable on every strict device.

## 5. Buffer layout

All root and table structs use `std430`-compatible layout.

Rules:

```text
use 4-byte scalar alignment for uint/int/float where valid
use 8-byte alignment for uint64/GpuAddress
avoid vec3 in shared ABI structs
prefer Vec4f, Vec4u, and explicit padding
arrays must use shader-compatible stride
field order must match between C3 and shader code
C3 side must assert sizeof and important offsets
```

`Vec2f`, `Vec4f`, and `Vec4u` are public aliases of the C3 SIMD vectors
(`float[<2>]`, `float[<4>]`, `uint[<4>]`); the schema DSL itself only
expresses the float vectors (`vec2`/`vec4`) — `Vec4u` exists for
hand-written structs outside the generator. C3 packs vector struct members
element-aligned, not vector-aligned, so std430's vec2/vec4 alignment is met
through explicit `_padN` fields — the ABI generator (§12) computes std430
layout and rejects any struct whose packed C3 layout would diverge from it.

Bad ABI struct:

```text
position : Vec3f
radius   : float
```

Preferred:

```text
position_radius : Vec4f
```

Bad:

```text
normal : Vec3f
roughness : float
```

Preferred:

```text
normal_roughness : Vec4f
```

## 6. Root struct conventions

Root struct names:

```text
ComputeRoot
VertexRoot
FragmentRoot
DrawRoot
PassRoot
FrameRoot
```

Field naming:

```text
frame_gpu
pass_gpu
draw_gpu
material_gpu
vertex_gpu
index_gpu
output_gpu
```

Texture/sampler fields:

```text
albedo_texture
normal_texture
material_sampler
```

Do not suffix every field with `index` if the type already says `TextureIndex` or `SamplerIndex`, unless it improves readability in shader code.

## 7. GpuAddress type

C3 side:

```text
GpuAddress
    ulong value
```

Shader side:

```text
uint64_t or equivalent 64-bit scalar
```

If a shader language path lacks native `uint64_t`, a two-`uint` representation may be used internally, but public ABI schemas should keep `GpuAddress` as the semantic type.

## 8. TextureIndex and SamplerIndex

Public semantic types:

```text
TextureIndex
SamplerIndex
```

Shader representation:

```text
uint
```

Both types are generation-free 32-bit values. Zero is invalid; a live value is
the zero-based heap slot plus one. CPU generation and device-owner metadata live
only in the `TextureView` ownership token. Published sampler indices remain
valid until device destruction. Shader helpers perform the minus-one decode;
do not index a native descriptor array directly with the encoded value.

Material example:

```text
Material
    base_color_factor : Vec4f
    albedo_texture    : TextureIndex
    normal_texture    : TextureIndex
    material_sampler  : SamplerIndex
    flags             : uint
```

Texture fetch example concept:

```text
texture_heap[GPU_HEAP_SLOT(material.albedo_texture)]
sampler_heap[GPU_HEAP_SLOT(material.material_sampler)]
```

The exact shader spelling depends on the private heap implementation, but material records remain unchanged.

## 9. Descriptor heap shader contract

Shaders that use textures/samplers include a heap-access helper. The library *publishes* this helper as a shader-side ABI include; it is not an application shader, and consuming projects add it to their shader include path:

```text
include/shaders/descriptor_heap.glsl
```

That file should define:

```text
sample_texture_2d(TextureIndex, SamplerIndex, Vec2f) -> Vec4f          (explicit LOD 0; compute-safe)
sample_texture_2d_implicit(TextureIndex, SamplerIndex, Vec2f) -> Vec4f (derivative LOD; fragment stage)
sample_shadow_2d(TextureIndex, SamplerIndex, Vec3f) -> float           (depth compare; needs a compare-enabled sampler)
load_storage_texture(TextureIndex, Vec2i) -> Vec4f
store_storage_texture(TextureIndex, Vec2i, Vec4f)
```

Backend-specific descriptor details should be hidden behind these helpers where possible.

## 10. Push constant contract

Minimum push constant payload:

```text
RootPush
    root_gpu : GpuAddress
```

Graphics may use:

```text
GraphicsRootPush
    vertex_root_gpu   : GpuAddress
    fragment_root_gpu : GpuAddress
```

Optional debug fields may be added later:

```text
frame_index
debug_mode
reserved
```

Changing push constant layout requires updating:

```text
docs/shader_abi.md
SPIR-V reflection validation
pipeline layout creation
```

## 11. Shader language policy

### First implementation

Use GLSL compiled to SPIR-V.

Required GLSL features for root pointer path:

```text
buffer reference / physical storage buffer capability
64-bit integer or supported equivalent
explicit set/binding for descriptor heap declarations
explicit locations for graphics stage interfaces
```

### Future support

Potential future source languages:

```text
HLSL through DXC
Slang
```

The public ABI should not depend on GLSL-specific naming. The ABI generator should be able to emit multiple shader language targets later.

## 12. ABI generator

`tools/gen_shader_abi/` is the source of truth for shared structs. It compiles
`.abi` schema files into a C3 file and a self-contained GLSL include, and owns
the only std430 layout math in the system.

Schema grammar (complete):

```text
file      := "abi" IDENT ";" decl*
decl      := const | typedecl | structdecl | rootdecl | pushdecl | externdecl
const     := "const" scalar IDENT "=" literal ";"
typedecl  := "type" IDENT ":" scalar ";"
structdecl:= "struct" IDENT "{" field+ "}"
rootdecl  := "root"   IDENT "{" field+ "}"
pushdecl  := "push"   IDENT "{" field+ "}"
externdecl:= "extern" "struct" IDENT "{" field+ "}"
field     := type IDENT ";"
type      := "uint" | "int" | "float" | "u64" | "vec2" | "vec4"
           | IDENT   (semantic type or previously declared struct)
scalar    := "uint" | "int" | "float" | "u64"
```

Keywords are contextual — structural only at declaration position, so `type`,
`root`, `push` etc. remain valid field and constant names. Constant literals
are range-checked against their declared type at generation time.

Block kinds and emission:

```text
struct        plain struct on both sides
root          C3 struct; GLSL layout(buffer_reference, std430,
              buffer_reference_align = A) buffer block
push          C3 struct; GLSL plain struct — the shader hand-writes the one
              binding line: layout(push_constant) uniform Push { RootPush pc; };
extern struct C3 declaration already exists (e.g. gpu/command.c3); emits the GLSL
              twin plus C3 size/offset asserts against the existing type
type X : uint user semantic type; C3 typedef, plain scalar in GLSL
const         primitive constant on both sides (workgroup sizes etc.)
```

Builtin semantic types: `GpuAddress`, `TextureIndex`, `SamplerIndex`.

The generator rejects `vec3`, fixed arrays, and any layout requiring implicit
padding — diagnostics name the exact `_padN` fields to insert. Generated C3
carries `$assert T::size` plus a `$reflect(T.field).offset` assert per field,
so consumer compiles prove the C3 layout equals std430.

Invocation (see `scripts/gen_abi.py` for the repo's generation map):

```text
gen_shader_abi --module <c3-module> --c3-out <path> --glsl-out <path> [--check] <files.abi...>
```

Outputs are committed; `--check` regenerates in memory and diffs against the
files on disk (the CI/test drift gate). No build-time trust elevation is
required of consumers. Library outputs:

```text
gpu/shader_abi.c3                          (library source, module gpu)
include/shaders/generated/shader_abi.glsl  (published shader-side include)
```

The generated GLSL include is self-contained: include guard derived from the
`abi` name plus the `#extension` lines its content requires. Shaders that
declare their own `buffer_reference` wrapper blocks (array views) still declare
the buffer-reference extensions themselves.

## 13. Ownership: generator vs shaders

The generator owns layouts; shaders own bindings. Hand-written shader code is
limited to:

```text
array-view wrapper blocks:  layout(buffer_reference, std430) readonly buffer
                            Instances { Instance items[]; }
push binding blocks:        layout(push_constant) uniform Push { RootPush pc; };
```

Wrappers reference generated layouts only, so they carry no drift risk;
`readonly`/`writeonly` stay at the use site where the semantics live. New
hand-mirrored ABI structs are not acceptable — add them to a schema instead.

## 14. SPIR-V reflection validation

Reflection should validate the convention, not define it.

Checks:

```text
entry point exists and matches the declared stage
push constant blocks fit the stage's root-push range
descriptor heap set/binding matches backend convention
shader stages use explicit locations
required capabilities are present
no unexpected descriptor sets are declared
```

Do not use reflection to create arbitrary per-shader public descriptor layouts. That moves the API back toward descriptor-set-driven design.

## 15. Root-pointer compute example

C3 conceptual struct:

```text
ComputeRoot
    GpuAddress input_gpu
    GpuAddress output_gpu
    uint count
    uint _pad0
    uint _pad1
    uint _pad2
```

Shader conceptual struct:

```text
struct ComputeRoot {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
```

The padding makes layout explicit and stable.

## 16. Material example

```text
Material
    Vec4f base_color_factor
    Vec4f emissive_factor_strength
    TextureIndex albedo_texture
    TextureIndex normal_texture
    TextureIndex roughness_texture
    SamplerIndex heap_sampler
    uint flags
    uint _pad0
    uint _pad1
    uint _pad2
```

Shader usage:

```text
material = material_table[material_index]
base_color = sample_texture_2d(material.albedo_texture, material.heap_sampler, uv)
```

Schema identifiers are emitted into GLSL verbatim, so the generator rejects
any declaration, field, type, or constant name that collides with a GLSL
reserved word or builtin type name — diagnosed at the schema line with a
rename suggestion (e.g. `sampler` → `heap_sampler`).

## 17. ABI acceptance criteria

The shader ABI is acceptable when:

```text
root-pointer compute works with no public descriptor-set binding
graphics draw can pass separate vertex/fragment roots
textures are sampled through TextureIndex
samplers are accessed through SamplerIndex
all shared structs have generated or manual size checks
SPIR-V reflection rejects unexpected descriptor declarations
no sample stores vk descriptor objects in material data
```
