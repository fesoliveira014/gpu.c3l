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
cmd_dispatch(command_list, pipeline, root_gpu, groups)
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
cmd_draw_indexed(command_list, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count)
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
`DispatchIndirectCommand` (12 B) — declared on the C3 side in `command.c3`
(size-asserted) and on the GLSL side in
`include/shaders/indirect_commands.glsl` with identical field names. Compute
shaders write them std430-tight; no padding exists in any of the three.

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
texture_heap[material.albedo_texture]
sampler_heap[material.material_sampler]
```

The exact shader spelling depends on descriptor buffer vs descriptor indexing implementation, but material records remain unchanged.

## 9. Descriptor heap shader contract

Shaders that use textures/samplers include a heap-access helper. The library *publishes* this helper as a shader-side ABI include; it is not an application shader, and consuming projects add it to their shader include path:

```text
include/shaders/descriptor_heap.glsl
```

That file should define:

```text
sample_texture_2d(TextureIndex, SamplerIndex, Vec2f) -> Vec4f
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
C3 push constant structs
GLSL push constant declarations
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

The generator should become the source of truth for shared structs.

Input concept:

```text
abi/root.abi
abi/material.abi
abi/draw.abi
```

Outputs (the C3 struct ships as library source; the GLSL files are published shader-side ABI includes the consumer's own shaders `#include`):

```text
shader_abi.c3                              (library source)
include/shaders/generated/shader_abi.glsl
include/shaders/generated/shader_abi_offsets.glsl
```

Generated C3 should include:

```text
struct declarations
constants
sizeof asserts
offset asserts where C3 supports them
```

Generated shader code should include:

```text
struct declarations
constants
layout helper macros or declarations
```

## 13. Manual ABI phase

Before the generator exists, manual structs are allowed with strict rules:

```text
field order documented in shader_abi.md
C3 and GLSL names match
C3 side has sizeof asserts
shader code uses explicit constants
no vec3
no implicit packing assumptions
small number of structs only
```

Manual ABI structs should be considered technical debt and migrated into generated ABI files by the shader ABI generator milestone.

## 14. SPIR-V reflection validation

Reflection should validate the convention, not define it.

Checks:

```text
entry point exists
push constant size matches expected struct
push constant offset for root pointer is zero
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
    SamplerIndex sampler
    uint flags
    uint _pad0
    uint _pad1
    uint _pad2
```

Shader usage:

```text
material = material_table[material_index]
base_color = sample_texture_2d(material.albedo_texture, material.sampler, uv)
```

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
