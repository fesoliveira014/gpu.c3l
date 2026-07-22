#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = HEAP_TILE, local_size_y = HEAP_TILE) in;

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    WriteRoot root = WriteRoot(pc.root_gpu);
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= root.width || p.y >= root.height) return;
    vec4 value = vec4(
        float(p.x) / 255.0,
        float(p.y) / 255.0,
        float(p.x ^ p.y) / 255.0,
        1.0);
    store_storage_texture(root.texture_index, ivec2(p), value);
}
