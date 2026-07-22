#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = HEAP_TILE, local_size_y = HEAP_TILE) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { vec4 texels[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    SampleRoot root = SampleRoot(pc.root_gpu);
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= root.width || p.y >= root.height) return;
    vec2 uv = (vec2(p) + 0.5) / vec2(root.width, root.height);
    vec4 texel = sample_texture_2d(root.texture_index, root.sampler_index, uv);
    OutBuf(root.output_gpu).texels[p.y * root.width + p.x] = texel;
}
