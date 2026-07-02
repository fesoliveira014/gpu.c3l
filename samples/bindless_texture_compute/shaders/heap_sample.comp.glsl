#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "descriptor_heap.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { vec4 texels[]; };
layout(buffer_reference, std430) buffer Root {
    uint64_t output_gpu;
    uint texture_index;
    uint sampler_index;
    uint width;
    uint height;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    Root root = Root(root_gpu);
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= root.width || p.y >= root.height) return;
    vec2 uv = (vec2(p) + 0.5) / vec2(root.width, root.height);
    vec4 texel = sample_texture_2d(root.texture_index, root.sampler_index, uv);
    OutBuf(root.output_gpu).texels[p.y * root.width + p.x] = texel;
}
