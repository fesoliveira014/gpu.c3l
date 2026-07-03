#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { float value; };
layout(buffer_reference, std430) readonly buffer Root {
    uint64_t out_gpu;
    uint texture_index;
    uint sampler_index;
    float u;
    float v;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    Root root = Root(root_gpu);
    float depth = sample_texture_2d(root.texture_index, root.sampler_index, vec2(root.u, root.v)).r;
    OutBuf(root.out_gpu).value = depth;
}
