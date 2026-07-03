#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "descriptor_heap.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(buffer_reference, std430) readonly buffer FragData {
    uint texture_index;
    uint sampler_index;
};
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
};

void main() {
    FragData frag_data = FragData(fragment_root_gpu);
    out_color = sample_texture_2d(frag_data.texture_index, frag_data.sampler_index, in_uv);
}
