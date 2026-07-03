#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) out vec4 o_color;

layout(buffer_reference, std430) readonly buffer FragData { vec4 color; };
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
};

void main() {
    o_color = FragData(fragment_root_gpu).color;
}
