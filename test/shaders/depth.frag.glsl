#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#include "generated/shader_abi.glsl"

layout(location = 0) out vec4 o_color;

layout(buffer_reference, std430) readonly buffer FragData { vec4 color; };
layout(push_constant) uniform Push { GraphicsRootPush pc; };

void main() {
    o_color = FragData(pc.fragment_root_gpu).color;
}
