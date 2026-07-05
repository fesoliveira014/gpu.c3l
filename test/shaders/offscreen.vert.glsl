#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#include "generated/shader_abi.glsl"

layout(location = 0) out vec2 out_uv;

layout(buffer_reference, std430) readonly buffer VertexData { vec4 verts[]; };
layout(push_constant) uniform Push { GraphicsRootPush pc; };

void main() {
    vec4 v = VertexData(pc.vertex_root_gpu).verts[gl_VertexIndex];
    gl_Position = vec4(v.xy, 0.0, 1.0);
    out_uv = v.zw;
}
