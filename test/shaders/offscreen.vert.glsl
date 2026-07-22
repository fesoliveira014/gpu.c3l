#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#include "generated/shader_abi.glsl"

layout(location = 0) out vec2 out_uv;

layout(buffer_reference, std430) readonly buffer VertexData { vec4 verts[]; };
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    if (pc.vertex_root_gpu == 0ul) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        out_uv = vec2(0.0);
        return;
    }

    vec4 v = VertexData(pc.vertex_root_gpu).verts[gl_VertexIndex];
    gl_Position = vec4(v.xy, 0.0, 1.0);
    out_uv = v.zw;
}
