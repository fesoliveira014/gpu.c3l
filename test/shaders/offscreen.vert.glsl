#version 460
#include "generated/shader_abi.glsl"
#include "buffer_reference.glsl"

layout(location = 0) out vec2 out_uv;

GPU_DECLARE_READONLY_ARRAY_REF(VertexData, vec4);
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

    vec4 v = VertexData(pc.vertex_root_gpu).values[gl_VertexIndex];
    gl_Position = vec4(v.xy, 0.0, 1.0);
    out_uv = v.zw;
}
