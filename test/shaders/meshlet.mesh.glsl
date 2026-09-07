#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#include "generated/shader_abi.glsl"

layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

layout(buffer_reference, std430) readonly buffer VertexData { vec4 verts[]; };
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    if (pc.vertex_root_gpu == 0ul) {
        SetMeshOutputsEXT(0, 0);
        return;
    }
    SetMeshOutputsEXT(3, 1);
    VertexData data = VertexData(pc.vertex_root_gpu);
    for (uint i = 0; i < 3; i++) {
        gl_MeshVerticesEXT[i].gl_Position = vec4(data.verts[i].xyz, 1.0);
    }
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
