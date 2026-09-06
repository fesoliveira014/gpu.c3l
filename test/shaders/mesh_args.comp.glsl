#version 460
#include "generated/shader_abi.glsl"
#include "generated/mesh_abi.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer MeshArgs { DrawMeshTasksIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer CountBuf { uint value; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    MeshArgsRoot root = MeshArgsRoot(pc.root_gpu);
    MeshArgs args = MeshArgs(root.args_gpu);
    args.cmds[0] = DrawMeshTasksIndirectCommand(1u, 1u, 1u);
    args.cmds[1] = DrawMeshTasksIndirectCommand(1u, 1u, 1u);
    CountBuf(root.count_gpu).value = root.count_value;
}
