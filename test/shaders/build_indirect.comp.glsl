#version 460
#include "generated/shader_abi.glsl"
#include "generated/indirect_abi.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer IndexedArgs { DrawIndexedIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer PlainArgs { DrawIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer DispatchArgs { DispatchIndirectCommand cmd; };
layout(buffer_reference, std430) writeonly buffer CountBuf { uint value; };
layout(push_constant) uniform Push { RootPush pc; };

void main() {
    BuildRoot root = BuildRoot(pc.root_gpu);

    IndexedArgs indexed = IndexedArgs(root.indexed_args_gpu);
    indexed.cmds[0] = DrawIndexedIndirectCommand(root.index_count, 1u, 0u, 0, 0u);
    indexed.cmds[1] = DrawIndexedIndirectCommand(root.index_count, 1u, 0u, 0, 0u);

    PlainArgs plain = PlainArgs(root.plain_args_gpu);
    plain.cmds[0] = DrawIndirectCommand(root.vertex_count, 1u, 0u, 0u);

    DispatchArgs(root.dispatch_args_gpu).cmd = DispatchIndirectCommand(root.dispatch_x, 1u, 1u);

    CountBuf(root.count_gpu).value = root.count_value;
}
