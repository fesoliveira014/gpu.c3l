#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "indirect_commands.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer IndexedArgs { DrawIndexedIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer PlainArgs { DrawIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer DispatchArgs { DispatchIndirectCommand cmd; };
layout(buffer_reference, std430) writeonly buffer CountBuf { uint value; };
layout(buffer_reference, std430) buffer Root {
    uint64_t indexed_args_gpu;
    uint64_t plain_args_gpu;
    uint64_t dispatch_args_gpu;
    uint64_t count_gpu;
    uint index_count;
    uint vertex_count;
    uint dispatch_x;
    uint count_value;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    Root root = Root(root_gpu);

    IndexedArgs indexed = IndexedArgs(root.indexed_args_gpu);
    indexed.cmds[0] = DrawIndexedIndirectCommand(root.index_count, 1u, 0u, 0, 0u);
    indexed.cmds[1] = DrawIndexedIndirectCommand(root.index_count, 1u, 0u, 0, 0u);

    PlainArgs plain = PlainArgs(root.plain_args_gpu);
    plain.cmds[0] = DrawIndirectCommand(root.vertex_count, 1u, 0u, 0u);

    DispatchArgs(root.dispatch_args_gpu).cmd = DispatchIndirectCommand(root.dispatch_x, 1u, 1u);

    CountBuf(root.count_gpu).value = root.count_value;
}
