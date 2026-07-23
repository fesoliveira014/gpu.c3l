#version 460
#include "generated/shader_abi.glsl"

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer DispatchArgs {
    uint x;
    uint y;
    uint z;
};

layout(buffer_reference, std430) readonly buffer DispatchBuildRoot {
    uint64_t arguments_gpu;
    uint group_count;
    uint _pad0;
};

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    DispatchBuildRoot root = DispatchBuildRoot(pc.root_gpu);
    DispatchArgs arguments = DispatchArgs(root.arguments_gpu);
    arguments.x = root.group_count;
    arguments.y = 1u;
    arguments.z = 1u;
}
