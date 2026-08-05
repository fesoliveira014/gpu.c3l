#version 460
#include "generated/shader_abi.glsl"

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer TraceArgs {
    TraceRaysIndirectCommand command;
};

layout(buffer_reference, std430) readonly buffer TraceArgsRoot {
    uint64_t arguments_gpu;
};

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    TraceArgsRoot root = TraceArgsRoot(pc.root_gpu);
    TraceArgs(root.arguments_gpu).command =
        TraceRaysIndirectCommand(4u, 1u, 1u);
}
