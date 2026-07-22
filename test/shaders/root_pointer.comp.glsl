#version 460
#include "generated/shader_abi.glsl"
#include "generated/root_pointer_abi.glsl"

layout(local_size_x = ROOT_POINTER_WORKGROUP) in;

layout(buffer_reference, std430) readonly  buffer InBuf  { float v[]; };
layout(buffer_reference, std430) writeonly buffer OutBuf { float v[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 0ul) return;

    ComputeRoot root = ComputeRoot(pc.root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).v[i] = InBuf(root.input_gpu).v[i] * 2.0;
    }
}
