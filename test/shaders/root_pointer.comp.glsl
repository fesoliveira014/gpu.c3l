#version 460
#include "generated/shader_abi.glsl"
#include "generated/root_pointer_abi.glsl"
#include "buffer_reference.glsl"

layout(local_size_x = ROOT_POINTER_WORKGROUP) in;

GPU_DECLARE_READONLY_ARRAY_REF(InBuf, float);
GPU_DECLARE_WRITEONLY_ARRAY_REF(OutBuf, float);
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 0ul) return;

    ComputeRoot root = ComputeRoot(pc.root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).values[i] = InBuf(root.input_gpu).values[i] * 2.0;
    }
}
