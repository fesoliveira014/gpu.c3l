#version 460
#include "generated/shader_abi.glsl"
#include "generated/overlap_abi.glsl"

layout(local_size_x = OVERLAP_WORKGROUP) in;

layout(buffer_reference, std430) writeonly buffer DataBuf { float v[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    OverlapComputeRoot root = OverlapComputeRoot(pc.root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.elem_count) {
        float acc = float(i);
        uint iters = root.iters;
        for (uint j = 0; j < iters; j++) {
            acc = fma(acc, 1.0000001, 0.0000001);
        }
        DataBuf(root.data_gpu).v[i] = acc;
    }
}
