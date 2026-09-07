#version 460
#include "buffer_reference.glsl"
#include "generated/shader_abi.glsl"
#include "generated/root_pointer_abi.glsl"

layout(local_size_x = ROOT_POINTER_WORKGROUP) in;

GPU_DECLARE_WRITEONLY_ARRAY_REF(OutBuf, vec4);
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    MatrixRoot root = MatrixRoot(pc.root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i >= root.count) return;
    vec4 v = root.transform * root.planes[i % 4u];
    v.w += float(root.ids[i % 8u]);
    OutBuf(root.output_gpu).values[i] = v;
}
