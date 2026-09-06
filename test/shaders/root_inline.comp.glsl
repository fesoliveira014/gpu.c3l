#version 460
#include "buffer_reference.glsl"
#include "generated/shader_abi.glsl"
#include "generated/inline_root_abi.glsl"

layout(local_size_x = INLINE_ROOT_WORKGROUP) in;

GPU_DECLARE_READONLY_ARRAY_REF(InBuf, float);
GPU_DECLARE_WRITEONLY_ARRAY_REF(OutBuf, float);

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.count) {
        OutBuf(pc.output_gpu).values[i] = InBuf(pc.input_gpu).values[i] * 2.0;
    }
}
