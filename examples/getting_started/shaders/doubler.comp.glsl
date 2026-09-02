#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "buffer_reference.glsl"

layout(local_size_x = 64) in;

// The root is read and written through one unqualified block, so it is
// declared directly; the two array views use the canonical helpers.
layout(buffer_reference, std430) buffer DoublerRoot {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint     count;
};
GPU_DECLARE_READONLY_ARRAY_REF(InBuf, float);
GPU_DECLARE_WRITEONLY_ARRAY_REF(OutBuf, float);

layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    DoublerRoot root = DoublerRoot(root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).values[i] = InBuf(root.input_gpu).values[i] * 2.0;
    }
}
