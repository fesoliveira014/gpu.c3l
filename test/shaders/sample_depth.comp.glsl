#version 460
#include "generated/shader_abi.glsl"
#include "generated/depth_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { float value; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    SampleDepthRoot root = SampleDepthRoot(pc.root_gpu);
    float depth = sample_texture_2d(root.texture_index, root.sampler_index, vec2(root.u, root.v)).r;
    OutBuf(root.out_gpu).value = depth;
}
