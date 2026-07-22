#version 460
#include "generated/shader_abi.glsl"
#include "generated/depth_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { float lo; float hi; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    ShadowProbeRoot root = ShadowProbeRoot(pc.root_gpu);
    OutBuf out_buf = OutBuf(root.out_gpu);
    out_buf.lo = sample_shadow_2d(root.depth_texture, root.shadow_sampler, vec3(root.u, root.v, root.ref_lo));
    out_buf.hi = sample_shadow_2d(root.depth_texture, root.shadow_sampler, vec3(root.u, root.v, root.ref_hi));
}
