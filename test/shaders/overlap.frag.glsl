#version 460
#include "generated/shader_abi.glsl"
#include "generated/overlap_abi.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    OverlapFragRoot root = OverlapFragRoot(pc.fragment_root_gpu);
    vec2 acc = in_uv;
    uint iters = root.iters;
    for (uint i = 0; i < iters; i++) {
        acc = fma(acc, vec2(1.0000001), vec2(0.0000001));
    }
    out_color = vec4(acc, 0.0, 1.0);
}
