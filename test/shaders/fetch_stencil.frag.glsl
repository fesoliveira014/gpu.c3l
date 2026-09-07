#version 460
#include "generated/shader_abi.glsl"
#include "generated/depth_abi.glsl"
#include "descriptor_heap.glsl"

layout(location = 0) out uint out_value;

layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    FetchStencilRoot root = FetchStencilRoot(pc.fragment_root_gpu);
    out_value = gpu_fetch_uint(root.texture_index, ivec2(gl_FragCoord.xy), 0);
}
