#version 460
#include "generated/shader_abi.glsl"
#include "generated/depth_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { uint value; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    FetchStencilRoot root = FetchStencilRoot(pc.root_gpu);
    OutBuf(root.out_gpu).value = gpu_fetch_uint(root.texture_index, ivec2(int(root.x), int(root.y)), 0);
}
