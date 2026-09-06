#version 460
#include "generated/shader_abi.glsl"
#include "generated/indirect_abi.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer Indices { uint values[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    IndexWriteRoot root = IndexWriteRoot(pc.root_gpu);
    Indices indices = Indices(root.indices_gpu);
    const uint quad[6] = uint[6](0u, 1u, 2u, 2u, 1u, 3u);
    for (uint i = 0u; i < root.index_count; i++) indices.values[i] = quad[i % 6u];
}
