#version 460
#include "generated/shader_abi.glsl"
#include "generated/multisample_abi.glsl"
#include "multisample_heap.glsl"

const uint MODE_PACKED_COLOR = 0u;
const uint MODE_DEPTH_BITS = 1u;
const uint MODE_UNSIGNED = 2u;
const uint MODE_SINGLE_SAMPLE_DEPTH_BITS = 3u;

layout(local_size_x = 8, local_size_y = 8) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { uint values[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    MultisampleFetchRoot root = MultisampleFetchRoot(pc.root_gpu);
    uvec2 texel = gl_GlobalInvocationID.xy;
    if (texel.x >= root.width || texel.y >= root.height) return;

    uint base = (texel.y * root.width + texel.x) * root.samples;
    for (uint sample_index = 0u; sample_index < root.samples; sample_index++) {
        uint value;
        if (root.mode == MODE_PACKED_COLOR) {
            value = packUnorm4x8(gpu_fetch_ms(root.texture_index, ivec2(texel), int(sample_index)));
        } else if (root.mode == MODE_DEPTH_BITS) {
            value = floatBitsToUint(gpu_fetch_ms(root.texture_index, ivec2(texel), int(sample_index)).x);
        } else if (root.mode == MODE_UNSIGNED) {
            value = gpu_fetch_ms_uint(root.texture_index, ivec2(texel), int(sample_index));
        } else {
            value = floatBitsToUint(texelFetch(
                gpu_texture_heap[nonuniformEXT(GPU_HEAP_SLOT(root.texture_index))],
                ivec2(texel),
                0).x);
        }

        OutBuf(root.out_gpu).values[base + sample_index] = value;
    }
}
