#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    if (pc.fragment_root_gpu == 0ul) {
        out_color = vec4(0.0);
        return;
    }

    RowSampleRoot root = RowSampleRoot(pc.fragment_root_gpu);
    uint index = root.base + root.frame * root.attachment_count + root.attachment;
    out_color = sample_texture_2d(index, root.sampler_index, in_uv);
}
