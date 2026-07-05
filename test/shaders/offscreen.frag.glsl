#version 460
#include "generated/shader_abi.glsl"
#include "generated/offscreen_abi.glsl"
#include "descriptor_heap.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push { GraphicsRootPush pc; };

void main() {
    FragRoot frag_root = FragRoot(pc.fragment_root_gpu);
    out_color = sample_texture_2d(frag_root.texture_index, frag_root.sampler_index, in_uv);
}
