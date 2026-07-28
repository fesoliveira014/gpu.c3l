#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { vec4 texels[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    VolumeHeapRoot root = VolumeHeapRoot(pc.root_gpu);
    uvec3 p = uvec3(1u, 2u, 3u);
    vec3 uvw = (vec3(p) + 0.5) / vec3(root.width, root.height, root.depth);
    vec2 uv = (vec2(p.xy) + 0.5) / vec2(root.width, root.height);
    uint output_index = root.width * root.height * root.depth
        + root.width * root.height;
    OutBuf(root.output_gpu).texels[output_index] = sample_texture_3d(
        root.texture_3d_index,
        root.sampler_index,
        uvw);
    OutBuf(root.output_gpu).texels[output_index + 1] = sample_texture_2d(
        root.texture_2d_index,
        root.sampler_index,
        uv);
}
