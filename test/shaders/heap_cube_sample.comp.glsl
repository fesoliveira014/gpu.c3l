#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { vec4 texels[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

// Output layout: [0..6) the six face axes at LOD 0, [6] +X at LOD 1,
// [7] the +X/+Y edge with linear filtering, [8] face 3 through a 2D view.
void main() {
    CubeHeapRoot root = CubeHeapRoot(pc.root_gpu);
    const vec3 axes[6] = vec3[6](
        vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0));
    for (uint face = 0u; face < 6u; face++) {
        OutBuf(root.output_gpu).texels[face] = sample_texture_cube(
            root.cube_index,
            root.nearest_sampler,
            axes[face]);
    }
    OutBuf(root.output_gpu).texels[6] = sample_texture_cube_lod(
        root.cube_index,
        root.nearest_sampler,
        axes[0],
        1.0);
    OutBuf(root.output_gpu).texels[7] = sample_texture_cube(
        root.cube_index,
        root.linear_sampler,
        normalize(vec3(1.0, 1.0, 0.0)));
    OutBuf(root.output_gpu).texels[8] = sample_texture_2d(
        root.face_index,
        root.nearest_sampler,
        vec2(0.5, 0.5));
}
