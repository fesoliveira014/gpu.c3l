#version 460
#include "descriptor_heap.glsl"

layout(local_size_x = 1) in;

void main() {
    vec4 sampled = sample_texture_2d(1u, 1u, vec2(0.5))
        + sample_texture_3d(1u, 1u, vec3(0.5));
    store_storage_texture(1u, ivec2(0), sampled);
    store_storage_texture_3d(1u, ivec3(0), sampled);
}
