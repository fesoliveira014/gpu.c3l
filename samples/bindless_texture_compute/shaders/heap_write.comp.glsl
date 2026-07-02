#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "descriptor_heap.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(buffer_reference, std430) buffer Root {
    uint texture_index;
    uint width;
    uint height;
    uint _pad0;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    Root root = Root(root_gpu);
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= root.width || p.y >= root.height) return;
    vec4 value = vec4(
        float(p.x) / 255.0,
        float(p.y) / 255.0,
        float(p.x ^ p.y) / 255.0,
        1.0);
    store_storage_texture(root.texture_index, ivec2(p), value);
}
