#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(set = 1, binding = 5) uniform accelerationStructureEXT bad_heap[];
layout(buffer_reference, std430) buffer Result { uint value; };
layout(push_constant) uniform Push { uint64_t root_gpu; } pc;
layout(local_size_x = 1) in;

void main() {
    if (pc.root_gpu == 0ul) return;
    rayQueryEXT query;
    rayQueryInitializeEXT(
        query, bad_heap[nonuniformEXT(0u)], gl_RayFlagsOpaqueEXT, 0xffu,
        vec3(0.0), 0.0, vec3(0.0, 0.0, 1.0), 1.0);
    while (rayQueryProceedEXT(query)) {}
    Result(pc.root_gpu).value =
        uint(rayQueryGetIntersectionTypeEXT(query, true));
}
