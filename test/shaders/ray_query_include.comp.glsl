#version 460
#extension GL_EXT_buffer_reference : require
#include "generated/shader_abi.glsl"
#include "ray_query.glsl"

layout(local_size_x = 1) in;
layout(buffer_reference, std430) buffer Result {
    uint value;
};
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 0ul) return;
    AccelerationStructureInstance instance =
        gpu_make_acceleration_structure_instance(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, 1.0, 0.0, 0.0),
            vec4(0.0, 0.0, 1.0, 0.0),
            7u,
            0xffu,
            GPU_ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE,
            uint64_t(0x1000u));
    rayQueryEXT query;
    GPU_RAY_QUERY_INITIALIZE(
        query,
        1u,
        gl_RayFlagsOpaqueEXT,
        0xffu,
        vec3(0.0),
        0.0,
        vec3(0.0, 0.0, 1.0),
        100.0);
    while (rayQueryProceedEXT(query)) {
        if (GPU_RAY_QUERY_CANDIDATE_IS_AABB(query)) {
            GPU_RAY_QUERY_CONFIRM_AABB(query, 1.0);
        }
    }
    Result(pc.root_gpu).value = instance.custom_index_and_mask
        + uint(rayQueryGetIntersectionTypeEXT(query, true));
}
