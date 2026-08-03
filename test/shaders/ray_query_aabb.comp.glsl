#version 460
#extension GL_EXT_buffer_reference : require
#include "generated/shader_abi.glsl"
#include "ray_query.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) readonly buffer RayQueryRoot {
    uint acceleration_structure_index;
    uint _pad0;
    uint64_t output_gpu;
};

struct RayQueryHitResult {
    uint hit;
    uint instance_custom_index;
    uint primitive_index;
    float distance;
};

layout(buffer_reference, std430) writeonly buffer RayQueryResults {
    RayQueryHitResult values[];
};

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

bool intersect_procedural_sphere(vec3 origin, vec3 direction, out float t) {
    vec3 offset = origin - vec3(0.5);
    float projected = dot(offset, direction);
    float discriminant = projected * projected
        - (dot(offset, offset) - 0.25 * 0.25);
    if (discriminant < 0.0) return false;
    t = -projected - sqrt(discriminant);
    return t >= 0.0;
}

void main() {
    if (pc.root_gpu == 0ul) return;
    RayQueryRoot root = RayQueryRoot(pc.root_gpu);
    uint probe = gl_GlobalInvocationID.x;
    vec3 origin = probe == 0u
        ? vec3(0.5, 0.5, -1.0)
        : probe == 1u
            ? vec3(0.1, 0.1, -1.0)
            : vec3(2.25, 0.25, -1.0);
    vec3 direction = vec3(0.0, 0.0, 1.0);
    rayQueryEXT query;
    GPU_RAY_QUERY_INITIALIZE(
        query,
        root.acceleration_structure_index,
        gl_RayFlagsNoneEXT,
        0xffu,
        origin,
        0.0,
        direction,
        10.0);
    while (rayQueryProceedEXT(query)) {
        if (GPU_RAY_QUERY_CANDIDATE_IS_AABB(query)) {
            float t;
            if (intersect_procedural_sphere(origin, direction, t)) {
                GPU_RAY_QUERY_CONFIRM_AABB(query, t);
            }
        }
    }

    uint committed_type = rayQueryGetIntersectionTypeEXT(query, true);
    bool hit = committed_type == gl_RayQueryCommittedIntersectionGeneratedEXT
        || committed_type == gl_RayQueryCommittedIntersectionTriangleEXT;
    RayQueryResults results = RayQueryResults(root.output_gpu);
    results.values[probe].hit = hit ? 1u : 0u;
    results.values[probe].instance_custom_index = hit
        ? rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)
        : 0u;
    results.values[probe].primitive_index = hit
        ? rayQueryGetIntersectionPrimitiveIndexEXT(query, true)
        : 0u;
    results.values[probe].distance = hit
        ? rayQueryGetIntersectionTEXT(query, true)
        : 0.0;
}
