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

void main() {
    if (pc.root_gpu == 0ul) return;
    RayQueryRoot root = RayQueryRoot(pc.root_gpu);
    uint probe = gl_GlobalInvocationID.x;
    vec3 origin = probe < 3u
        ? vec3(0.25 + 2.0 * float(probe), 0.25, -1.0)
        : probe == 3u
            ? vec3(6.0, 2.0, -1.0)
            : vec3(0.25, 0.25, -1.0);
    uint cull_mask = probe == 4u ? 0xf0u : 0x0fu;
    rayQueryEXT query;
    GPU_RAY_QUERY_INITIALIZE(
        query,
        root.acceleration_structure_index,
        gl_RayFlagsOpaqueEXT,
        cull_mask,
        origin,
        0.0,
        vec3(0.0, 0.0, 1.0),
        10.0);
    while (rayQueryProceedEXT(query)) {}

    bool hit = rayQueryGetIntersectionTypeEXT(query, true)
        == gl_RayQueryCommittedIntersectionTriangleEXT;
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
