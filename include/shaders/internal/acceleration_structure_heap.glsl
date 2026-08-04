// Shared acceleration-structure heap and instance helpers.

#ifndef GPU_ACCELERATION_STRUCTURE_HEAP_GLSL
#define GPU_ACCELERATION_STRUCTURE_HEAP_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "generated/shader_abi.glsl"

layout(set = 0, binding = 5) uniform accelerationStructureEXT
    gpu_acceleration_structure_heap[];

#define GPU_ACCELERATION_STRUCTURE_SLOT(index) ((index) - 1u)
#define GPU_ACCELERATION_STRUCTURE(index) \
    gpu_acceleration_structure_heap[ \
        nonuniformEXT(GPU_ACCELERATION_STRUCTURE_SLOT(index))]

const uint GPU_ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_CULL_DISABLE = 0x01u;
const uint GPU_ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_FLIP = 0x02u;
const uint GPU_ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE = 0x04u;
const uint GPU_ACCELERATION_STRUCTURE_INSTANCE_FORCE_NON_OPAQUE = 0x08u;

uint gpu_pack_acceleration_structure_custom_index_and_mask(
    uint custom_index,
    uint mask
) {
    return custom_index | (mask << 24u);
}

uint gpu_pack_acceleration_structure_record_offset_and_flags(uint flags) {
    return flags << 24u;
}

AccelerationStructureInstance gpu_make_acceleration_structure_instance(
    vec4 transform_row_0,
    vec4 transform_row_1,
    vec4 transform_row_2,
    uint custom_index,
    uint mask,
    uint flags,
    uint64_t acceleration_structure
) {
    return AccelerationStructureInstance(
        transform_row_0,
        transform_row_1,
        transform_row_2,
        gpu_pack_acceleration_structure_custom_index_and_mask(
            custom_index,
            mask),
        gpu_pack_acceleration_structure_record_offset_and_flags(flags),
        acceleration_structure);
}

#define GPU_RAY_QUERY_INITIALIZE( \
    query, acceleration_structure_index, ray_flags, cull_mask, \
    origin, t_min, direction, t_max) \
    rayQueryInitializeEXT( \
        query, \
        GPU_ACCELERATION_STRUCTURE(acceleration_structure_index), \
        ray_flags, cull_mask, origin, t_min, direction, t_max)

#define GPU_RAY_QUERY_CANDIDATE_IS_AABB(query) \
    (rayQueryGetIntersectionTypeEXT(query, false) \
        == gl_RayQueryCandidateIntersectionAABBEXT)

// Invoke only after consumer shader code computes and accepts a true
// procedural intersection for the current AABB candidate.
#define GPU_RAY_QUERY_CONFIRM_AABB(query, t) \
    rayQueryGenerateIntersectionEXT(query, t)

#endif
