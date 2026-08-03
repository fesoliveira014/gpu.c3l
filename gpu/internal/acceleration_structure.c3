module gpu::internal @private;

import gpu;

const uint ACCELERATION_STRUCTURE_INSTANCE_FIELD_MAX @private = 0x00FF_FFFF;
const uint ACCELERATION_STRUCTURE_INSTANCE_MASK_MAX @private = 0xFF;
const uint ACCELERATION_STRUCTURE_INSTANCE_FLAGS @private = 0x0F;
const uint ACCELERATION_STRUCTURE_BUILD_FLAGS @private = 0x0F;
const uint ACCELERATION_STRUCTURE_GEOMETRY_FLAGS @private = 0x01;

enum AccelerationStructureBuildEligibility : char {
    UNBUILT,
    PENDING,
    BUILT,
}

struct NormalizedAccelerationStructureGeometry {
    AccelerationStructureGeometryKind  kind;
    AccelerationStructureGeometryFlags flags;
    uint                               max_vertex_count;
    uint                               max_primitive_count;
    AccelerationStructureIndexType     index_type;
    bool                               has_transform;
    uint                               built_vertex_count;
    uint                               built_primitive_count;
    bool                               built_with_transform;
}

struct NormalizedAccelerationStructureDesc {
    AccelerationStructureKind                 kind;
    NormalizedAccelerationStructureGeometry[] geometries;
    uint                                      max_instance_count;
    AccelerationStructureBuildFlags           flags;
    AccelerationStructureRequirements         requirements;
}

fn void? normalize_acceleration_structure_desc(
    AccelerationStructureDesc* desc,
    RayQueryCaps* caps,
    NormalizedAccelerationStructureGeometry[] geometry_storage,
    NormalizedAccelerationStructureDesc* out_desc,
) @private {
    if (desc == null || caps == null || out_desc == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!caps.enabled) return gpu::UNSUPPORTED_FEATURE~;
    uint build_flags = (uint)desc.flags;
    if ((build_flags & ~ACCELERATION_STRUCTURE_BUILD_FLAGS) != 0
        || (desc.flags.prefer_fast_trace
            && desc.flags.prefer_fast_build)) {
        return gpu::INVALID_ARGUMENT~;
    }

    switch (desc.kind) {
        case AccelerationStructureKind.BOTTOM_LEVEL:
            if (desc.geometries.len == 0
                || desc.geometries.len > (usz)caps.max_geometry_count
                || desc.max_instance_count != 0
                || geometry_storage.len != desc.geometries.len) {
                return gpu::INVALID_ARGUMENT~;
            }
        case AccelerationStructureKind.TOP_LEVEL:
            if (desc.geometries.len != 0
                || desc.max_instance_count == 0) {
                return gpu::INVALID_ARGUMENT~;
            }
            if (desc.max_instance_count > caps.max_instance_count) {
                return gpu::UNSUPPORTED_FEATURE~;
            }
            if (geometry_storage.len != 0) return gpu::INVALID_ARGUMENT~;
        default:
            return gpu::INVALID_ARGUMENT~;
    }

    uint primitive_total;
    AccelerationStructureGeometryKind blas_geometry_kind;
    foreach (i, &geometry : desc.geometries) {
        if (i == 0) {
            blas_geometry_kind = geometry.kind;
        } else if (geometry.kind != blas_geometry_kind) {
            return gpu::INVALID_ARGUMENT~;
        }
        uint geometry_flags = (uint)geometry.flags;
        if ((geometry_flags & ~ACCELERATION_STRUCTURE_GEOMETRY_FLAGS) != 0) {
            return gpu::INVALID_ARGUMENT~;
        }
        uint primitive_count;
        NormalizedAccelerationStructureGeometry normalized = {
            .kind  = geometry.kind,
            .flags = geometry.flags,
        };
        switch (geometry.kind) {
            case AccelerationStructureGeometryKind.TRIANGLES:
                if (geometry.triangles.max_vertex_count == 0
                    || geometry.triangles.max_primitive_count == 0) {
                    return gpu::INVALID_ARGUMENT~;
                }
                if (geometry.triangles.index_type
                    == AccelerationStructureIndexType.NONE) {
                    if (geometry.triangles.max_primitive_count
                            > uint::max / 3
                        || geometry.triangles.max_vertex_count
                            < geometry.triangles.max_primitive_count * 3) {
                        return gpu::INVALID_ARGUMENT~;
                    }
                } else if (geometry.triangles.index_type
                        != AccelerationStructureIndexType.U16
                    && geometry.triangles.index_type
                        != AccelerationStructureIndexType.U32) {
                    return gpu::INVALID_ARGUMENT~;
                }
                primitive_count = geometry.triangles.max_primitive_count;
                normalized.max_vertex_count =
                    geometry.triangles.max_vertex_count;
                normalized.max_primitive_count = primitive_count;
                normalized.index_type = geometry.triangles.index_type;
                normalized.has_transform = geometry.triangles.has_transform;
            case AccelerationStructureGeometryKind.AABBS:
                if (geometry.aabbs.max_primitive_count == 0) {
                    return gpu::INVALID_ARGUMENT~;
                }
                primitive_count = geometry.aabbs.max_primitive_count;
                normalized.max_primitive_count = primitive_count;
                normalized.index_type = AccelerationStructureIndexType.NONE;
            default:
                return gpu::INVALID_ARGUMENT~;
        }
        if (primitive_count > caps.max_primitive_count) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        if (primitive_total > uint::max - primitive_count) {
            return gpu::INVALID_ARGUMENT~;
        }
        primitive_total += primitive_count;
        if (primitive_total > caps.max_primitive_count) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        geometry_storage[i] = normalized;
    }

    *out_desc = {
        .kind               = desc.kind,
        .geometries         = geometry_storage,
        .max_instance_count = desc.max_instance_count,
        .flags              = desc.flags,
    };
}

fn AccelerationStructureInstance? pack_acceleration_structure_instance(
    AccelerationStructureInstanceDesc* desc,
    GpuAddress address,
) @private {
    if (desc == null
        || address == (GpuAddress)0
        || desc.custom_index > ACCELERATION_STRUCTURE_INSTANCE_FIELD_MAX
        || desc.mask > ACCELERATION_STRUCTURE_INSTANCE_MASK_MAX) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint flags = (uint)desc.flags;
    if ((flags & ~ACCELERATION_STRUCTURE_INSTANCE_FLAGS) != 0
        || (desc.flags.force_opaque && desc.flags.force_non_opaque)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .transform_row_0          = desc.transform_row_0,
        .transform_row_1          = desc.transform_row_1,
        .transform_row_2          = desc.transform_row_2,
        .custom_index_and_mask    = desc.custom_index | (desc.mask << 24),
        .record_offset_and_flags  = flags << 24,
        .acceleration_structure   = address,
    };
}
