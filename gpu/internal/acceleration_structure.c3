module gpu::internal @private;

import gpu;

const uint ACCELERATION_STRUCTURE_INSTANCE_FIELD_MAX @private = 0x00FF_FFFF;
const uint ACCELERATION_STRUCTURE_INSTANCE_MASK_MAX @private = 0xFF;
const uint ACCELERATION_STRUCTURE_INSTANCE_FLAGS @private = 0x0F;
const uint ACCELERATION_STRUCTURE_BUILD_FLAGS @private = 0x0F;
const uint ACCELERATION_STRUCTURE_GEOMETRY_FLAGS @private = 0x01;
const usz ACCELERATION_STRUCTURE_TRIANGLE_VERTEX_BYTES @private = 12;
const usz ACCELERATION_STRUCTURE_AABB_BYTES @private = 24;

struct NormalizedAccelerationStructureGeometry {
    AccelerationStructureGeometryKind  kind;
    AccelerationStructureGeometryFlags flags;
    uint                               max_vertex_count;
    uint                               max_primitive_count;
    AccelerationStructureIndexType     index_type;
    bool                               has_transform;
}

struct NormalizedAccelerationStructureDesc {
    AccelerationStructureKind                 kind;
    NormalizedAccelerationStructureGeometry[] geometries;
    uint                                      max_instance_count;
    AccelerationStructureBuildFlags           flags;
    AccelerationStructureRequirements         requirements;
}

<* A strided run of `count` elements: nonzero, with a stride that covers one
   element and a byte extent, `(count - 1) * stride + element_size`, that fits
   in usz. *>
fn bool acceleration_structure_strided_bytes_fit(
    uint count,
    usz stride,
    usz element_size,
) @private {
    if (count == 0 || stride < element_size) return false;
    return (usz)count - 1 <= (usz::max - element_size) / stride;
}

<* One build-time geometry: a defined kind, nonzero counts, strides that cover
   one element, and a strided byte extent that fits in usz. *>
fn bool acceleration_structure_geometry_valid(
    AccelerationStructureGeometryBuildDesc* geometry,
) @private {
    switch (geometry.kind) {
        case AccelerationStructureGeometryKind.TRIANGLES:
            return geometry.triangles.primitive_count > 0
                && acceleration_structure_strided_bytes_fit(
                    geometry.triangles.vertex_count,
                    geometry.triangles.vertex_stride,
                    ACCELERATION_STRUCTURE_TRIANGLE_VERTEX_BYTES,
                );
        case AccelerationStructureGeometryKind.AABBS:
            return acceleration_structure_strided_bytes_fit(
                geometry.aabbs.primitive_count,
                geometry.aabbs.stride,
                ACCELERATION_STRUCTURE_AABB_BYTES,
            );
        default:
            return false;
    }
}

<* Rules on a build description that read no resource state: every geometry
   satisfies `acceleration_structure_geometry_valid`. *>
fn bool acceleration_structure_build_desc_valid(
    AccelerationStructureBuildDesc* desc,
) @private {
    if (desc.geometries.len > 0 && desc.geometries.ptr == null) return false;
    foreach (&geometry : desc.geometries) {
        if (!acceleration_structure_geometry_valid(geometry)) return false;
    }
    return true;
}

fn bool acceleration_structure_capacity_geometry_valid(
    AccelerationStructureGeometryDesc* geometry,
    AccelerationStructureGeometryKind first_kind,
    uint* primitive_total,
) @private {
    if (geometry.kind != first_kind
        || ((uint)geometry.flags & ~ACCELERATION_STRUCTURE_GEOMETRY_FLAGS) != 0) {
        return false;
    }
    uint primitive_count;
    switch (geometry.kind) {
        case AccelerationStructureGeometryKind.TRIANGLES:
            if (geometry.triangles.max_vertex_count == 0
                || geometry.triangles.max_primitive_count == 0) {
                return false;
            }
            if (geometry.triangles.index_type
                == AccelerationStructureIndexType.NONE) {
                if (geometry.triangles.max_primitive_count > uint::max / 3
                    || geometry.triangles.max_vertex_count
                        < geometry.triangles.max_primitive_count * 3) {
                    return false;
                }
            } else if (geometry.triangles.index_type
                    != AccelerationStructureIndexType.U16
                && geometry.triangles.index_type
                    != AccelerationStructureIndexType.U32) {
                return false;
            }
            primitive_count = geometry.triangles.max_primitive_count;
        case AccelerationStructureGeometryKind.AABBS:
            if (geometry.aabbs.max_primitive_count == 0) return false;
            primitive_count = geometry.aabbs.max_primitive_count;
        default:
            return false;
    }
    if (*primitive_total > uint::max - primitive_count) return false;
    *primitive_total += primitive_count;
    return true;
}

<* Rules on a capacity description that read no device state: known flag bits
   without both fast-trace and fast-build, a defined kind, a bottom-level
   description with geometries of one defined kind and no instances, a
   top-level description with instances and no geometries, per-geometry counts,
   index type and flags, and a primitive total that fits in uint. *>
fn bool acceleration_structure_desc_valid(
    AccelerationStructureDesc* desc,
) @private {
    if (((uint)desc.flags & ~ACCELERATION_STRUCTURE_BUILD_FLAGS) != 0
        || (desc.flags.prefer_fast_trace && desc.flags.prefer_fast_build)) {
        return false;
    }
    switch (desc.kind) {
        case AccelerationStructureKind.BOTTOM_LEVEL:
            if (desc.geometries.len == 0 || desc.max_instance_count != 0) {
                return false;
            }
        case AccelerationStructureKind.TOP_LEVEL:
            if (desc.geometries.len != 0 || desc.max_instance_count == 0) {
                return false;
            }
        default:
            return false;
    }
    uint primitive_total;
    foreach (&geometry : desc.geometries) {
        if (!acceleration_structure_capacity_geometry_valid(
            geometry,
            desc.geometries[0].kind,
            &primitive_total,
        )) return false;
    }
    return true;
}

<* Rules on an instance description that read no device state: a custom index
   that fits 24 bits, a mask that fits 8 bits, and known flag bits that do not
   combine force_opaque with force_non_opaque. *>
fn bool acceleration_structure_instance_desc_valid(
    AccelerationStructureInstanceDesc* desc,
) @private {
    return desc.custom_index <= ACCELERATION_STRUCTURE_INSTANCE_FIELD_MAX
        && desc.mask <= ACCELERATION_STRUCTURE_INSTANCE_MASK_MAX
        && ((uint)desc.flags & ~ACCELERATION_STRUCTURE_INSTANCE_FLAGS) == 0
        && !(desc.flags.force_opaque && desc.flags.force_non_opaque);
}

fn void? normalize_acceleration_structure_desc(
    AccelerationStructureDesc* desc,
    AccelerationStructureCaps* caps,
    NormalizedAccelerationStructureGeometry[] geometry_storage,
    NormalizedAccelerationStructureDesc* out_desc,
) @private {
    assert(
        desc != null && caps != null && out_desc != null,
        "desc, caps and out_desc must not be null"
    );
    if (!caps.enabled) return gpu::UNSUPPORTED_FEATURE~;

    if (desc.kind == AccelerationStructureKind.BOTTOM_LEVEL) {
        assert(
            desc.geometries.len <= (usz)caps.max_geometry_count,
            "desc.geometries.len must not exceed DeviceCaps.acceleration_structures.max_geometry_count: %s geometries, limit %s",
            desc.geometries.len,
            caps.max_geometry_count
        );
        assert(
            geometry_storage.len == desc.geometries.len,
            "normalization storage must hold one entry per geometry"
        );
    } else {
        if (desc.max_instance_count > caps.max_instance_count) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        assert(
            geometry_storage.len == 0,
            "top-level normalization storage must be empty"
        );
    }

    uint primitive_total;
    foreach (i, &geometry : desc.geometries) {
        uint primitive_count;
        NormalizedAccelerationStructureGeometry normalized = {
            .kind  = geometry.kind,
            .flags = geometry.flags,
        };
        if (geometry.kind == AccelerationStructureGeometryKind.TRIANGLES) {
            primitive_count = geometry.triangles.max_primitive_count;
            normalized.max_vertex_count = geometry.triangles.max_vertex_count;
            normalized.max_primitive_count = primitive_count;
            normalized.index_type = geometry.triangles.index_type;
            normalized.has_transform = geometry.triangles.has_transform;
        } else {
            primitive_count = geometry.aabbs.max_primitive_count;
            normalized.max_primitive_count = primitive_count;
            normalized.index_type = AccelerationStructureIndexType.NONE;
        }
        if (primitive_count > caps.max_primitive_count) {
            return gpu::UNSUPPORTED_FEATURE~;
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

fn AccelerationStructureInstance pack_acceleration_structure_instance(
    AccelerationStructureInstanceDesc* desc,
    GpuAddress address,
) @private {
    assert(desc != null, "desc must not be null");
    assert(address != (GpuAddress)0, "acceleration structure address must be nonzero");
    uint flags = (uint)desc.flags;
    return {
        .transform_row_0          = desc.transform_row_0,
        .transform_row_1          = desc.transform_row_1,
        .transform_row_2          = desc.transform_row_2,
        .custom_index_and_mask    = desc.custom_index | (desc.mask << 24),
        .record_offset_and_flags  = flags << 24,
        .acceleration_structure   = address,
    };
}
