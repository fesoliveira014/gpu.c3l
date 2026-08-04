module gpu_test;

import gpu @public;
import gpu::internal @public;

fn void ray_query_configuration_values_are_zero_compatible() @test {
    gpu::RuntimeDesc runtime_desc;
    gpu::DeviceDesc device_desc;
    gpu::CommandAllocatorDesc allocator_desc;
    gpu::DeviceCaps caps;
    gpu::StageMask acceleration_structure_build = {
        .acceleration_structure_build = true,
    };

    assert(runtime_desc.acceleration_structure_heap_capacity == 0);
    assert(!device_desc.enable_ray_queries);
    assert(allocator_desc.max_acceleration_structure_geometries_per_build == 0);
    assert(!caps.acceleration_structures.enabled);
    assert(!caps.ray_queries.enabled);
    assert((uint)acceleration_structure_build == (1 << 10));
}

fn void acceleration_structure_storage_and_scratch_ranges_are_separate() @test {
    AccelerationStructureRequirements requirements = {
        .storage_size = 1024,
        .storage_alignment = 256,
        .build_scratch_size = 512,
        .update_scratch_size = 256,
        .scratch_alignment = 128,
    };
    assert(gpu::internal::acceleration_structure_storage_fits(
        &requirements,
        2048,
        256,
        256,
    ));
    assert(!gpu::internal::acceleration_structure_storage_fits(
        &requirements,
        2048,
        128,
        256,
    ));
    assert(!gpu::internal::acceleration_structure_storage_fits(
        &requirements,
        2048,
        256,
        257,
    ));
    assert(!gpu::internal::acceleration_structure_storage_fits(
        &requirements,
        1279,
        256,
        256,
    ));

    assert(gpu::internal::acceleration_structure_scratch_fits(
        &requirements,
        512,
        (GpuAddress)0x1000,
        false,
    ));
    assert(gpu::internal::acceleration_structure_scratch_fits(
        &requirements,
        256,
        (GpuAddress)0x1080,
        true,
    ));
    assert(!gpu::internal::acceleration_structure_scratch_fits(
        &requirements,
        511,
        (GpuAddress)0x1000,
        false,
    ));
    assert(!gpu::internal::acceleration_structure_scratch_fits(
        &requirements,
        512,
        (GpuAddress)0x1001,
        false,
    ));
}

fn void acceleration_structure_public_values_preserve_identity_and_layout() @test {
    AccelerationStructureHandle handle = {
        .owner      = 7,
        .index      = 3,
        .generation = 11,
    };
    AccelerationStructureHandle different = handle;
    different.generation++;

    assert(AccelerationStructureHandle::size <= ulong::size * 2,
        "acceleration structure handles fit in two machine words");
    assert(handle.is_valid(), "nonzero owner and generation are valid");
    assert(handle == handle, "identical acceleration structure identities compare equal");
    assert(handle != different, "generation participates in acceleration structure identity");
    assert(!gpu::ACCELERATION_STRUCTURE_HANDLE_INVALID.is_valid(),
        "the zero acceleration structure handle is invalid");

    AccelerationStructureIndex index = { .value = 5 };
    AccelerationStructureView view = {
        .owner      = 7,
        .index      = index,
        .generation = 13,
    };
    assert(AccelerationStructureIndex::size == uint::size,
        "acceleration structure indices are ABI-pinned uints");
    assert((uint)index == 5 && index.is_valid(),
        "the shader index contains only slot-plus-one");
    assert(view.is_valid() && (uint)view.index == 5,
        "the owner-bearing view contains the raw shader index");
    assert(!gpu::ACCELERATION_STRUCTURE_INDEX_INVALID.is_valid(),
        "the zero acceleration structure index is invalid");
    assert(!gpu::ACCELERATION_STRUCTURE_VIEW_INVALID.is_valid(),
        "the zero acceleration structure view is invalid");

    assert((int)DebugResourceKind.TIMESTAMP_POOL == 13,
        "pre-existing debug resource kinds keep their values");
    assert((int)DebugResourceKind.ACCELERATION_STRUCTURE == 14,
        "acceleration structures append to the debug resource vocabulary");
    assert((int)DebugResourceKind.ACCELERATION_STRUCTURE_VIEW == 15,
        "acceleration structure views append after structures");
}

fn void acceleration_structure_descriptors_represent_supported_geometry() @test {
    AccelerationStructureGeometryDesc[2] geometries = {
        {
            .kind  = AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque = true },
            .triangles = {
                .max_vertex_count    = 12,
                .max_primitive_count = 4,
                .index_type          = AccelerationStructureIndexType.U16,
            },
        },
        {
            .kind = AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count    = 9,
                .max_primitive_count = 3,
                .index_type          = AccelerationStructureIndexType.U32,
            },
        },
    };
    AccelerationStructureDesc blas = {
        .kind       = AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = geometries[..],
        .flags      = {
            .allow_update      = true,
            .prefer_fast_trace = true,
        },
        .debug_name = "multi-geometry blas",
    };
    AccelerationStructureDesc tlas = {
        .kind               = AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 8,
        .debug_name         = "scene tlas",
    };

    assert(blas.geometries.len == 2 && blas.geometries[0].flags.opaque,
        "BLAS descriptors preserve ordered geometry capacities");
    assert(blas.geometries[0].triangles.index_type == AccelerationStructureIndexType.U16,
        "triangle descriptors preserve the selected index width");
    assert(blas.geometries[1].triangles.max_primitive_count == 3
            && blas.geometries[1].triangles.index_type
                == AccelerationStructureIndexType.U32,
        "each ordered triangle descriptor preserves its own capacity and index width");
    assert(blas.flags.allow_update && blas.flags.prefer_fast_trace,
        "build flags are composable bitstruct fields");
    assert(tlas.max_instance_count == 8 && tlas.geometries.len == 0,
        "TLAS descriptors expose only instance capacity");
}

fn void acceleration_structure_instance_is_std430_compatible() @test {
    AccelerationStructureInstance instance = {
        .transform_row_0           = { 1.0f, 0.0f, 0.0f, 4.0f },
        .transform_row_1           = { 0.0f, 1.0f, 0.0f, 5.0f },
        .transform_row_2           = { 0.0f, 0.0f, 1.0f, 6.0f },
        .custom_index_and_mask     = 0xAB12_3456,
        .record_offset_and_flags  = 0xCD00_0000,
        .acceleration_structure   = (GpuAddress)0x1122_3344_5566_7788,
    };

    assert(AccelerationStructureInstance::size == 64,
        "the Vulkan instance ABI is exactly 64 bytes");
    assert($reflect(AccelerationStructureInstance.transform_row_0).offset == 0);
    assert($reflect(AccelerationStructureInstance.transform_row_1).offset == 16);
    assert($reflect(AccelerationStructureInstance.transform_row_2).offset == 32);
    assert($reflect(AccelerationStructureInstance.custom_index_and_mask).offset == 48);
    assert($reflect(AccelerationStructureInstance.record_offset_and_flags).offset == 52);
    assert($reflect(AccelerationStructureInstance.acceleration_structure).offset == 56);
    assert(instance.transform_row_2[3] == 6.0f);
    assert(instance.acceleration_structure == (GpuAddress)0x1122_3344_5566_7788);
}

fn void acceleration_structure_dynamic_inputs_are_gpu_spans() @test {
    GpuSpan vertices = { .size = 144 };
    GpuSpan indices = { .size = 24 };
    GpuSpan transform = { .size = 48 };
    AccelerationStructureGeometryBuildDesc geometry = {
        .kind = AccelerationStructureGeometryKind.TRIANGLES,
        .triangles = {
            .vertices       = vertices,
            .vertex_stride  = 12,
            .vertex_count   = 12,
            .indices        = indices,
            .primitive_count = 4,
            .transform      = transform,
        },
    };
    AccelerationStructureGeometryBuildDesc[1] build_geometries = { geometry };
    AccelerationStructureBuildDesc build = {
        .destination = {
            .owner      = 7,
            .index      = 3,
            .generation = 11,
        },
        .geometries = build_geometries[..],
        .scratch    = { .size = 256 },
    };

    assert(build.geometries.len == 1);
    assert(build.geometries[0].triangles.vertex_stride == 12);
    assert(build.geometries[0].triangles.transform.size == 48);
    assert(build.scratch.size == 256);
}

fn void acceleration_structure_caps_and_requirements_are_actionable() @test {
    AccelerationStructureCaps caps = {
        .enabled             = true,
        .heap_capacity       = 64,
        .max_geometry_count  = 8,
        .max_primitive_count = 4096,
        .max_instance_count  = 1024,
        .scratch_alignment   = 256,
    };
    AccelerationStructureRequirements requirements = {
        .storage_size      = 4096,
        .storage_alignment = 256,
        .build_scratch_size = 2048,
        .update_scratch_size = 1024,
        .scratch_alignment = 256,
    };

    assert(caps.enabled && caps.heap_capacity == 64);
    assert(caps.max_geometry_count == 8 && caps.max_instance_count == 1024);
    assert(requirements.storage_size == 4096);
    assert(requirements.build_scratch_size == 2048);
    assert(requirements.update_scratch_size == 1024);
    assert(requirements.scratch_alignment == 256);
}
