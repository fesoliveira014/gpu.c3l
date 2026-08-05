module gpu::internal::vk;

import gpu;
import gpu::internal @public;
import std::core::mem;
import std::core::mem::allocators;
import std::atomic;
import vk;

struct AccelerationStructureRequirementsCapture {
    uint calls;
    vk::AccelerationStructureBuildTypeKHR build_type;
    vk::AccelerationStructureTypeKHR type;
    vk::BuildAccelerationStructureFlagsKHR flags;
    uint geometry_count;
    vk::GeometryTypeKHR[2] geometry_types;
    vk::GeometryFlagsKHR[2] geometry_flags;
    vk::Format[2] triangle_formats;
    vk::IndexType[2] triangle_index_types;
    vk::DeviceSize[2] triangle_strides;
    uint[2] triangle_max_vertices;
    vk::DeviceSize[2] aabb_strides;
    bool[2] transform_present;
    uint[2] primitive_counts;
}

struct AccelerationStructureCreateCapture {
    vk::Result result;
    uint create_count;
    uint destroy_count;
    vk::Buffer buffer;
    vk::DeviceSize offset;
    vk::DeviceSize size;
    vk::AccelerationStructureTypeKHR type;
}

struct AccelerationStructureBuildCapture {
    uint calls;
    vk::AccelerationStructureBuildGeometryInfoKHR info;
    vk::AccelerationStructureBuildRangeInfoKHR range;
}

struct AccelerationStructureIndirectBuildCapture {
    uint calls;
    vk::AccelerationStructureBuildGeometryInfoKHR info;
    vk::DeviceAddress range_address;
    uint range_stride;
    uint max_primitive_count;
}

struct AccelerationStructureCloneCapture {
    uint calls;
    vk::CopyAccelerationStructureInfoKHR info;
}

struct AccelerationStructureDescriptorCapture {
    uint                         calls;
    uint                         slot;
    vk::AccelerationStructureKHR acceleration_structure;
}

fn void capture_acceleration_structure_build(
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint info_count,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_infos,
    vk::AccelerationStructureBuildRangeInfoKHR** build_ranges,
) {
    (void)command_buffer;
    AccelerationStructureBuildCapture* capture =
        (AccelerationStructureBuildCapture*)user_data;
    capture.calls++;
    assert(info_count == 1);
    capture.info = build_infos[0];
    capture.range = build_ranges[0][0];
}

fn void capture_acceleration_structure_indirect_build(
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint info_count,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_infos,
    vk::DeviceAddress* range_addresses,
    uint* range_strides,
    uint** max_primitive_counts,
) {
    (void)command_buffer;
    AccelerationStructureIndirectBuildCapture* capture =
        (AccelerationStructureIndirectBuildCapture*)user_data;
    capture.calls++;
    assert(info_count == 1);
    capture.info = build_infos[0];
    capture.range_address = range_addresses[0];
    capture.range_stride = range_strides[0];
    capture.max_primitive_count = max_primitive_counts[0][0];
}

fn void capture_acceleration_structure_clone(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::CopyAccelerationStructureInfoKHR* info,
) {
    (void)command_buffer;
    AccelerationStructureCloneCapture* capture =
        (AccelerationStructureCloneCapture*)user_data;
    capture.calls++;
    capture.info = *info;
}

fn void capture_acceleration_structure_descriptor(
    void* user_data,
    VkDeviceState* state,
    uint slot,
    vk::AccelerationStructureKHR acceleration_structure,
) {
    (void)state;
    AccelerationStructureDescriptorCapture* capture =
        (AccelerationStructureDescriptorCapture*)user_data;
    capture.calls++;
    capture.slot = slot;
    capture.acceleration_structure = acceleration_structure;
}

uint acceleration_structure_native_destroy_count;
uint acceleration_structure_address_count;

fn void count_acceleration_structure_destroy(
    vk::Device device,
    vk::AccelerationStructureKHR acceleration_structure,
    vk::AllocationCallbacks* allocator,
) {
    (void)device;
    (void)acceleration_structure;
    (void)allocator;
    acceleration_structure_native_destroy_count++;
}

fn vk::DeviceAddress return_acceleration_structure_address(
    vk::Device device,
    vk::AccelerationStructureDeviceAddressInfoKHR* info,
) {
    (void)device;
    (void)info;
    acceleration_structure_address_count++;
    return 0x1234_5000;
}

fn vk::Result capture_acceleration_structure_create(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureCreateInfoKHR* create_info,
    vk::AccelerationStructureKHR* out_acceleration_structure,
) {
    (void)device;
    AccelerationStructureCreateCapture* capture =
        (AccelerationStructureCreateCapture*)user_data;
    capture.create_count++;
    capture.buffer = create_info.buffer;
    capture.offset = create_info.offset;
    capture.size = create_info.size;
    capture.type = create_info.type;
    if (capture.result == vk::Result.SUCCESS) {
        *out_acceleration_structure =
            (vk::AccelerationStructureKHR)(uptr)0xACCE1;
    }
    return capture.result;
}

fn void capture_acceleration_structure_destroy(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureKHR acceleration_structure,
) {
    (void)device;
    (void)acceleration_structure;
    AccelerationStructureCreateCapture* capture =
        (AccelerationStructureCreateCapture*)user_data;
    capture.destroy_count++;
}

fn void capture_acceleration_structure_build_sizes(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureBuildTypeKHR build_type,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_info,
    uint* max_primitive_counts,
    vk::AccelerationStructureBuildSizesInfoKHR* size_info,
) {
    (void)device;
    AccelerationStructureRequirementsCapture* capture =
        (AccelerationStructureRequirementsCapture*)user_data;
    capture.calls++;
    capture.build_type = build_type;
    capture.type = build_info.type;
    capture.flags = build_info.flags;
    capture.geometry_count = build_info.geometry_count;
    for (uint i = 0; i < build_info.geometry_count; i++) {
        vk::AccelerationStructureGeometryKHR* geometry =
            &build_info.p_geometries[i];
        capture.geometry_types[i] = geometry.geometry_type;
        capture.geometry_flags[i] = geometry.flags;
        capture.primitive_counts[i] = max_primitive_counts[i];
        if (geometry.geometry_type == vk::GeometryTypeKHR.TRIANGLES_KHR) {
            capture.triangle_formats[i] = geometry.geometry.triangles.vertex_format;
            capture.triangle_index_types[i] = geometry.geometry.triangles.index_type;
            capture.triangle_strides[i] = geometry.geometry.triangles.vertex_stride;
            capture.triangle_max_vertices[i] = geometry.geometry.triangles.max_vertex;
            capture.transform_present[i] =
                geometry.geometry.triangles.transform_data.host_address
                    != null;
        } else if (geometry.geometry_type
            == vk::GeometryTypeKHR.AABBS_KHR) {
            capture.aabb_strides[i] = geometry.geometry.aabbs.stride;
        }
    }
    size_info.acceleration_structure_size = 4096;
    size_info.build_scratch_size = 2048;
    size_info.update_scratch_size = 1024;
}

fn gpu::AccelerationStructureCaps acceleration_structure_test_caps() {
    return {
        .enabled = true,
        .heap_capacity = 8,
        .max_geometry_count = 4,
        .max_primitive_count = 16,
        .max_instance_count = 32,
        .scratch_alignment = 256,
    };
}

fn void acceleration_structure_desc_normalization_is_capacity_first() @test {
    gpu::AccelerationStructureCaps caps = acceleration_structure_test_caps();
    gpu::AccelerationStructureGeometryDesc[2] geometries = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque = true },
            .triangles = {
                .max_vertex_count = 12,
                .max_primitive_count = 4,
                .index_type = gpu::AccelerationStructureIndexType.NONE,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count = 9,
                .max_primitive_count = 3,
                .index_type = gpu::AccelerationStructureIndexType.U32,
            },
        },
    };
    gpu::AccelerationStructureDesc source = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = geometries[..],
        .flags = { .allow_update = true },
    };
    gpu::internal::NormalizedAccelerationStructureGeometry[2] storage;
    gpu::internal::NormalizedAccelerationStructureDesc normalized;

    gpu::internal::normalize_acceleration_structure_desc(
        &source,
        &caps,
        storage[..],
        &normalized,
    )!!;
    assert(normalized.kind == source.kind
            && normalized.geometries.len == 2
            && normalized.geometries[0].index_type
                == gpu::AccelerationStructureIndexType.NONE
            && normalized.geometries[1].index_type
                == gpu::AccelerationStructureIndexType.U32,
        "normalization preserves the ordered immutable schema");

    gpu::AccelerationStructureDesc tlas = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = caps.max_instance_count,
    };
    gpu::internal::normalize_acceleration_structure_desc(
        &tlas,
        &caps,
        {},
        &normalized,
    )!!;
    assert(normalized.geometries.len == 0
            && normalized.max_instance_count == caps.max_instance_count);
}

fn void acceleration_structure_desc_normalization_rejects_invalid_shapes() @test {
    gpu::AccelerationStructureCaps caps = acceleration_structure_test_caps();
    gpu::internal::NormalizedAccelerationStructureGeometry[1] storage;
    gpu::internal::NormalizedAccelerationStructureDesc normalized;
    gpu::AccelerationStructureGeometryDesc geometry = {
        .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
        .triangles = {
            .max_vertex_count = 2,
            .max_primitive_count = 1,
            .index_type = gpu::AccelerationStructureIndexType.NONE,
        },
    };
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = (&geometry)[:1],
    };
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        storage[..],
        &normalized,
    )) == gpu::INVALID_ARGUMENT,
        "non-indexed triangles require three vertices per primitive");

    geometry.triangles.max_vertex_count = 3;
    geometry.triangles.max_primitive_count = caps.max_primitive_count + 1;
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        storage[..],
        &normalized,
    )) == gpu::INVALID_ARGUMENT,
        "unsafe non-indexed capacity arithmetic rejects before limit checks");

    geometry.triangles.max_vertex_count = uint::max;
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        storage[..],
        &normalized,
    )) == gpu::UNSUPPORTED_FEATURE,
        "device primitive limits reject otherwise representable schemas");

    gpu::AccelerationStructureGeometryDesc[2] over_total = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count = 9,
                .max_primitive_count = 9,
                .index_type = gpu::AccelerationStructureIndexType.U32,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count = 8,
                .max_primitive_count = 8,
                .index_type = gpu::AccelerationStructureIndexType.U32,
            },
        },
    };
    gpu::internal::NormalizedAccelerationStructureGeometry[2]
        over_total_storage;
    desc.geometries = over_total[..];
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        over_total_storage[..],
        &normalized,
    )) == gpu::UNSUPPORTED_FEATURE,
        "the device primitive limit applies to the whole BLAS");

    desc.flags = {
        .prefer_fast_trace = true,
        .prefer_fast_build = true,
    };
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        storage[..],
        &normalized,
    )) == gpu::INVALID_ARGUMENT,
        "conflicting portable build preferences are incoherent");

    gpu::AccelerationStructureGeometryDesc[2] heterogeneous = {
        geometry,
        {
            .kind = gpu::AccelerationStructureGeometryKind.AABBS,
            .aabbs.max_primitive_count = 1,
        },
    };
    heterogeneous[0].triangles.max_vertex_count = 3;
    heterogeneous[0].triangles.max_primitive_count = 1;
    desc.flags = {};
    desc.geometries = heterogeneous[..];
    gpu::internal::NormalizedAccelerationStructureGeometry[2]
        heterogeneous_storage;
    assert(@catch(gpu::internal::normalize_acceleration_structure_desc(
        &desc,
        &caps,
        heterogeneous_storage[..],
        &normalized,
    )) == gpu::INVALID_ARGUMENT,
        "one BLAS cannot mix triangle and AABB geometry types");
}

fn void acceleration_structure_requirements_lower_exact_capacity_schema() @test {
    gpu::AccelerationStructureGeometryDesc[2] geometries = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque = true },
            .triangles = {
                .max_vertex_count = 12,
                .max_primitive_count = 4,
                .index_type = gpu::AccelerationStructureIndexType.U16,
                .has_transform = true,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .max_vertex_count = 9,
                .max_primitive_count = 3,
                .index_type = gpu::AccelerationStructureIndexType.U16,
            },
        },
    };
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = geometries[..],
        .flags = {
            .allow_update = true,
            .prefer_fast_trace = true,
            .minimize_memory = true,
        },
    };
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .caps.acceleration_structures = acceleration_structure_test_caps(),
    };
    AccelerationStructureRequirementsCapture capture;
    AccelerationStructureRequirementsOps ops = {
        .user_data = &capture,
        .get_build_sizes = &capture_acceleration_structure_build_sizes,
    };

    gpu::AccelerationStructureRequirements requirements =
        get_acceleration_structure_requirements_with_ops(
            &state,
            &desc,
            &ops,
        )!!;
    uint expected_flags =
        (uint)vk::BuildAccelerationStructureFlagBitsKHR.ALLOW_UPDATE_BIT_KHR
        | (uint)vk::BuildAccelerationStructureFlagBitsKHR.PREFER_FAST_TRACE_BIT_KHR
        | (uint)vk::BuildAccelerationStructureFlagBitsKHR.LOW_MEMORY_BIT_KHR;
    assert(capture.calls == 1
            && capture.build_type
                == vk::AccelerationStructureBuildTypeKHR.DEVICE_KHR
            && capture.type
                == vk::AccelerationStructureTypeKHR.BOTTOM_LEVEL_KHR
            && (uint)capture.flags == expected_flags
            && capture.geometry_count == 2,
        "requirements use one device-build query with portable flags");
    assert(capture.geometry_types[0] == vk::GeometryTypeKHR.TRIANGLES_KHR
            && capture.triangle_formats[0] == vk::Format.R32G32B32_SFLOAT
            && capture.triangle_index_types[0] == vk::IndexType.UINT16
            && capture.triangle_strides[0] == 12
            && capture.triangle_max_vertices[0] == 11
            && capture.transform_present[0]
            && capture.primitive_counts[0] == 4,
        "triangle capacities lower to float32 XYZ and the chosen index width");
    assert(capture.geometry_types[1] == vk::GeometryTypeKHR.TRIANGLES_KHR
            && capture.triangle_index_types[1] == vk::IndexType.UINT16
            && capture.triangle_max_vertices[1] == 8
            && !capture.transform_present[1]
            && capture.primitive_counts[1] == 3,
        "ordered triangle capacities lower independently");
    assert(requirements.storage_size == 4096
            && requirements.storage_alignment == 256
            && requirements.build_scratch_size == 2048
            && requirements.update_scratch_size == 1024
            && requirements.scratch_alignment == 256,
        "public requirements preserve persistent and scratch contracts");

    gpu::AccelerationStructureGeometryDesc aabb_geometry = {
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .flags = { .opaque = true },
        .aabbs.max_primitive_count = 4,
    };
    desc.geometries = (&aabb_geometry)[:1];
    desc.flags = {};
    capture = {};
    (void)get_acceleration_structure_requirements_with_ops(
        &state,
        &desc,
        &ops,
    )!!;
    assert(capture.geometry_types[0] == vk::GeometryTypeKHR.AABBS_KHR
            && capture.aabb_strides[0] == 24
            && (uint)capture.geometry_flags[0]
                == (uint)vk::GeometryFlagBitsKHR.OPAQUE_BIT_KHR,
        "AABB capacity lowers the exact stride and geometry flags");

    aabb_geometry.flags = {};
    capture = {};
    (void)get_acceleration_structure_requirements_with_ops(
        &state,
        &desc,
        &ops,
    )!!;
    assert((uint)capture.geometry_flags[0] == 0,
        "a geometry without the opaque flag lowers to no geometry flags");
}

fn void acceleration_structure_requirements_lower_tlas_and_static_update() @test {
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 17,
    };
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .caps.acceleration_structures = acceleration_structure_test_caps(),
    };
    state.caps.acceleration_structures.max_instance_count = 17;
    AccelerationStructureRequirementsCapture capture;
    AccelerationStructureRequirementsOps ops = {
        .user_data = &capture,
        .get_build_sizes = &capture_acceleration_structure_build_sizes,
    };

    gpu::AccelerationStructureRequirements requirements =
        get_acceleration_structure_requirements_with_ops(
            &state,
            &desc,
            &ops,
        )!!;
    assert(capture.type == vk::AccelerationStructureTypeKHR.TOP_LEVEL_KHR
            && capture.geometry_count == 1
            && capture.geometry_types[0]
                == vk::GeometryTypeKHR.INSTANCES_KHR
            && capture.primitive_counts[0] == 17,
        "TLAS capacity lowers to one instances geometry");
    assert(requirements.update_scratch_size == 0,
        "a static descriptor publishes no update scratch contract");
}

fn void acceleration_structure_native_creation_is_transactional() @test {
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 8,
        .flags = { .allow_update = true },
        .debug_name = "transactional tlas",
    };
    gpu::AccelerationStructureRequirements requirements = {
        .storage_size = 4096,
        .storage_alignment = 256,
        .build_scratch_size = 2048,
        .update_scratch_size = 1024,
        .scratch_alignment = 256,
    };
    AccelerationStructureCell[] cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .caps.acceleration_structures = acceleration_structure_test_caps(),
        .acceleration_structures = {
            .slots = cells,
            .owner = 0xB5,
        },
    };
    AccelerationStructureCreateCapture capture;
    AccelerationStructureNativeOps ops = {
        .user_data = &capture,
        .create = &capture_acceleration_structure_create,
        .destroy = &capture_acceleration_structure_destroy,
    };
    vk::Buffer buffer = (vk::Buffer)(uptr)0xB00F;

    gpu::AccelerationStructureHandle handle =
        create_acceleration_structure_on_buffer_with_ops(
            state:               &state,
            desc:                &desc,
            requirements:        requirements,
            buffer:              buffer,
            offset:              256,
            backing_buffer:      {},
            placement:           {},
            owns_backing_buffer: false,
            ops:                 &ops,
        )!!;
    AccelerationStructureSlot* slot =
        state.acceleration_structures.get(handle)!!;
    assert(capture.create_count == 1 && capture.destroy_count == 0
            && capture.buffer == buffer && capture.offset == 256
            && capture.size == requirements.storage_size
            && capture.type
                == vk::AccelerationStructureTypeKHR.TOP_LEVEL_KHR,
        "native construction receives the exact persistent range");
    assert(slot.desc.max_instance_count == 8
            && slot.desc.flags.allow_update
            && slot.desc.requirements.storage_size == 4096
            && slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        "publication commits immutable metadata and initial build state");
    AccelerationStructureSlot released =
        state.acceleration_structures.free(handle)!!;
    ops.destroy(ops.user_data, state.device, released.native);
    if (released.desc.geometries.ptr != null) {
        mem::free(released.desc.geometries.ptr);
    }

    gpu::AccelerationStructureGeometryDesc geometry = {
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .aabbs.max_primitive_count = 1,
    };
    gpu::AccelerationStructureDesc blas_desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = (&geometry)[:1],
    };
    state.host_allocator = &allocators::LIBC_ALLOCATOR;
    capture = {};
    mem::@scoped(&allocators::NULL_ALLOCATOR) {
        handle = create_acceleration_structure_on_buffer_with_ops(
            state:               &state,
            desc:                &blas_desc,
            requirements:        requirements,
            buffer:              buffer,
            offset:              256,
            backing_buffer:      {},
            placement:           {},
            owns_backing_buffer: false,
            ops:                 &ops,
        )!!;
    };
    released = state.acceleration_structures.free(handle)!!;
    ops.destroy(ops.user_data, state.device, released.native);
    if (released.desc.geometries.ptr != null) {
        alloc::free(state.host_allocator, released.desc.geometries.ptr);
    }
    assert(capture.create_count == 1 && capture.destroy_count == 1,
        "persistent schema storage uses the device host allocator");

    state.host_allocator = &allocators::NULL_ALLOCATOR;
    capture = {};
    assert(@catch(create_acceleration_structure_on_buffer_with_ops(
        state:               &state,
        desc:                &blas_desc,
        requirements:        requirements,
        buffer:              buffer,
        offset:              256,
        backing_buffer:      {},
        placement:           {},
        owns_backing_buffer: false,
        ops:                 &ops,
    )) == gpu::OUT_OF_HOST_MEMORY);
    assert(capture.create_count == 0 && capture.destroy_count == 0,
        "schema allocation failure precedes native creation");
    state.host_allocator = &allocators::LIBC_ALLOCATOR;

    capture = { .result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY };
    assert(@catch(create_acceleration_structure_on_buffer_with_ops(
        state:               &state,
        desc:                &desc,
        requirements:        requirements,
        buffer:              buffer,
        offset:              0,
        backing_buffer:      {},
        placement:           {},
        owns_backing_buffer: false,
        ops:                 &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(capture.create_count == 1 && capture.destroy_count == 0,
        "failed native construction publishes and destroys nothing");

    AccelerationStructureTable full_table = {
        .slots = {},
        .owner = 0xB6,
    };
    state.acceleration_structures = full_table;
    capture = {};
    assert(@catch(create_acceleration_structure_on_buffer_with_ops(
        state:               &state,
        desc:                &desc,
        requirements:        requirements,
        buffer:              buffer,
        offset:              0,
        backing_buffer:      {},
        placement:           {},
        owns_backing_buffer: false,
        ops:                 &ops,
    )) == gpu::SLOT_TABLE_FULL);
    assert(capture.create_count == 1 && capture.destroy_count == 1,
        "publication failure destroys the completed native object once");
    state.acceleration_structures.slots = {};
}

fn void acceleration_structure_destroy_checks_use_without_waiting() @test {
    AccelerationStructureCell[] cells =
        mem::new_array(AccelerationStructureCell, 3);
    defer mem::free(cells.ptr);
    AllocationCell[] allocations = mem::new_array(AllocationCell, 1);
    defer mem::free(allocations.ptr);
    VkAccelerationStructureDispatch dispatch = {
        .destroy_acceleration_structure =
            &count_acceleration_structure_destroy,
    };
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .acceleration_structure_operations = &dispatch,
        .acceleration_structures = {
            .slots = cells,
            .owner = 0xC5,
        },
        .allocations = {
            .slots = allocations,
            .owner = 0xC6,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    gpu::GpuAllocation placement = {
        .owner = state.allocations.owner,
        .generation = 1,
    };
    allocations[0] = {
        .generation = 1,
        .used = true,
        .value.live_placements = 1,
    };
    state.allocations.next_new = 1;
    AccelerationStructureSlot slot = {
        .native = (vk::AccelerationStructureKHR)(uptr)0xCAFE,
        .placement = placement,
        .desc.kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .build_eligibility =
            gpu::internal::AccelerationStructureBuildEligibility.BUILT,
    };
    gpu::AccelerationStructureHandle handle =
        state.acceleration_structures.alloc(slot)!!;
    acceleration_structure_native_destroy_count = 0;

    state.acceleration_structures.get(handle)!!.live_view_count = 1;
    assert(@catch(vk_destroy_acceleration_structure(&state, handle))
        == gpu::RESOURCE_IN_USE);
    assert(acceleration_structure_native_destroy_count == 0
            && allocations[0].value.live_placements == 1,
        "live views reject without native destruction or placement mutation");

    state.acceleration_structures.get(handle)!!.live_view_count = 0;
    state.acceleration_structures.get(handle)!!
        .tracked_command_references.store(1, AtomicOrdering.RELAXED);
    assert(@catch(vk_destroy_acceleration_structure(&state, handle))
        == gpu::RESOURCE_IN_USE);
    assert(acceleration_structure_native_destroy_count == 0,
        "tracked use rejects without a completion wait or native call");

    state.acceleration_structures.get(handle)!!
        .tracked_command_references.store(0, AtomicOrdering.RELAXED);
    vk_destroy_acceleration_structure(&state, handle)!!;
    assert(acceleration_structure_native_destroy_count == 1
            && allocations[0].value.live_placements == 0,
        "eligible destruction releases native and placement ownership once");
    assert(@catch(vk_destroy_acceleration_structure(&state, handle))
        == gpu::INVALID_HANDLE);
}

fn void acceleration_structure_address_is_blas_only_and_generation_checked() @test {
    AccelerationStructureCell[] cells =
        mem::new_array(AccelerationStructureCell, 2);
    defer mem::free(cells.ptr);
    VkAccelerationStructureDispatch dispatch = {
        .get_device_address = &return_acceleration_structure_address,
    };
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .acceleration_structure_operations = &dispatch,
        .acceleration_structures = {
            .slots = cells,
            .owner = 0xD5,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    gpu::AccelerationStructureHandle blas =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xB1A5,
            .desc.kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        })!!;
    gpu::AccelerationStructureHandle tlas =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0x71A5,
            .desc.kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        })!!;
    acceleration_structure_address_count = 0;

    assert(vk_get_acceleration_structure_address(&state, blas)!!
            == (gpu::GpuAddress)0x1234_5000
            && acceleration_structure_address_count == 1,
        "BLAS lookup returns an ordinary nonzero GpuAddress");
    assert(@catch(vk_get_acceleration_structure_address(&state, tlas))
        == gpu::INVALID_ARGUMENT);
    gpu::AccelerationStructureHandle foreign = blas;
    foreign.owner++;
    assert(@catch(vk_get_acceleration_structure_address(&state, foreign))
        == gpu::INVALID_HANDLE);
    assert(acceleration_structure_address_count == 1,
        "wrong-kind and foreign handles reach no native address query");
}

fn gpu::GpuSpan acceleration_structure_test_span(
    BufferHandle handle,
    usz size,
) {
    return {
        .owner = handle.owner,
        .index = handle.index,
        .generation = handle.generation,
        .size = size,
    };
}

fn void acceleration_structure_dynamic_multi_geometry_lowers_once() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 5);
    defer mem::free(buffer_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .buffers = { .slots = buffer_cells, .owner = 0xE1 },
        .ray_tracing_support = {
            .acceleration_structure_indirect_build = true,
        },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xE2,
        },
    };
    BufferUsage input_usage = {
        .addressable,
        .acceleration_structure_build_input,
    };
    BufferHandle vertices = state.buffers.alloc({
        .size = 144,
        .gpu_base = (gpu::GpuAddress)0x1000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    BufferHandle indices = state.buffers.alloc({
        .size = 24,
        .gpu_base = (gpu::GpuAddress)0x2000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    BufferHandle second_vertices = state.buffers.alloc({
        .size = 108,
        .gpu_base = (gpu::GpuAddress)0x3000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    BufferHandle scratch_buffer = state.buffers.alloc({
        .size = 2048,
        .gpu_base = (gpu::GpuAddress)0x4000,
        .usage = { .addressable, .storage },
        .access = { .compute },
    })!!;
    BufferHandle transform_buffer = state.buffers.alloc({
        .size = 12 * float::size,
        .gpu_base = (gpu::GpuAddress)0x5000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    gpu::internal::NormalizedAccelerationStructureGeometry[2] schema = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque = true },
            .max_vertex_count = 12,
            .max_primitive_count = 4,
            .index_type = gpu::AccelerationStructureIndexType.U16,
            .has_transform = true,
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .max_vertex_count = 9,
            .max_primitive_count = 3,
            .index_type = gpu::AccelerationStructureIndexType.NONE,
        },
    };
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .desc = {
                .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
                .geometries = schema[..],
                .requirements = {
                    .build_scratch_size = 2048,
                    .update_scratch_size = 1024,
                    .scratch_alignment = 256,
                },
            },
        })!!;
    vk::AccelerationStructureGeometryKHR[2] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[2] native_ranges;
    uint[2] maximum_counts;
    CommandScratch command_scratch = {
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
        .acceleration_structure_max_primitive_counts = maximum_counts[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .compute },
        .scratch = &command_scratch,
    };
    gpu::AccelerationStructureGeometryBuildDesc[2] inputs = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = acceleration_structure_test_span(vertices, 144),
                .vertex_stride = 12,
                .vertex_count = 12,
                .indices = acceleration_structure_test_span(indices, 24),
                .transform = acceleration_structure_test_span(
                    transform_buffer,
                    12 * float::size,
                ),
                .primitive_count = 4,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = acceleration_structure_test_span(
                    second_vertices,
                    108,
                ),
                .vertex_stride = 12,
                .vertex_count = 9,
                .primitive_count = 3,
            },
        },
    };
    gpu::AccelerationStructureBuildDesc build = {
        .destination = destination,
        .geometries = inputs[..],
        .scratch = acceleration_structure_test_span(scratch_buffer, 2048),
    };

    PreparedAccelerationStructureBuild prepared =
        prepare_acceleration_structure_build(
            &record,
            &build,
            false,
            true,
        )!!;
    assert(prepared.geometry_count == 2
            && command_scratch.acceleration_structure_geometry_count == 2
            && native_geometries[0].geometry_type
                == vk::GeometryTypeKHR.TRIANGLES_KHR
            && native_geometries[0].geometry.triangles.vertex_data.device_address
                == 0x1000
            && native_geometries[0].geometry.triangles.index_data.device_address
                == 0x2000
            && native_geometries[0].geometry.triangles.index_type
                == vk::IndexType.UINT16
            && native_geometries[0].geometry.triangles.transform_data.device_address
                == 0x5000
            && native_ranges[0].primitive_count == 4,
        "triangle dynamic input lowers exact addresses and primitive count");
    assert(native_geometries[1].geometry_type
                == vk::GeometryTypeKHR.TRIANGLES_KHR
            && native_geometries[1].geometry.triangles
                .vertex_data.device_address
                == 0x3000
            && native_geometries[1].geometry.triangles.vertex_stride == 12
            && native_ranges[1].primitive_count == 3,
        "each ordered triangle input lowers independently");

    prepared = prepare_acceleration_structure_build_mode(
        record:   &record,
        desc:     &build,
        update:   false,
        validate: true,
        indirect: true,
    )!!;
    fill_acceleration_structure_max_primitive_counts(
        &record,
        &build,
        &prepared,
    );
    assert(native_geometries[0].geometry.triangles.max_vertex == 11
            && native_geometries[1].geometry.triangles.max_vertex == 8
            && maximum_counts[0] == 4
            && maximum_counts[1] == 3,
        "indirect triangles keep exact vertex bounds and dense CPU maxima");

    inputs[0].triangles.transform = acceleration_structure_test_span(
        transform_buffer,
        12 * float::size - 1,
    );
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "triangle transforms require the complete 3-by-4 float matrix");
    inputs[0].triangles.transform = acceleration_structure_test_span(
        transform_buffer,
        12 * float::size,
    );

    inputs[1].triangles.vertex_stride = 8;
    command_scratch.acceleration_structure_geometry_count = 0;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )) == gpu::INVALID_ARGUMENT);
    assert(command_scratch.acceleration_structure_geometry_count == 0,
        "validation failure commits no logical native-array count");

    command_scratch.acceleration_structure_geometries =
        native_geometries[:1];
    command_scratch.acceleration_structure_ranges = native_ranges[:1];
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "allocator geometry overflow rejects before lowering mutation");

    AccelerationStructureBuildCapture capture;
    AccelerationStructureBuildCommandOps ops = {
        .user_data = &capture,
        .build = &capture_acceleration_structure_build,
    };
    record.pending_acceleration_structure_head =
        ACCELERATION_STRUCTURE_PENDING_NONE;
    assert(@catch(record_acceleration_structure_build_with_ops(
        &record,
        &build,
        false,
        true,
        true,
        &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(capture.calls == 0
            && command_scratch.reference_count == 0
            && command_scratch.acceleration_structure_geometry_count == 0
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "N+1 geometries reject before emission, references, lowering count, or pending-state mutation");

    command_scratch.acceleration_structure_geometries =
        native_geometries[..];
    command_scratch.acceleration_structure_ranges = native_ranges[..];
    inputs[1].triangles.vertex_stride = 12;
    record_acceleration_structure_build_with_ops(
        &record,
        &build,
        false,
        true,
        false,
        &ops,
    )!!;
    complete_pending_acceleration_structure_builds(&record);

    inputs[0].triangles.primitive_count = 3;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "BLAS updates preserve each completed primitive count");
    inputs[0].triangles.primitive_count = 4;
    inputs[0].triangles.vertex_count = 11;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "BLAS updates preserve each completed max vertex value");
    inputs[0].triangles.vertex_count = 12;
    inputs[0].triangles.transform = {};
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "BLAS updates preserve transform presence");

    gpu::internal::NormalizedAccelerationStructureGeometry[1] aabb_schema = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.AABBS,
            .flags = { .opaque = true },
            .max_primitive_count = 1,
        },
    };
    AccelerationStructureSlot* destination_slot =
        state.acceleration_structures.get(destination)!!;
    destination_slot.desc.geometries = aabb_schema[..];
    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    gpu::AccelerationStructureGeometryBuildDesc[1] aabb_inputs = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.AABBS,
            .aabbs = {
                .aabbs = acceleration_structure_test_span(vertices, 24),
                .stride = 24,
                .primitive_count = 1,
            },
        },
    };
    build.geometries = aabb_inputs[..];
    (void)prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )!!;
    assert(native_geometries[0].geometry_type
                == vk::GeometryTypeKHR.AABBS_KHR
            && native_geometries[0].geometry.aabbs.stride == 24
            && (uint)native_geometries[0].flags
                == (uint)vk::GeometryFlagBitsKHR.OPAQUE_BIT_KHR,
        "AABB dynamic input lowers its exact stride and geometry flags");
    PreparedAccelerationStructureBuild aabb_prepared =
        prepare_acceleration_structure_build_mode(
            record:   &record,
            desc:     &build,
            update:   false,
            validate: true,
            indirect: true,
        )!!;
    fill_acceleration_structure_max_primitive_counts(
        &record,
        &build,
        &aabb_prepared,
    );
    assert(maximum_counts[0] == 1,
        "indirect AABB lowering publishes its descriptor maximum");
    aabb_inputs[0].aabbs.stride = 8;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "AABB dynamic input requires a 24-byte aligned record stride");
}

fn void acceleration_structure_dynamic_tlas_uses_exact_instance_abi() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 2);
    defer mem::free(buffer_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .buffers = { .slots = buffer_cells, .owner = 0xF1 },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xF2,
        },
    };
    BufferHandle instances = state.buffers.alloc({
        .size = gpu::AccelerationStructureInstance::size * 4,
        .gpu_base = (gpu::GpuAddress)0x5000,
        .usage = { .addressable, .acceleration_structure_build_input },
        .access = { .graphics },
    })!!;
    BufferHandle scratch_buffer = state.buffers.alloc({
        .size = 512,
        .gpu_base = (gpu::GpuAddress)0x6000,
        .usage = { .addressable, .storage },
        .access = { .graphics },
    })!!;
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .desc = {
                .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 4,
                .requirements = {
                    .build_scratch_size = 512,
                    .scratch_alignment = 256,
                },
            },
        })!!;
    vk::AccelerationStructureGeometryKHR[1] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[1] native_ranges;
    CommandScratch command_scratch = {
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .graphics },
        .scratch = &command_scratch,
    };
    gpu::AccelerationStructureBuildDesc build = {
        .destination = destination,
        .instances = acceleration_structure_test_span(
            instances,
            gpu::AccelerationStructureInstance::size * 4,
        ),
        .instance_count = 4,
        .scratch = acceleration_structure_test_span(scratch_buffer, 512),
    };

    (void)prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )!!;
    assert(native_geometries[0].geometry_type
            == vk::GeometryTypeKHR.INSTANCES_KHR
            && native_geometries[0].geometry.instances.data.device_address
                == 0x5000
            && native_ranges[0].primitive_count == 4,
        "TLAS lowering consumes exact 64-byte instance records");
}

fn void acceleration_structure_indirect_tlas_is_fixed_and_transactional() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 3);
    defer mem::free(buffer_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .ray_tracing_support = {
            .acceleration_structure_indirect_build = true,
        },
        .buffers = { .slots = buffer_cells, .owner = 0xF21 },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xF22,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    BufferHandle instances = state.buffers.alloc({
        .size = gpu::AccelerationStructureInstance::size * 4,
        .gpu_base = (gpu::GpuAddress)0x5000,
        .usage = { .addressable, .acceleration_structure_build_input },
        .access = { .compute },
    })!!;
    BufferHandle scratch_buffer = state.buffers.alloc({
        .size = 512,
        .gpu_base = (gpu::GpuAddress)0x6000,
        .usage = { .addressable, .storage },
        .access = { .compute },
    })!!;
    BufferHandle ranges_buffer = state.buffers.alloc({
        .size = 32,
        .gpu_base = (gpu::GpuAddress)0x7000,
        .usage = { .addressable, .indirect },
        .access = { .compute },
    })!!;
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xF513,
            .desc = {
                .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 4,
                .flags = { .allow_update = true },
                .requirements = {
                    .build_scratch_size = 512,
                    .update_scratch_size = 256,
                    .scratch_alignment = 256,
                },
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    vk::AccelerationStructureGeometryKHR[1] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[1] native_ranges;
    uint[1] maximum_counts;
    CommandResourceReference[4] references;
    CommandScratch command_scratch = {
        .references = references[..],
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
        .acceleration_structure_max_primitive_counts = maximum_counts[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .compute },
        .scratch = &command_scratch,
        .pending_acceleration_structure_head =
            ACCELERATION_STRUCTURE_PENDING_NONE,
    };
    gpu::AccelerationStructureBuildDesc build = {
        .destination = destination,
        .instances = acceleration_structure_test_span(
            instances,
            gpu::AccelerationStructureInstance::size * 4,
        ),
        .instance_count = 4,
        .scratch = acceleration_structure_test_span(scratch_buffer, 512),
    };
    gpu::GpuSpan ranges = acceleration_structure_test_span(
        ranges_buffer,
        gpu::AccelerationStructureIndirectBuildRange::size,
    );
    AccelerationStructureIndirectBuildCapture capture;
    AccelerationStructureIndirectBuildCommandOps ops = {
        .user_data = &capture,
        .build_indirect = &capture_acceleration_structure_indirect_build,
    };

    state.ray_tracing_support.acceleration_structure_indirect_build = false;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    state.ray_tracing_support.acceleration_structure_indirect_build = true;

    ranges.size = gpu::AccelerationStructureIndirectBuildRange::size - 1;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    ranges = acceleration_structure_test_span(
        ranges_buffer,
        gpu::AccelerationStructureIndirectBuildRange::size,
    );
    ranges.owner++;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_HANDLE);
    ranges = acceleration_structure_test_span(ranges_buffer, 32);
    ranges.offset = usz::max - 7;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    ranges = acceleration_structure_test_span(ranges_buffer, 32);
    ranges.offset = 1;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    ranges = acceleration_structure_test_span(
        ranges_buffer,
        gpu::AccelerationStructureIndirectBuildRange::size,
    );
    state.buffers.get(ranges_buffer)!!.usage.indirect = false;
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    state.buffers.get(ranges_buffer)!!.usage.indirect = true;
    assert(capture.calls == 0 && command_scratch.reference_count == 0);

    command_scratch.references = references[:3];
    assert(@catch(record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    AccelerationStructureSlot* destination_slot =
        state.acceleration_structures.get(destination)!!;
    assert(capture.calls == 0
            && command_scratch.reference_count == 0
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT
            && !destination_slot.built_counts_are_maxima,
        "reference failure rolls back before indirect emission and state");

    command_scratch.references = references[..];
    record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           false,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )!!;
    assert(capture.calls == 1
            && capture.info.mode
                == vk::BuildAccelerationStructureModeKHR.BUILD_KHR
            && capture.range_address == 0x7000
            && capture.range_stride
                == gpu::AccelerationStructureIndirectBuildRange::size
            && capture.max_primitive_count == 4,
        "one indirect TLAS build emits one fixed packet and maximum count");
    assert(command_scratch.reference_count == 4
            && destination_slot.built_counts_are_maxima
            && destination_slot.built_instance_count == 4
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.PENDING,
        "full indirect build retains every explicit owner and records maxima");
    complete_pending_acceleration_structure_builds(&record);
    clear_command_scratch_references(&command_scratch);

    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "direct update rejects an indirect build with unknown actual counts");

    capture = {};
    record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           true,
        validate:         true,
        track_references: true,
        ops:              &ops,
    )!!;
    assert(capture.calls == 1
            && capture.info.mode
                == vk::BuildAccelerationStructureModeKHR.UPDATE_KHR
            && destination_slot.built_counts_are_maxima,
        "indirect update preserves maximum-count knowledge");
    clear_command_scratch_references(&command_scratch);

    destination_slot.built_counts_are_maxima = false;
    capture = {};
    record_acceleration_structure_indirect_build_with_ops(
        record:           &record,
        desc:             &build,
        ranges:           ranges,
        update:           true,
        validate:         false,
        track_references: false,
        ops:              &ops,
    )!!;
    assert(capture.calls == 1 && !destination_slot.built_counts_are_maxima,
        "an indirect update preserves prior exact-count knowledge");
    destination_slot.built_counts_are_maxima = true;

    capture = {};
    mem::@scoped(&allocators::NULL_ALLOCATOR) {
        for (uint i = 0; i < 64; i++) {
            record_acceleration_structure_indirect_build_with_ops(
                record:           &record,
                desc:             &build,
                ranges:           ranges,
                update:           true,
                validate:         false,
                track_references: false,
                ops:              &ops,
            )!!;
        }
    };
    assert(capture.calls == 64 && command_scratch.reference_count == 0,
        "warmed trusted indirect updates allocate and retain nothing");
}

fn void acceleration_structure_scratch_rejects_destination_overlap() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 1);
    defer mem::free(buffer_cells.ptr);
    AllocationCell[] allocation_cells = mem::new_array(AllocationCell, 1);
    defer mem::free(allocation_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .buffers = { .slots = buffer_cells, .owner = 0xF31 },
        .allocations = { .slots = allocation_cells, .owner = 0xF32 },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xF33,
        },
    };
    BufferHandle instances = state.buffers.alloc({
        .size = gpu::AccelerationStructureInstance::size,
        .gpu_base = (gpu::GpuAddress)0x7000,
        .usage = { .addressable, .acceleration_structure_build_input },
        .access = { .compute },
    })!!;
    gpu::GpuAllocation placement = state.allocations.alloc({
        .size = 1024,
        .backing = {
            .size = 1024,
            .gpu_base = (gpu::GpuAddress)0x8000,
            .usage = { .addressable, .storage,
                .acceleration_structure_storage },
            .access = { .compute },
        },
    })!!;
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .placement = placement,
            .placement_offset = 256,
            .desc = {
                .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 1,
                .requirements = {
                    .storage_size = 512,
                    .build_scratch_size = 256,
                    .scratch_alignment = 256,
                },
            },
        })!!;
    vk::AccelerationStructureGeometryKHR[1] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[1] native_ranges;
    CommandScratch command_scratch = {
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .compute },
        .scratch = &command_scratch,
    };
    gpu::AccelerationStructureBuildDesc build = {
        .destination = destination,
        .instances = acceleration_structure_test_span(
            instances,
            gpu::AccelerationStructureInstance::size,
        ),
        .instance_count = 1,
        .scratch = {
            .owner = placement.owner,
            .index = placement.index,
            .generation = placement.generation,
            .offset = 512,
            .size = 256,
        },
    };

    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "scratch cannot overlap placed acceleration-structure storage");
    build.scratch.offset = 768;
    (void)prepare_acceleration_structure_build(
        &record,
        &build,
        false,
        true,
    )!!;
}

fn void acceleration_structure_build_tracks_emits_and_transitions_atomically() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 2);
    defer mem::free(buffer_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 1);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .buffers = { .slots = buffer_cells, .owner = 0xA11 },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xA12,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    BufferHandle instances = state.buffers.alloc({
        .size = gpu::AccelerationStructureInstance::size * 2,
        .gpu_base = (gpu::GpuAddress)0x7000,
        .usage = { .addressable, .acceleration_structure_build_input },
        .access = { .compute },
    })!!;
    BufferHandle scratch_buffer = state.buffers.alloc({
        .size = 512,
        .gpu_base = (gpu::GpuAddress)0x8000,
        .usage = { .addressable, .storage },
        .access = { .compute },
    })!!;
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xA513,
            .desc = {
                .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 2,
                .flags = { .allow_update = true },
                .requirements = {
                    .build_scratch_size = 512,
                    .update_scratch_size = 256,
                    .scratch_alignment = 256,
                },
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
            .built_counts_are_maxima = true,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    vk::AccelerationStructureGeometryKHR[1] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[1] native_ranges;
    CommandResourceReference[3] references;
    CommandScratch command_scratch = {
        .references = references[..],
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .identity = { .index = 5 },
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .compute },
        .scratch = &command_scratch,
        .pending_acceleration_structure_head =
            ACCELERATION_STRUCTURE_PENDING_NONE,
    };
    gpu::AccelerationStructureBuildDesc build = {
        .destination = destination,
        .instances = acceleration_structure_test_span(
            instances,
            gpu::AccelerationStructureInstance::size * 2,
        ),
        .instance_count = 2,
        .scratch = acceleration_structure_test_span(scratch_buffer, 512),
    };
    AccelerationStructureBuildCapture capture;
    AccelerationStructureBuildCommandOps ops = {
        .user_data = &capture,
        .build = &capture_acceleration_structure_build,
    };

    record_acceleration_structure_build_with_ops(
        &record,
        &build,
        false,
        true,
        true,
        &ops,
    )!!;
    AccelerationStructureSlot* destination_slot =
        state.acceleration_structures.get(destination)!!;
    assert(capture.calls == 1
            && capture.info.mode
                == vk::BuildAccelerationStructureModeKHR.BUILD_KHR
            && capture.info.dst_acceleration_structure
                == destination_slot.native
            && capture.info.scratch_data.device_address == 0x8000
            && capture.range.primitive_count == 2,
        "one full build emits one exact native build command");
    assert(command_scratch.reference_count == 3
            && destination_slot.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 1
            && state.buffers.get(instances)!!.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 1
            && state.buffers.get(scratch_buffer)!!
                .tracked_command_references.load(AtomicOrdering.RELAXED) == 1,
        "full validation retains destination, instances, and scratch");
    assert(destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.PENDING
            && !destination_slot.built_counts_are_maxima
            && record.pending_acceleration_structure_head
                == destination.index,
        "first build links its destination without allocation");

    abandon_pending_acceleration_structure_builds(&record, false);
    assert(destination_slot.build_eligibility
        == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT);
    activate_pending_acceleration_structure_builds(&record);
    assert(destination_slot.build_eligibility
        == gpu::internal::AccelerationStructureBuildEligibility.PENDING);
    complete_pending_acceleration_structure_builds(&record);
    assert(destination_slot.build_eligibility
            == gpu::internal::AccelerationStructureBuildEligibility.BUILT
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "failed submit, accepted submit, and retirement transition exactly");
    clear_command_scratch_references(&command_scratch);
    assert(destination_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);

    build.instance_count = 1;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &build,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "TLAS updates preserve the completed build instance count");
    build.instance_count = 2;

    command_scratch.acceleration_structure_geometry_count = 0;
    capture = {};
    record_acceleration_structure_build_with_ops(
        &record,
        &build,
        true,
        true,
        true,
        &ops,
    )!!;
    assert(capture.calls == 1
            && capture.info.mode
                == vk::BuildAccelerationStructureModeKHR.UPDATE_KHR
            && capture.info.src_acceleration_structure
                == destination_slot.native
            && capture.info.dst_acceleration_structure
                == destination_slot.native
            && capture.info.scratch_data.device_address == 0x8000,
        "update emits one in-place native command with update scratch");
    assert(command_scratch.reference_count == 3
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.BUILT
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "update retains resources without changing completed build state");
    clear_command_scratch_references(&command_scratch);

    capture = {};
    mem::@scoped(&allocators::NULL_ALLOCATOR) {
        for (uint i = 0; i < 64; i++) {
            command_scratch.acceleration_structure_geometry_count = 0;
            record_acceleration_structure_build_with_ops(
                &record,
                &build,
                true,
                true,
                true,
                &ops,
            )!!;
            clear_command_scratch_references(&command_scratch);
        }
    };
    assert(capture.calls == 64
            && command_scratch.reference_count == 0
            && command_scratch.acceleration_structure_geometry_count == 1,
        "warm in-place updates reuse fixed lowering and reference storage with no temporary allocation");

    command_scratch.acceleration_structure_geometry_count = 0;
    capture = {};
    record_acceleration_structure_build_with_ops(
        &record,
        &build,
        true,
        false,
        false,
        &ops,
    )!!;
    assert(capture.calls == 1 && command_scratch.reference_count == 0,
        "trusted update retains the mandatory safety floor without reference work");

    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    command_scratch.acceleration_structure_geometry_count = 0;
    capture = {};
    assert(@catch(record_acceleration_structure_build_with_ops(
        &record,
        &build,
        true,
        true,
        true,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.BUILT;
    destination_slot.desc.flags.allow_update = false;
    assert(@catch(record_acceleration_structure_build_with_ops(
        &record,
        &build,
        true,
        true,
        true,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    destination_slot.desc.flags.allow_update = true;
    assert(capture.calls == 0 && command_scratch.reference_count == 0
            && command_scratch.acceleration_structure_geometry_count == 0,
        "update rejects unbuilt and static destinations before mutation");

    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    command_scratch.references = references[:2];
    command_scratch.reference_count = 0;
    command_scratch.acceleration_structure_geometry_count = 0;
    capture = {};
    assert(@catch(record_acceleration_structure_build_with_ops(
        &record,
        &build,
        false,
        true,
        true,
        &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(capture.calls == 0 && command_scratch.reference_count == 0
            && command_scratch.acceleration_structure_geometry_count == 0
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        "reference exhaustion rolls back before emission and pending state");
}

fn void acceleration_structure_clone_tracks_emits_and_transitions_atomically() @test {
    BufferCell[] buffer_cells = mem::new_array(BufferCell, 3);
    defer mem::free(buffer_cells.ptr);
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 5);
    defer mem::free(acceleration_structure_cells.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .buffers = { .slots = buffer_cells, .owner = 0xC11 },
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xC12,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();

    gpu::internal::NormalizedAccelerationStructureGeometry[1]
        source_geometries = {
            {
                .kind                  = gpu::AccelerationStructureGeometryKind.TRIANGLES,
                .flags                 = { .opaque = true },
                .max_vertex_count      = 3,
                .max_primitive_count   = 1,
                .index_type            = gpu::AccelerationStructureIndexType.NONE,
                .has_transform         = true,
                .built_vertex_count    = 3,
                .built_primitive_count = 1,
                .built_with_transform  = true,
            },
        };
    gpu::internal::NormalizedAccelerationStructureGeometry[1]
        destination_geometries = {
            {
                .kind                = gpu::AccelerationStructureGeometryKind.TRIANGLES,
                .flags               = { .opaque = true },
                .max_vertex_count    = 3,
                .max_primitive_count = 1,
                .index_type          = gpu::AccelerationStructureIndexType.NONE,
                .has_transform       = true,
            },
        };
    gpu::internal::NormalizedAccelerationStructureGeometry[1]
        incompatible_geometries = {
            {
                .kind                = gpu::AccelerationStructureGeometryKind.TRIANGLES,
                .flags               = { .opaque = true },
                .max_vertex_count    = 6,
                .max_primitive_count = 2,
                .index_type          = gpu::AccelerationStructureIndexType.NONE,
            },
        };
    gpu::AccelerationStructureRequirements requirements = {
        .storage_size       = 4096,
        .update_scratch_size = 256,
        .scratch_alignment  = 256,
    };
    gpu::AccelerationStructureHandle source =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xC501,
            .desc = {
                .kind         = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
                .geometries   = source_geometries[..],
                .flags        = { .allow_update = true },
                .requirements = requirements,
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.BUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    gpu::AccelerationStructureHandle destination =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xC502,
            .desc = {
                .kind         = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
                .geometries   = destination_geometries[..],
                .flags        = { .allow_update = true },
                .requirements = requirements,
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    gpu::AccelerationStructureHandle incompatible =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xC503,
            .desc = {
                .kind         = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
                .geometries   = incompatible_geometries[..],
                .flags        = { .allow_update = true },
                .requirements = requirements,
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    gpu::AccelerationStructureHandle tlas_source =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xC504,
            .desc = {
                .kind               = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 8,
                .flags              = { .allow_update = true },
                .requirements       = requirements,
            },
            .built_instance_count = 7,
            .built_counts_are_maxima = true,
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.BUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;
    gpu::AccelerationStructureHandle tlas_destination =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xC505,
            .desc = {
                .kind               = gpu::AccelerationStructureKind.TOP_LEVEL,
                .max_instance_count = 8,
                .flags              = { .allow_update = true },
                .requirements       = requirements,
            },
            .build_eligibility =
                gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
            .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
        })!!;

    CommandResourceReference[2] references;
    vk::AccelerationStructureGeometryKHR[1] native_geometries;
    vk::AccelerationStructureBuildRangeInfoKHR[1] native_ranges;
    CommandScratch command_scratch = {
        .references = references[..],
        .acceleration_structure_geometries = native_geometries[..],
        .acceleration_structure_ranges = native_ranges[..],
    };
    CommandRecord record = {
        .device_state = &state,
        .state = gpu::internal::CommandState.RECORDING,
        .queue.roles = { .compute },
        .scratch = &command_scratch,
        .pending_acceleration_structure_head =
            ACCELERATION_STRUCTURE_PENDING_NONE,
    };
    AccelerationStructureCloneCapture capture;
    AccelerationStructureCloneCommandOps ops = {
        .user_data = &capture,
        .clone = &capture_acceleration_structure_clone,
    };

    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      source,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    AccelerationStructureSlot* source_slot =
        state.acceleration_structures.get(source)!!;
    source_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      destination,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    source_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.PENDING;
    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      destination,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    source_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.BUILT;
    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      incompatible,
        track_references: true,
        ops:              &ops,
    )) == gpu::INVALID_ARGUMENT);
    AccelerationStructureSlot* destination_slot =
        state.acceleration_structures.get(destination)!!;
    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.BUILT;
    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      destination,
        track_references: true,
        ops:              &ops,
    )) == gpu::RESOURCE_IN_USE);
    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    assert(capture.calls == 0
            && command_scratch.reference_count == 0
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "invalid clone endpoints cause no observable mutation");

    record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      destination,
        track_references: true,
        ops:              &ops,
    )!!;
    assert(capture.calls == 1
            && capture.info.src == source_slot.native
            && capture.info.dst == destination_slot.native
            && capture.info.mode
                == vk::CopyAccelerationStructureModeKHR.CLONE_KHR,
        "one clone emits the exact source, destination, and native mode");
    assert(command_scratch.reference_count == 2
            && source_slot.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 1
            && destination_slot.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 1,
        "full clone retains exactly its two explicit endpoints");
    assert(destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.PENDING
            && destination_slot.desc.geometries[0].built_vertex_count == 3
            && destination_slot.desc.geometries[0].built_primitive_count == 1
            && destination_slot.desc.geometries[0].built_with_transform,
        "clone copies completed BLAS shape before linking the destination");

    abandon_pending_acceleration_structure_builds(&record, false);
    assert(destination_slot.build_eligibility
        == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT);
    activate_pending_acceleration_structure_builds(&record);
    assert(destination_slot.build_eligibility
        == gpu::internal::AccelerationStructureBuildEligibility.PENDING);
    complete_pending_acceleration_structure_builds(&record);
    assert(destination_slot.build_eligibility
            == gpu::internal::AccelerationStructureBuildEligibility.BUILT
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "discard, accepted retry, and retirement use the existing lifecycle");
    clear_command_scratch_references(&command_scratch);

    BufferUsage input_usage = {
        .addressable,
        .acceleration_structure_build_input,
    };
    BufferHandle vertices = state.buffers.alloc({
        .size = 36,
        .gpu_base = (gpu::GpuAddress)0x9000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    BufferHandle scratch_buffer = state.buffers.alloc({
        .size = 256,
        .gpu_base = (gpu::GpuAddress)0xA000,
        .usage = { .addressable, .storage },
        .access = { .compute },
    })!!;
    BufferHandle transform_buffer = state.buffers.alloc({
        .size = 12 * float::size,
        .gpu_base = (gpu::GpuAddress)0xB000,
        .usage = input_usage,
        .access = { .compute },
    })!!;
    gpu::AccelerationStructureGeometryBuildDesc[1] update_geometries = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = acceleration_structure_test_span(vertices, 36),
                .vertex_stride = 12,
                .vertex_count = 3,
                .transform = acceleration_structure_test_span(
                    transform_buffer,
                    12 * float::size,
                ),
                .primitive_count = 1,
            },
        },
    };
    gpu::AccelerationStructureBuildDesc update = {
        .destination = destination,
        .geometries = update_geometries[..],
        .scratch = acceleration_structure_test_span(scratch_buffer, 256),
    };
    (void)prepare_acceleration_structure_build(
        &record,
        &update,
        true,
        true,
    )!!;
    destination_slot.built_counts_are_maxima = true;
    assert(@catch(prepare_acceleration_structure_build(
        &record,
        &update,
        true,
        true,
    )) == gpu::INVALID_ARGUMENT,
        "a direct update requires exact completed build counts");
    destination_slot.built_counts_are_maxima = false;
    command_scratch.acceleration_structure_geometry_count = 0;

    record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           tlas_source,
        destination:      tlas_destination,
        track_references: false,
        ops:              &ops,
    )!!;
    AccelerationStructureSlot* tlas_destination_slot =
        state.acceleration_structures.get(tlas_destination)!!;
    assert(tlas_destination_slot.built_instance_count == 7
            && tlas_destination_slot.built_counts_are_maxima
            && tlas_destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.PENDING,
        "TLAS clone copies its completed instance shape");
    abandon_pending_acceleration_structure_builds(&record, true);

    command_scratch.references = references[:1];
    command_scratch.reference_count = 0;
    destination_slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT;
    capture = {};
    assert(@catch(record_acceleration_structure_clone_with_ops(
        record:           &record,
        source:           source,
        destination:      destination,
        track_references: true,
        ops:              &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(capture.calls == 0
            && command_scratch.reference_count == 0
            && source_slot.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 0
            && destination_slot.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) == 0
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT
            && record.pending_acceleration_structure_head
                == ACCELERATION_STRUCTURE_PENDING_NONE,
        "second-endpoint retention failure rolls back before emission");

    command_scratch.references = references[..];
    capture = {};
    mem::@scoped(&allocators::NULL_ALLOCATOR) {
        for (uint i = 0; i < 64; i++) {
            record_acceleration_structure_clone_with_ops(
                record:           &record,
                source:           source,
                destination:      destination,
                track_references: false,
                ops:              &ops,
            )!!;
            abandon_pending_acceleration_structure_builds(&record, true);
        }
    };
    assert(capture.calls == 64
            && command_scratch.reference_count == 0
            && destination_slot.build_eligibility
                == gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        "warm trusted clone uses fixed storage and performs no allocation");
}

fn void acceleration_structure_views_are_transactional_and_generation_checked() @test {
    AccelerationStructureCell[] acceleration_structure_cells =
        mem::new_array(AccelerationStructureCell, 2);
    defer mem::free(acceleration_structure_cells.ptr);
    AccelerationStructureDescriptorCell[] descriptor_cells =
        mem::new_array(AccelerationStructureDescriptorCell, 1);
    defer mem::free(descriptor_cells.ptr);
    uint[] descriptor_free = mem::new_array(uint, 1);
    defer mem::free(descriptor_free.ptr);
    VkDeviceState state = {
        .acceleration_structures_enabled = true,
        .acceleration_structures = {
            .slots = acceleration_structure_cells,
            .owner = 0xD51,
        },
        .shader_heap.descriptors = {
            .acceleration_structure_view_owner = 0xD52,
            .acceleration_structure_cells = descriptor_cells,
            .acceleration_structure_free = descriptor_free,
        },
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    gpu::AccelerationStructureHandle top_level =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0x715,
            .desc.kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        })!!;
    gpu::AccelerationStructureHandle bottom_level =
        state.acceleration_structures.alloc({
            .native = (vk::AccelerationStructureKHR)(uptr)0xB15,
            .desc.kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        })!!;
    AccelerationStructureDescriptorCapture capture;
    AccelerationStructureDescriptorOps ops = {
        .user_data = &capture,
        .write = &capture_acceleration_structure_descriptor,
    };

    gpu::AccelerationStructureView first =
        create_acceleration_structure_descriptor_view_with_ops(
            &state,
            top_level,
            &ops,
        )!!;
    assert(first.owner == 0xD52 && first.index.value == 1
            && first.generation == 1 && capture.calls == 1
            && capture.slot == 0
            && capture.acceleration_structure
                == state.acceleration_structures.get(top_level)!!.native
            && state.acceleration_structures.get(top_level)!!
                .live_view_count == 1,
        "TLAS publication writes slot-plus-one and retains one live view");
    assert(@catch(create_acceleration_structure_descriptor_view_with_ops(
        &state,
        top_level,
        &ops,
    )) == gpu::DESCRIPTOR_HEAP_FULL);
    assert(@catch(create_acceleration_structure_descriptor_view_with_ops(
        &state,
        bottom_level,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(capture.calls == 1
            && state.acceleration_structures.get(top_level)!!
                .live_view_count == 1,
        "heap exhaustion and BLAS rejection publish nothing");

    destroy_acceleration_structure_descriptor_view(&state, first)!!;
    gpu::AccelerationStructureView second =
        create_acceleration_structure_descriptor_view_with_ops(
            &state,
            top_level,
            &ops,
        )!!;
    assert(second.index == first.index
            && second.generation > first.generation,
        "retired slots are reused only with a new generation");
    assert(@catch(destroy_acceleration_structure_descriptor_view(
        &state,
        first,
    )) == gpu::INVALID_HANDLE);
    assert(@catch(destroy_acceleration_structure_descriptor_view(
        &state,
        { .owner = 0xBAD, .index = second.index,
            .generation = second.generation },
    )) == gpu::INVALID_HANDLE);
    assert(descriptor_cells[0].used
            && state.acceleration_structures.get(top_level)!!
                .live_view_count == 1,
        "stale and foreign views cannot retire the current descriptor");
    destroy_acceleration_structure_descriptor_view(&state, second)!!;
    assert(state.acceleration_structures.get(top_level)!!
            .live_view_count == 0,
        "view destruction releases exactly one TLAS relationship");
}

fn void acceleration_structure_model_keeps_capacity_and_state_backend_free() @test {
    gpu::internal::NormalizedAccelerationStructureGeometry[2] geometries = {
        {
            .kind                = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .max_vertex_count    = 12,
            .max_primitive_count = 4,
            .index_type          = gpu::AccelerationStructureIndexType.U16,
        },
        {
            .kind                = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .max_vertex_count    = 9,
            .max_primitive_count = 3,
            .index_type          = gpu::AccelerationStructureIndexType.U32,
        },
    };
    gpu::internal::NormalizedAccelerationStructureDesc desc = {
        .kind       = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = geometries[..],
        .flags      = { .allow_update = true },
        .requirements = {
            .storage_size       = 4096,
            .storage_alignment  = 256,
            .build_scratch_size = 2048,
            .update_scratch_size = 1024,
            .scratch_alignment  = 256,
        },
    };

    assert(desc.geometries.len == 2
            && desc.geometries[0].max_vertex_count == 12
            && desc.geometries[1].max_primitive_count == 3
            && desc.flags.allow_update
            && desc.requirements.update_scratch_size == 1024,
        "the normalized model retains immutable schema and requirements");
}

fn void acceleration_structure_table_is_generation_and_owner_checked() @test {
    AccelerationStructureTable table = {
        .slots = mem::new_array(AccelerationStructureCell, 2),
        .owner = 0xA5,
    };
    defer mem::free(table.slots.ptr);
    AccelerationStructureSlot slot = {
        .desc.kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .desc.max_instance_count = 8,
        .build_eligibility =
            gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        .pending_next = ACCELERATION_STRUCTURE_PENDING_NONE,
    };

    gpu::AccelerationStructureHandle first = table.alloc(slot)!!;
    assert(table.get(first)!!.desc.max_instance_count == 8);

    gpu::AccelerationStructureHandle foreign = first;
    foreign.owner++;
    assert(@catch(table.get(foreign)) == gpu::INVALID_HANDLE);

    AccelerationStructureSlot released = table.free(first)!!;
    assert(released.desc.kind == gpu::AccelerationStructureKind.TOP_LEVEL);
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE);

    gpu::AccelerationStructureHandle second = table.alloc(slot)!!;
    assert(second.index == first.index
            && second.generation != first.generation,
        "slot reuse changes the public generation");
}

fn void acceleration_structure_table_retires_exhausted_generations() @test {
    AccelerationStructureTable table = {
        .slots = mem::new_array(AccelerationStructureCell, 1),
        .owner = 0xA6,
    };
    defer mem::free(table.slots.ptr);
    table.slots[0] = {
        .generation = HANDLE_GENERATION_MAX - 1,
        .used       = true,
    };
    table.next_new = 1;
    gpu::AccelerationStructureHandle final = {
        .owner      = table.owner,
        .generation = HANDLE_GENERATION_MAX - 1,
    };

    (void)table.free(final)!!;
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "a generation-exhausted slot has no successor identity");
}
