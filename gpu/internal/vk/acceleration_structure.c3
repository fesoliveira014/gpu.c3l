module gpu::internal::vk @private;

import gpu @public;
import gpu::internal @public;
import vk;
import std::core::mem;
import std::core::mem::alloc;
import std::atomic;
import std::atomic::types;

const usz MAX_ACCELERATION_STRUCTURES = 4096;
const uint ACCELERATION_STRUCTURE_FREE_NONE = 0;
const uint ACCELERATION_STRUCTURE_PENDING_NONE = uint::max;
const usz ACCELERATION_STRUCTURE_STORAGE_ALIGNMENT = 256;

alias AccelerationStructureBuildSizesFn = fn void (
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureBuildTypeKHR build_type,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_info,
    uint* max_primitive_counts,
    vk::AccelerationStructureBuildSizesInfoKHR* size_info,
);

struct AccelerationStructureRequirementsOps {
    void*                             user_data;
    AccelerationStructureBuildSizesFn get_build_sizes;
}

alias AccelerationStructureCreateFn = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureCreateInfoKHR* create_info,
    vk::AccelerationStructureKHR* out_acceleration_structure,
);
alias AccelerationStructureDestroyFn = fn void (
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureKHR acceleration_structure,
);

struct AccelerationStructureNativeOps {
    void*                          user_data;
    AccelerationStructureCreateFn  create;
    AccelerationStructureDestroyFn destroy;
}

struct PreparedAccelerationStructureBuild {
    AccelerationStructureSlot*       destination;
    gpu::AccelerationStructureHandle destination_handle;
    ResolvedSpan                     scratch;
    gpu::GpuSpan                     scratch_span;
    uint                             geometry_count;
    bool                             update;
}

alias AccelerationStructureBuildCommandFn = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint info_count,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_infos,
    vk::AccelerationStructureBuildRangeInfoKHR** build_ranges,
);

struct AccelerationStructureBuildCommandOps {
    void*                               user_data;
    AccelerationStructureBuildCommandFn build;
}

fn void set_pending_acceleration_structure_builds(
    CommandRecord* record,
    gpu::internal::AccelerationStructureBuildEligibility state,
    bool clear,
) {
    if (record == null || record.device_state == null) return;
    uint index = record.pending_acceleration_structure_head;
    while (index != ACCELERATION_STRUCTURE_PENDING_NONE) {
        if (index >= record.device_state.acceleration_structures.next_new) {
            break;
        }
        AccelerationStructureCell* cell =
            &record.device_state.acceleration_structures.slots[index];
        if (!cell.used) break;
        uint next = cell.value.pending_next;
        cell.value.build_eligibility = state;
        if (clear) {
            cell.value.pending_next = ACCELERATION_STRUCTURE_PENDING_NONE;
        }
        index = next;
    }
    if (clear) {
        record.pending_acceleration_structure_head =
            ACCELERATION_STRUCTURE_PENDING_NONE;
    }
}

fn void activate_pending_acceleration_structure_builds(
    CommandRecord* record,
) {
    set_pending_acceleration_structure_builds(
        record,
        gpu::internal::AccelerationStructureBuildEligibility.PENDING,
        false,
    );
}

fn void abandon_pending_acceleration_structure_builds(
    CommandRecord* record,
    bool clear,
) {
    set_pending_acceleration_structure_builds(
        record,
        gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        clear,
    );
}

fn void complete_pending_acceleration_structure_builds(
    CommandRecord* record,
) {
    set_pending_acceleration_structure_builds(
        record,
        gpu::internal::AccelerationStructureBuildEligibility.BUILT,
        true,
    );
}

fn void? link_pending_acceleration_structure_build(
    CommandRecord* record,
    PreparedAccelerationStructureBuild* prepared,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    AccelerationStructureSlot* slot = prepared.destination;
    if (slot.build_eligibility
        != gpu::internal::AccelerationStructureBuildEligibility.UNBUILT) {
        return gpu::RESOURCE_IN_USE~;
    }
    if (slot.desc.kind == gpu::AccelerationStructureKind.TOP_LEVEL) {
        slot.built_instance_count = desc.instance_count;
    } else {
        foreach (i, &input : desc.geometries) {
            gpu::internal::NormalizedAccelerationStructureGeometry* schema =
                &slot.desc.geometries[i];
            if (schema.kind
                == gpu::AccelerationStructureGeometryKind.TRIANGLES) {
                schema.built_vertex_count = input.triangles.vertex_count;
                schema.built_primitive_count =
                    input.triangles.primitive_count;
                schema.built_with_transform =
                    !gpu_span_is_zero(input.triangles.transform);
            } else {
                schema.built_primitive_count = input.aabbs.primitive_count;
            }
        }
    }
    slot.build_eligibility =
        gpu::internal::AccelerationStructureBuildEligibility.PENDING;
    slot.pending_next = record.pending_acceleration_structure_head;
    record.pending_acceleration_structure_head =
        prepared.destination_handle.index;
}

fn void? track_acceleration_structure_build_references(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    VkDeviceState* state = record.device_state;
    track_command_reference(
        state,
        record,
        desc.destination.owner,
        desc.destination.index,
        desc.destination.generation,
    )!;
    foreach (&geometry : desc.geometries) {
        if (geometry.kind
            == gpu::AccelerationStructureGeometryKind.TRIANGLES) {
            track_span_command_reference(
                state,
                record,
                &geometry.triangles.vertices,
            )!;
            if (!gpu_span_is_zero(geometry.triangles.indices)) {
                track_span_command_reference(
                    state,
                    record,
                    &geometry.triangles.indices,
                )!;
            }
            if (!gpu_span_is_zero(geometry.triangles.transform)) {
                track_span_command_reference(
                    state,
                    record,
                    &geometry.triangles.transform,
                )!;
            }
        } else {
            track_span_command_reference(
                state,
                record,
                &geometry.aabbs.aabbs,
            )!;
        }
    }
    if (!gpu_span_is_zero(desc.instances)) {
        track_span_command_reference(state, record, &desc.instances)!;
    }
    track_span_command_reference(state, record, &desc.scratch)!;
}

fn void acceleration_structure_build_command_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint info_count,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_infos,
    vk::AccelerationStructureBuildRangeInfoKHR** build_ranges,
) {
    VkAccelerationStructureDispatch* dispatch = (VkAccelerationStructureDispatch*)user_data;
    dispatch.cmd_build(
        command_buffer,
        info_count,
        build_infos,
        build_ranges,
    );
}

fn void emit_acceleration_structure_build(
    CommandRecord* record,
    PreparedAccelerationStructureBuild* prepared,
    AccelerationStructureBuildCommandOps* ops,
) {
    vk::AccelerationStructureBuildGeometryInfoKHR build_info =
        vk::acceleration_structure_build_geometry_info_khr()
            .set_type(acceleration_structure_type_to_vk(
                prepared.destination.desc.kind,
            ))
            .set_flags(acceleration_structure_flags_to_vk(
                prepared.destination.desc.flags,
            ))
            .set_mode(prepared.update
                ? vk::BuildAccelerationStructureModeKHR.UPDATE_KHR
                : vk::BuildAccelerationStructureModeKHR.BUILD_KHR)
            .set_dst_acceleration_structure(prepared.destination.native)
            .set_geometries(
                record.scratch.acceleration_structure_geometries[
                    :prepared.geometry_count
                ],
            )
            .set_scratch_data({ .device_address =
                (vk::DeviceAddress)prepared.scratch.address });
    if (prepared.update) {
        build_info = build_info.set_src_acceleration_structure(
            prepared.destination.native,
        );
    }
    vk::AccelerationStructureBuildRangeInfoKHR* ranges =
        record.scratch.acceleration_structure_ranges.ptr;
    ops.build(
        ops.user_data,
        record.command_buffer,
        1,
        &build_info,
        &ranges,
    );
}

fn void? record_acceleration_structure_build_with_ops(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
    bool update,
    bool validate,
    bool track_references,
    AccelerationStructureBuildCommandOps* ops,
) {
    uint geometry_checkpoint =
        record.scratch.acceleration_structure_geometry_count;
    PreparedAccelerationStructureBuild prepared =
        prepare_acceleration_structure_build(
            record,
            desc,
            update,
            validate,
        )!;
    uint reference_checkpoint = record.scratch.reference_count;
    defer catch {
        record.scratch.acceleration_structure_geometry_count =
            geometry_checkpoint;
        rollback_command_references(record, reference_checkpoint);
    }
    if (track_references) {
        track_acceleration_structure_build_references(record, desc)!;
    }
    if (!update) link_pending_acceleration_structure_build(
        record,
        &prepared,
        desc,
    )!;
    emit_acceleration_structure_build(record, &prepared, ops);
}

fn usz? acceleration_structure_strided_bytes(
    uint count,
    usz stride,
    usz element_size,
) {
    if (count == 0 || stride < element_size) return gpu::INVALID_ARGUMENT~;
    usz preceding = (usz)count - 1;
    if (preceding > (usz::max - element_size) / stride) {
        return gpu::INVALID_ARGUMENT~;
    }
    return preceding * stride + element_size;
}

fn usz? acceleration_structure_index_bytes(
    uint primitive_count,
    gpu::AccelerationStructureIndexType index_type,
) {
    usz element_size = index_type == gpu::AccelerationStructureIndexType.U16
        ? 2
        : 4;
    if (primitive_count == 0 || primitive_count > uint::max / 3) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz index_count = (usz)primitive_count * 3;
    if (index_count > usz::max / element_size) {
        return gpu::INVALID_ARGUMENT~;
    }
    return index_count * element_size;
}

fn bool gpu_span_is_zero(gpu::GpuSpan span) {
    return span.owner == 0 && span.index == 0 && span.generation == 0
        && span.offset == 0 && span.size == 0;
}

fn ResolvedSpan? lower_acceleration_structure_input_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    usz required,
    usz alignment,
    bool validate,
) {
    ResolvedSpan resolved = resolve_span_slot(state, span, required)!;
    if ((ulong)resolved.address == 0
        || (ulong)resolved.address % (ulong)alignment != 0
        || !resolved.slot.usage.acceleration_structure_build_input) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (validate) {
        validate_span_recording_access_only(
            state:     state,
            record:    record,
            resolved:  &resolved,
            span:      span,
            operation: "cmd_build_acceleration_structure",
        )!;
    }
    return resolved;
}

fn PreparedAccelerationStructureBuild?
prepare_acceleration_structure_build(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
    bool update,
    bool validate,
) {
    if (record == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (!record.queue.roles.graphics && !record.queue.roles.compute) {
        return gpu::INVALID_ARGUMENT~;
    }
    VkDeviceState* state = record.device_state;
    if (state == null || !state.acceleration_structures_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    AccelerationStructureSlot* destination =
        state.acceleration_structures.get(desc.destination)!;
    if (!update && destination.build_eligibility
        != gpu::internal::AccelerationStructureBuildEligibility.UNBUILT) {
        return gpu::RESOURCE_IN_USE~;
    }
    if (update
        && (!destination.desc.flags.allow_update
            || destination.desc.requirements.update_scratch_size == 0
            || destination.build_eligibility
                != gpu::internal::AccelerationStructureBuildEligibility.BUILT)) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint required_geometry_count = destination.desc.kind
        == gpu::AccelerationStructureKind.TOP_LEVEL
        ? 1
        : (uint)destination.desc.geometries.len;
    if (required_geometry_count
            > record.scratch.acceleration_structure_geometries.len
        || required_geometry_count
            > record.scratch.acceleration_structure_ranges.len) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }

    if (destination.desc.kind
        == gpu::AccelerationStructureKind.TOP_LEVEL) {
        if (desc.geometries.len != 0 || desc.instance_count == 0
            || desc.instance_count > destination.desc.max_instance_count
            || (update
                && desc.instance_count
                    != destination.built_instance_count)) {
            return gpu::INVALID_ARGUMENT~;
        }
        usz instance_bytes;
        if ((usz)desc.instance_count
            > usz::max / gpu::AccelerationStructureInstance::size) {
            return gpu::INVALID_ARGUMENT~;
        }
        instance_bytes = (usz)desc.instance_count
            * gpu::AccelerationStructureInstance::size;
        ResolvedSpan instances = lower_acceleration_structure_input_span(
            state,
            record,
            &desc.instances,
            instance_bytes,
            16,
            validate,
        )!;
        vk::AccelerationStructureGeometryDataKHR geometry_data;
        geometry_data.instances =
            vk::acceleration_structure_geometry_instances_data_khr()
                .set_array_of_pointers(vk::FALSE)
                .set_data({ .device_address =
                    (vk::DeviceAddress)instances.address });
        record.scratch.acceleration_structure_geometries[0] =
            vk::acceleration_structure_geometry_khr()
                .set_geometry_type(vk::GeometryTypeKHR.INSTANCES_KHR)
                .set_geometry(geometry_data);
        record.scratch.acceleration_structure_ranges[0] = {
            .primitive_count = desc.instance_count,
        };
    } else {
        if (desc.geometries.len != destination.desc.geometries.len
            || desc.instance_count != 0
            || !gpu_span_is_zero(desc.instances)) {
            return gpu::INVALID_ARGUMENT~;
        }
        foreach (i, &input : desc.geometries) {
            gpu::internal::NormalizedAccelerationStructureGeometry* schema =
                &destination.desc.geometries[i];
            if (input.kind != schema.kind) return gpu::INVALID_ARGUMENT~;
            vk::AccelerationStructureGeometryDataKHR geometry_data;
            if (schema.kind
                == gpu::AccelerationStructureGeometryKind.TRIANGLES) {
                if (input.triangles.vertex_count == 0
                    || input.triangles.vertex_count
                        > schema.max_vertex_count
                    || input.triangles.primitive_count == 0
                    || input.triangles.primitive_count
                        > schema.max_primitive_count
                    || input.triangles.vertex_stride < 12
                    || input.triangles.vertex_stride % 4 != 0
                    || !gpu_span_is_zero(input.triangles.transform)
                        != schema.has_transform
                    || (update
                        && (input.triangles.vertex_count
                                != schema.built_vertex_count
                            || input.triangles.primitive_count
                                != schema.built_primitive_count
                            || !gpu_span_is_zero(input.triangles.transform)
                                != schema.built_with_transform))) {
                    return gpu::INVALID_ARGUMENT~;
                }
                usz vertex_bytes = acceleration_structure_strided_bytes(
                    input.triangles.vertex_count,
                    input.triangles.vertex_stride,
                    12,
                )!;
                ResolvedSpan vertices =
                    lower_acceleration_structure_input_span(
                        state,
                        record,
                        &input.triangles.vertices,
                        vertex_bytes,
                        4,
                        validate,
                    )!;
                vk::DeviceOrHostAddressConstKHR index_address;
                if (schema.index_type
                    == gpu::AccelerationStructureIndexType.NONE) {
                    if (!gpu_span_is_zero(input.triangles.indices)
                        || input.triangles.vertex_count
                            < input.triangles.primitive_count * 3) {
                        return gpu::INVALID_ARGUMENT~;
                    }
                } else {
                    usz index_bytes = acceleration_structure_index_bytes(
                        input.triangles.primitive_count,
                        schema.index_type,
                    )!;
                    usz index_alignment = schema.index_type
                            == gpu::AccelerationStructureIndexType.U16
                        ? 2
                        : 4;
                    ResolvedSpan indices =
                        lower_acceleration_structure_input_span(
                            state,
                            record,
                            &input.triangles.indices,
                            index_bytes,
                            index_alignment,
                            validate,
                        )!;
                    index_address.device_address =
                        (vk::DeviceAddress)indices.address;
                }
                vk::DeviceOrHostAddressConstKHR transform_address;
                if (!gpu_span_is_zero(input.triangles.transform)) {
                    ResolvedSpan transform =
                        lower_acceleration_structure_input_span(
                            state,
                            record,
                            &input.triangles.transform,
                            12 * float::size,
                            16,
                            validate,
                        )!;
                    transform_address.device_address =
                        (vk::DeviceAddress)transform.address;
                }
                geometry_data.triangles =
                    vk::acceleration_structure_geometry_triangles_data_khr()
                        .set_vertex_format(vk::Format.R32G32B32_SFLOAT)
                        .set_vertex_data({ .device_address =
                            (vk::DeviceAddress)vertices.address })
                        .set_vertex_stride(
                            (vk::DeviceSize)input.triangles.vertex_stride,
                        )
                        .set_max_vertex(input.triangles.vertex_count - 1)
                        .set_index_type(
                            acceleration_structure_index_type_to_vk(
                                schema.index_type,
                            ),
                        )
                        .set_index_data(index_address)
                        .set_transform_data(transform_address);
                record.scratch.acceleration_structure_geometries[i] =
                    vk::acceleration_structure_geometry_khr()
                        .set_geometry_type(
                            vk::GeometryTypeKHR.TRIANGLES_KHR,
                        )
                        .set_geometry(geometry_data)
                        .set_flags(
                            acceleration_structure_geometry_flags_to_vk(
                                schema.flags,
                            ),
                        );
                record.scratch.acceleration_structure_ranges[i] = {
                    .primitive_count = input.triangles.primitive_count,
                };
            } else {
                if (input.aabbs.primitive_count == 0
                    || input.aabbs.primitive_count
                        > schema.max_primitive_count
                    || input.aabbs.stride < 24
                    || input.aabbs.stride % 8 != 0
                    || (update
                        && input.aabbs.primitive_count
                            != schema.built_primitive_count)) {
                    return gpu::INVALID_ARGUMENT~;
                }
                usz aabb_bytes = acceleration_structure_strided_bytes(
                    input.aabbs.primitive_count,
                    input.aabbs.stride,
                    24,
                )!;
                ResolvedSpan aabbs = lower_acceleration_structure_input_span(
                    state,
                    record,
                    &input.aabbs.aabbs,
                    aabb_bytes,
                    8,
                    validate,
                )!;
                geometry_data.aabbs =
                    vk::acceleration_structure_geometry_aabbs_data_khr()
                        .set_data({ .device_address =
                            (vk::DeviceAddress)aabbs.address })
                        .set_stride((vk::DeviceSize)input.aabbs.stride);
                record.scratch.acceleration_structure_geometries[i] =
                    vk::acceleration_structure_geometry_khr()
                        .set_geometry_type(vk::GeometryTypeKHR.AABBS_KHR)
                        .set_geometry(geometry_data)
                        .set_flags(
                            acceleration_structure_geometry_flags_to_vk(
                                schema.flags,
                            ),
                        );
                record.scratch.acceleration_structure_ranges[i] = {
                    .primitive_count = input.aabbs.primitive_count,
                };
            }
        }
    }

    usz required_scratch = update
        ? destination.desc.requirements.update_scratch_size
        : destination.desc.requirements.build_scratch_size;
    ResolvedSpan scratch = resolve_span_slot(
        state,
        &desc.scratch,
        required_scratch,
    )!;
    if (!scratch.slot.usage.storage
        || (ulong)scratch.address == 0
        || !gpu::internal::acceleration_structure_scratch_fits(
            &destination.desc.requirements,
            scratch.size,
            scratch.address,
            update,
        )) return gpu::INVALID_ARGUMENT~;
    if (destination.placement.is_valid()
        && desc.scratch.owner == destination.placement.owner
        && desc.scratch.index == destination.placement.index
        && desc.scratch.generation == destination.placement.generation
        && scratch.offset
            < destination.placement_offset
                + destination.desc.requirements.storage_size
        && destination.placement_offset
            < scratch.offset + required_scratch) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (validate) {
        validate_span_recording_access_only(
            state:     state,
            record:    record,
            resolved:  &scratch,
            span:      &desc.scratch,
            operation: update
                ? "cmd_update_acceleration_structure"
                : "cmd_build_acceleration_structure",
        )!;
    }
    record.scratch.acceleration_structure_geometry_count =
        required_geometry_count;
    return {
        .destination        = destination,
        .destination_handle = desc.destination,
        .scratch            = scratch,
        .scratch_span       = desc.scratch,
        .geometry_count     = required_geometry_count,
        .update             = update,
    };
}

struct AccelerationStructureSlot {
    vk::AccelerationStructureKHR                    native;
    BufferHandle                                    backing_buffer;
    gpu::GpuAllocation                              placement;
    usz                                             placement_offset;
    gpu::internal::NormalizedAccelerationStructureDesc desc;
    Atomic{uint}                                    tracked_command_references;
    uint                                            live_view_count;
    uint                                            built_instance_count;
    gpu::internal::AccelerationStructureBuildEligibility build_eligibility;
    uint                                            pending_next;
    char[DEBUG_NAME_CAP]                            debug_name;
    bool                                            owns_backing_buffer;
}

struct AccelerationStructureCell {
    AccelerationStructureSlot value;
    uint                      generation;
    uint                      next_free;
    bool                      used;
}

struct AccelerationStructureTable {
    AccelerationStructureCell[] slots;
    ulong                       owner;
    uint                        free_head;
    uint                        next_new;
}

fn gpu::AccelerationStructureHandle? AccelerationStructureTable.alloc(
    &self,
    AccelerationStructureSlot value,
) {
    uint index;
    bool found;
    while (self.free_head != ACCELERATION_STRUCTURE_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = ACCELERATION_STRUCTURE_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }
    self.slots[index].value = value;
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = self.slots[index].generation,
    };
}

fn AccelerationStructureSlot*? AccelerationStructureTable.get(
    &self,
    gpu::AccelerationStructureHandle handle,
) {
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint index = handle.index;
    if (index >= self.slots.len || !self.slots[index].used) {
        return gpu::INVALID_HANDLE~;
    }
    if (self.slots[index].generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &self.slots[index].value;
}

fn AccelerationStructureSlot? AccelerationStructureTable.free(
    &self,
    gpu::AccelerationStructureHandle handle,
) {
    AccelerationStructureSlot released = *self.get(handle)!;
    uint index = handle.index;
    self.slots[index].value = {};
    self.slots[index].used = false;
    if (!should_recycle_handle_generation(self.slots[index].generation)) {
        self.slots[index].generation = HANDLE_GENERATION_MAX;
        self.slots[index].next_free = ACCELERATION_STRUCTURE_FREE_NONE;
        return released;
    }
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].next_free = self.free_head;
    self.free_head = index + 1;
    return released;
}

fn void? create_acceleration_structure_state(VkDeviceState* state) {
    if (state == null || !state.acceleration_structures_enabled) {
        return gpu::INVALID_ARGUMENT~;
    }
    state.acceleration_structures.owner =
        gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.ACCELERATION_STRUCTURE,
        );
    state.acceleration_structures.slots = mem::new_array(
        AccelerationStructureCell,
        MAX_ACCELERATION_STRUCTURES,
    );
}

fn void destroy_acceleration_structure_state(VkDeviceState* state) {
    if (state.acceleration_structures.slots.ptr != null) {
        foreach (&cell : state.acceleration_structures.slots[
            :state.acceleration_structures.next_new
        ]) {
            if (cell.used && cell.value.native != null
                && state.acceleration_structure_operations != null
                && state.acceleration_structure_operations
                    .destroy_acceleration_structure != null) {
                state.acceleration_structure_operations.destroy_acceleration_structure(
                    state.device,
                    cell.value.native,
                    null,
                );
                cell.value.native = (vk::AccelerationStructureKHR)null;
            }
            if (cell.value.desc.geometries.ptr != null) {
                alloc::free(
                    state.host_allocator,
                    cell.value.desc.geometries.ptr,
                );
                cell.value.desc.geometries = {};
            }
        }
        mem::free(state.acceleration_structures.slots.ptr);
        state.acceleration_structures.slots = {};
    }
    state.acceleration_structures.owner = 0;
    state.acceleration_structure_operations = null;
}

fn vk::AccelerationStructureTypeKHR acceleration_structure_type_to_vk(
    gpu::AccelerationStructureKind kind,
) {
    return kind == gpu::AccelerationStructureKind.TOP_LEVEL
        ? vk::AccelerationStructureTypeKHR.TOP_LEVEL_KHR
        : vk::AccelerationStructureTypeKHR.BOTTOM_LEVEL_KHR;
}

fn vk::BuildAccelerationStructureFlagsKHR acceleration_structure_flags_to_vk(
    gpu::AccelerationStructureBuildFlags flags,
) {
    uint bits;
    if (flags.allow_update) {
        bits |= (uint)vk::BuildAccelerationStructureFlagBitsKHR
            .ALLOW_UPDATE_BIT_KHR;
    }
    if (flags.prefer_fast_trace) {
        bits |= (uint)vk::BuildAccelerationStructureFlagBitsKHR
            .PREFER_FAST_TRACE_BIT_KHR;
    }
    if (flags.prefer_fast_build) {
        bits |= (uint)vk::BuildAccelerationStructureFlagBitsKHR
            .PREFER_FAST_BUILD_BIT_KHR;
    }
    if (flags.minimize_memory) {
        bits |= (uint)vk::BuildAccelerationStructureFlagBitsKHR
            .LOW_MEMORY_BIT_KHR;
    }
    return (vk::BuildAccelerationStructureFlagsKHR)bits;
}

fn vk::GeometryFlagsKHR acceleration_structure_geometry_flags_to_vk(
    gpu::AccelerationStructureGeometryFlags flags,
) {
    return flags.opaque
        ? (vk::GeometryFlagsKHR)vk::GeometryFlagBitsKHR.OPAQUE_BIT_KHR
        : (vk::GeometryFlagsKHR)0;
}

fn vk::IndexType acceleration_structure_index_type_to_vk(
    gpu::AccelerationStructureIndexType index_type,
) {
    switch (index_type) {
        case gpu::AccelerationStructureIndexType.U16:
            return vk::IndexType.UINT16;
        case gpu::AccelerationStructureIndexType.U32:
            return vk::IndexType.UINT32;
        default:
            return vk::IndexType.NONE_KHR;
    }
}

fn void lower_acceleration_structure_geometry_capacity(
    gpu::internal::NormalizedAccelerationStructureGeometry* source,
    vk::AccelerationStructureGeometryKHR* out_geometry,
) {
    vk::AccelerationStructureGeometryDataKHR data;
    if (source.kind == gpu::AccelerationStructureGeometryKind.TRIANGLES) {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles =
            vk::acceleration_structure_geometry_triangles_data_khr()
                .set_vertex_format(vk::Format.R32G32B32_SFLOAT)
                .set_vertex_stride(3 * float::size)
                .set_max_vertex(source.max_vertex_count - 1)
                .set_index_type(acceleration_structure_index_type_to_vk(
                    source.index_type,
                ));
        if (source.has_transform) {
            triangles = triangles.set_transform_data({
                .host_address = (void*)1,
            });
        }
        data.triangles = triangles;
        *out_geometry = vk::acceleration_structure_geometry_khr()
            .set_geometry_type(vk::GeometryTypeKHR.TRIANGLES_KHR)
            .set_geometry(data)
            .set_flags(acceleration_structure_geometry_flags_to_vk(
                source.flags,
            ));
        return;
    }
    data.aabbs = vk::acceleration_structure_geometry_aabbs_data_khr()
        .set_stride(6 * float::size);
    *out_geometry = vk::acceleration_structure_geometry_khr()
        .set_geometry_type(vk::GeometryTypeKHR.AABBS_KHR)
        .set_geometry(data)
        .set_flags(acceleration_structure_geometry_flags_to_vk(
            source.flags,
        ));
}

fn gpu::AccelerationStructureRequirements?
get_acceleration_structure_requirements_with_ops(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
    AccelerationStructureRequirementsOps* ops,
) {
    if (state == null || desc == null || ops == null
        || ops.get_build_sizes == null) return gpu::INVALID_ARGUMENT~;
    if (!state.acceleration_structures_enabled) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        usz normalized_count = desc.kind
            == gpu::AccelerationStructureKind.BOTTOM_LEVEL
            ? desc.geometries.len
            : 0;
        gpu::internal::NormalizedAccelerationStructureGeometry[] normalized =
            mem::talloc_array(
                gpu::internal::NormalizedAccelerationStructureGeometry,
                (sz)normalized_count,
            );
        gpu::internal::NormalizedAccelerationStructureDesc normalized_desc;
        gpu::internal::normalize_acceleration_structure_desc(
            desc,
            &state.caps.acceleration_structures,
            normalized,
            &normalized_desc,
        )!;

        usz native_geometry_count = normalized_count > 0
            ? normalized_count
            : 1;
        vk::AccelerationStructureGeometryKHR[] geometries =
            mem::talloc_array(
                vk::AccelerationStructureGeometryKHR,
                (sz)native_geometry_count,
            );
        uint[] primitive_counts = mem::talloc_array(
            uint,
            (sz)native_geometry_count,
        );
        if (normalized_desc.kind
            == gpu::AccelerationStructureKind.TOP_LEVEL) {
            vk::AccelerationStructureGeometryDataKHR data;
            data.instances =
                vk::acceleration_structure_geometry_instances_data_khr()
                    .set_array_of_pointers(vk::FALSE);
            geometries[0] = vk::acceleration_structure_geometry_khr()
                .set_geometry_type(vk::GeometryTypeKHR.INSTANCES_KHR)
                .set_geometry(data);
            primitive_counts[0] = normalized_desc.max_instance_count;
        } else {
            foreach (i, &geometry : normalized_desc.geometries) {
                lower_acceleration_structure_geometry_capacity(
                    geometry,
                    &geometries[i],
                );
                primitive_counts[i] = geometry.max_primitive_count;
            }
        }

        vk::AccelerationStructureBuildGeometryInfoKHR build_info =
            vk::acceleration_structure_build_geometry_info_khr()
                .set_type(acceleration_structure_type_to_vk(
                    normalized_desc.kind,
                ))
                .set_flags(acceleration_structure_flags_to_vk(
                    normalized_desc.flags,
                ))
                .set_mode(vk::BuildAccelerationStructureModeKHR.BUILD_KHR)
                .set_geometries(geometries);
        vk::AccelerationStructureBuildSizesInfoKHR size_info =
            vk::acceleration_structure_build_sizes_info_khr();
        ops.get_build_sizes(
            ops.user_data,
            state.device,
            vk::AccelerationStructureBuildTypeKHR.DEVICE_KHR,
            &build_info,
            primitive_counts.ptr,
            &size_info,
        );
        if (size_info.acceleration_structure_size == 0
            || size_info.build_scratch_size == 0
            || (normalized_desc.flags.allow_update
                && size_info.update_scratch_size == 0)
            || size_info.acceleration_structure_size > (ulong)usz::max
            || size_info.build_scratch_size > (ulong)usz::max
            || size_info.update_scratch_size > (ulong)usz::max) {
            return gpu::BACKEND_ERROR~;
        }
        return (gpu::AccelerationStructureRequirements){
            .storage_size = (usz)size_info.acceleration_structure_size,
            .storage_alignment = ACCELERATION_STRUCTURE_STORAGE_ALIGNMENT,
            .build_scratch_size = (usz)size_info.build_scratch_size,
            .update_scratch_size = normalized_desc.flags.allow_update
                ? (usz)size_info.update_scratch_size
                : 0,
            .scratch_alignment = state.caps.acceleration_structures.scratch_alignment,
        };
    };
}

fn void acceleration_structure_get_build_sizes_real(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureBuildTypeKHR build_type,
    vk::AccelerationStructureBuildGeometryInfoKHR* build_info,
    uint* max_primitive_counts,
    vk::AccelerationStructureBuildSizesInfoKHR* size_info,
) {
    VkAccelerationStructureDispatch* dispatch = (VkAccelerationStructureDispatch*)user_data;
    dispatch.get_build_sizes(
        device,
        build_type,
        build_info,
        max_primitive_counts,
        size_info,
    );
}

fn vk::Result acceleration_structure_create_real(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureCreateInfoKHR* create_info,
    vk::AccelerationStructureKHR* out_acceleration_structure,
) {
    VkAccelerationStructureDispatch* dispatch = (VkAccelerationStructureDispatch*)user_data;
    return dispatch.create_acceleration_structure(
        device,
        create_info,
        null,
        out_acceleration_structure,
    );
}

fn void acceleration_structure_destroy_real(
    void* user_data,
    vk::Device device,
    vk::AccelerationStructureKHR acceleration_structure,
) {
    VkAccelerationStructureDispatch* dispatch = (VkAccelerationStructureDispatch*)user_data;
    dispatch.destroy_acceleration_structure(
        device,
        acceleration_structure,
        null,
    );
}

fn AccelerationStructureNativeOps acceleration_structure_native_ops(
    VkDeviceState* state,
) {
    return {
        .user_data = state.acceleration_structure_operations,
        .create    = &acceleration_structure_create_real,
        .destroy   = &acceleration_structure_destroy_real,
    };
}

fn bool placed_acceleration_structure_overlaps(
    VkDeviceState* state,
    gpu::GpuAllocation placement,
    usz offset,
    usz size,
) {
    foreach (&cell : state.acceleration_structures.slots[
        :state.acceleration_structures.next_new
    ]) {
        if (!cell.used || cell.value.placement != placement) continue;
        usz existing_size = cell.value.desc.requirements.storage_size;
        if (offset < cell.value.placement_offset + existing_size
            && cell.value.placement_offset < offset + size) return true;
    }
    return false;
}

fn gpu::AccelerationStructureHandle?
create_acceleration_structure_on_buffer_with_ops(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
    gpu::AccelerationStructureRequirements requirements,
    vk::Buffer buffer,
    usz offset,
    BufferHandle backing_buffer,
    gpu::GpuAllocation placement,
    bool owns_backing_buffer,
    AccelerationStructureNativeOps* ops,
) {
    if (state == null || desc == null || buffer == null || ops == null
        || ops.create == null || ops.destroy == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz geometry_count = desc.kind
        == gpu::AccelerationStructureKind.BOTTOM_LEVEL
        ? desc.geometries.len
        : 0;
    gpu::internal::NormalizedAccelerationStructureGeometry[] schema;
    if (geometry_count > 0) {
        gpu::internal::NormalizedAccelerationStructureGeometry[]?
            allocated_schema = alloc::new_array_try(
            state.host_allocator,
            gpu::internal::NormalizedAccelerationStructureGeometry,
            (sz)geometry_count,
        );
        if (catch allocated_schema) return gpu::OUT_OF_HOST_MEMORY~;
        schema = allocated_schema;
    }
    bool owns_schema = schema.ptr != null;
    vk::AccelerationStructureKHR native;
    bool owns_native;
    defer catch {
        if (owns_native) {
            ops.destroy(ops.user_data, state.device, native);
        }
        if (owns_schema) alloc::free(state.host_allocator, schema.ptr);
    }
    gpu::internal::NormalizedAccelerationStructureDesc normalized;
    gpu::internal::normalize_acceleration_structure_desc(
        desc,
        &state.caps.acceleration_structures,
        schema,
        &normalized,
    )!;
    normalized.requirements = requirements;

    vk::AccelerationStructureCreateInfoKHR create_info =
        vk::acceleration_structure_create_info_khr()
            .set_buffer(buffer)
            .set_offset((vk::DeviceSize)offset)
            .set_size((vk::DeviceSize)requirements.storage_size)
            .set_type(acceleration_structure_type_to_vk(normalized.kind));
    check_device_result(state, ops.create(
        ops.user_data,
        state.device,
        &create_info,
        &native,
    ))!;
    owns_native = true;

    AccelerationStructureSlot slot = {
        .native                = native,
        .backing_buffer        = backing_buffer,
        .placement             = placement,
        .placement_offset      = offset,
        .desc                  = normalized,
        .build_eligibility     = gpu::internal::AccelerationStructureBuildEligibility.UNBUILT,
        .pending_next          = ACCELERATION_STRUCTURE_PENDING_NONE,
        .owns_backing_buffer   = owns_backing_buffer,
    };
    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    gpu::AccelerationStructureHandle handle =
        state.acceleration_structures.alloc(slot)!;
    owns_native = false;
    owns_schema = false;
    if (desc.debug_name != null) {
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.ACCELERATION_STRUCTURE_KHR,
            object_handle: (ulong)native,
            name:          desc.debug_name,
        );
    }
    return handle;
}

fn gpu::AccelerationStructureRequirements?
vk_get_acceleration_structure_requirements(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    if (state == null || state.acceleration_structure_operations == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    AccelerationStructureRequirementsOps ops = {
        .user_data = state.acceleration_structure_operations,
        .get_build_sizes = &acceleration_structure_get_build_sizes_real,
    };
    return get_acceleration_structure_requirements_with_ops(
        state,
        desc,
        &ops,
    );
}

fn gpu::AccelerationStructureHandle? vk_create_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    gpu::AccelerationStructureRequirements requirements =
        vk_get_acceleration_structure_requirements(state, desc)!;
    BufferDesc buffer_desc = {
        .size         = requirements.storage_size,
        .usage        = {
            .addressable,
            .acceleration_structure_storage,
        },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = state.selected_queue_roles,
        .debug_name   = desc.debug_name,
    };
    BufferHandle backing = create_buffer_for_state(state, &buffer_desc)!;
    bool owns_backing = true;
    defer catch {
        if (owns_backing) (void)destroy_buffer_for_state(state, backing);
    }
    BufferSlot* buffer = state.buffers.get(backing)!;
    buffer.internal = true;
    AccelerationStructureNativeOps ops =
        acceleration_structure_native_ops(state);
    state.resource_mutex.lock()!!;
    gpu::AccelerationStructureHandle? created =
        create_acceleration_structure_on_buffer_with_ops(
            state:               state,
            desc:                desc,
            requirements:        requirements,
            buffer:              buffer.buffer,
            offset:              0,
            backing_buffer:      backing,
            placement:           {},
            owns_backing_buffer: true,
            ops:                 &ops,
        );
    state.resource_mutex.unlock();
    gpu::AccelerationStructureHandle result = created!;
    owns_backing = false;
    return result;
}

fn gpu::AccelerationStructureHandle? vk_create_placed_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
    gpu::GpuAllocation allocation,
    usz offset,
) {
    gpu::AccelerationStructureRequirements requirements =
        vk_get_acceleration_structure_requirements(state, desc)!;
    AccelerationStructureNativeOps ops =
        acceleration_structure_native_ops(state);

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    AllocationSlot* allocation_slot = state.allocations.get(allocation)!;
    if (allocation_slot.live_placements == uint::max) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    if (allocation_slot.texture_memory
        || !allocation_slot.backing.usage.acceleration_structure_storage
        || !gpu::internal::acceleration_structure_storage_fits(
            &requirements,
            allocation_slot.size,
            allocation_slot.alignment,
            offset,
        )
        || placed_acceleration_structure_overlaps(
            state,
            allocation,
            offset,
            requirements.storage_size,
        )) {
        return gpu::INVALID_ARGUMENT~;
    }

    allocation_slot.live_placements++;
    bool retained_placement = true;
    defer catch {
        if (retained_placement) allocation_slot.live_placements--;
    }
    gpu::AccelerationStructureHandle handle =
        create_acceleration_structure_on_buffer_with_ops(
            state:               state,
            desc:                desc,
            requirements:        requirements,
            buffer:              allocation_slot.backing.buffer,
            offset:              offset,
            backing_buffer:      {},
            placement:           allocation,
            owns_backing_buffer: false,
            ops:                 &ops,
        )!;
    retained_placement = false;
    return handle;
}

fn gpu::DedicatedAccelerationStructure?
vk_create_dedicated_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    gpu::AccelerationStructureRequirements requirements =
        vk_get_acceleration_structure_requirements(state, desc)!;
    gpu::AllocationDesc allocation_desc = {
        .size         = requirements.storage_size,
        .alignment    = requirements.storage_alignment,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = state.selected_queue_roles,
        .debug_name   = desc.debug_name,
    };
    gpu::GpuAllocation allocation =
        vk_allocate_memory(state, &allocation_desc)!;
    bool owns_allocation = true;
    defer catch {
        if (owns_allocation) (void)vk_free_allocation(state, allocation);
    }
    gpu::AccelerationStructureHandle acceleration_structure =
        vk_create_placed_acceleration_structure(
            state,
            desc,
            allocation,
            0,
        )!;
    owns_allocation = false;
    return {
        .acceleration_structure = acceleration_structure,
        .allocation             = allocation,
    };
}

fn void? vk_destroy_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    if (state == null || !state.acceleration_structures_enabled
        || state.acceleration_structure_operations == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    AccelerationStructureSlot released;
    state.resource_mutex.lock()!!;
    AccelerationStructureSlot*? resolved_slot =
        state.acceleration_structures.get(acceleration_structure);
    if (catch err = resolved_slot) {
        state.resource_mutex.unlock();
        return err~;
    }
    AccelerationStructureSlot* slot = resolved_slot;
    if (slot.live_view_count > 0
        || slot.tracked_command_references.load(AtomicOrdering.RELAXED) > 0
        || slot.build_eligibility
            == gpu::internal::AccelerationStructureBuildEligibility.PENDING) {
        state.resource_mutex.unlock();
        return gpu::RESOURCE_IN_USE~;
    }
    gpu::GpuAllocation placement = slot.placement;
    if (placement.is_valid()) {
        AllocationSlot*? resolved_allocation = state.allocations.get(placement);
        if (catch err = resolved_allocation) {
            state.resource_mutex.unlock();
            return err~;
        }
        AllocationSlot* allocation = resolved_allocation;
        if (allocation.live_placements == 0) {
            state.resource_mutex.unlock();
            return gpu::INVALID_RESOURCE_STATE~;
        }
        allocation.live_placements--;
    }
    released = state.acceleration_structures.free(
        acceleration_structure,
    )!;
    state.resource_mutex.unlock();

    state.acceleration_structure_operations.destroy_acceleration_structure(
        state.device,
        released.native,
        null,
    );
    if (released.desc.geometries.ptr != null) {
        alloc::free(state.host_allocator, released.desc.geometries.ptr);
    }
    if (released.owns_backing_buffer) {
        destroy_buffer_for_state(state, released.backing_buffer)!;
    }
}

fn gpu::GpuAddress? vk_get_acceleration_structure_address(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    if (state == null || !state.acceleration_structures_enabled
        || state.acceleration_structure_operations == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    AccelerationStructureSlot* slot =
        state.acceleration_structures.get(acceleration_structure)!;
    if (slot.desc.kind != gpu::AccelerationStructureKind.BOTTOM_LEVEL) {
        return gpu::INVALID_ARGUMENT~;
    }
    vk::AccelerationStructureDeviceAddressInfoKHR info =
        vk::acceleration_structure_device_address_info_khr()
            .set_acceleration_structure(slot.native);
    vk::DeviceAddress address = state.acceleration_structure_operations
        .get_device_address(state.device, &info);
    if (address == 0) return gpu::BACKEND_ERROR~;
    return (gpu::GpuAddress)address;
}

fn gpu::AccelerationStructureView? vk_create_acceleration_structure_view(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    AccelerationStructureDescriptorOps ops = {
        .write = &write_acceleration_structure_descriptor_real,
    };
    return create_acceleration_structure_descriptor_view_with_ops(
        state,
        acceleration_structure,
        &ops,
    );
}

fn void? vk_destroy_acceleration_structure_view(
    VkDeviceState* state,
    gpu::AccelerationStructureView view,
) {
    return destroy_acceleration_structure_descriptor_view(state, view);
}

fn void? trusted_build_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    AccelerationStructureBuildCommandOps ops = {
        .user_data = record.device_state.acceleration_structure_operations,
        .build = &acceleration_structure_build_command_real,
    };
    return record_acceleration_structure_build_with_ops(
        record,
        desc,
        false,
        false,
        false,
        &ops,
    );
}

fn void? full_build_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    AccelerationStructureBuildCommandOps ops = {
        .user_data = record.device_state.acceleration_structure_operations,
        .build = &acceleration_structure_build_command_real,
    };
    return record_acceleration_structure_build_with_ops(
        record,
        desc,
        false,
        true,
        true,
        &ops,
    );
}

fn void? trusted_update_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    AccelerationStructureBuildCommandOps ops = {
        .user_data = record.device_state.acceleration_structure_operations,
        .build = &acceleration_structure_build_command_real,
    };
    return record_acceleration_structure_build_with_ops(
        record,
        desc,
        true,
        false,
        false,
        &ops,
    );
}

fn void? full_update_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    AccelerationStructureBuildCommandOps ops = {
        .user_data = record.device_state.acceleration_structure_operations,
        .build = &acceleration_structure_build_command_real,
    };
    return record_acceleration_structure_build_with_ops(
        record,
        desc,
        true,
        true,
        true,
        &ops,
    );
}
