module gpu_test;

import gpu;

const char[*] RAY_QUERY_TRIANGLE_SPIRV =
    $embed("shaders/ray_query_triangle.comp.spv");
const char[*] RAY_QUERY_AABB_SPIRV =
    $embed("shaders/ray_query_aabb.comp.spv");

struct RayQueryRoot {
    gpu::AccelerationStructureIndex acceleration_structure_index;
    uint                            _pad0;
    gpu::GpuAddress                 output_gpu;
}
$assert RayQueryRoot::size == 16;

struct RayQueryResult {
    uint  hit;
    uint  instance_custom_index;
    uint  primitive_index;
    float distance;
}
$assert RayQueryResult::size == 16;

fn gpu::Device? create_ray_query_test_device(
    gpu::Runtime* out_runtime,
) {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_ray_query_test",
        .contract_validation = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .acceleration_structure_heap_capacity = 32,
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceDesc device_desc = {
        .enable_ray_queries = true,
    };
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter, &device_desc)!;
        if (!support.supported) continue;
        gpu::Device? candidate = gpu::create_device(&adapter, &device_desc);
        if (catch f = candidate) {
            if (f == gpu::UNSUPPORTED_FEATURE) continue;
            return f~;
        }
        *out_runtime = runtime;
        return candidate;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void lavapipe_ray_query_device_path_is_actionable() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_query_test_device(&runtime)!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.ray_queries.enabled
            && caps.acceleration_structures.heap_capacity == 32
            && caps.acceleration_structures.max_geometry_count > 0
            && caps.acceleration_structures.max_instance_count > 0
            && caps.acceleration_structures.scratch_alignment > 0,
        "the selected opted-in device publishes actionable ray-query caps");
}

fn void lavapipe_acceleration_structure_storage_models_are_explicit() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_query_test_device(&runtime)!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::AccelerationStructureDesc desc = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 1,
        .debug_name = "ray_query_storage_model_tlas",
    };
    gpu::AccelerationStructureRequirements requirements =
        gpu::get_acceleration_structure_requirements(&device, &desc)!!;

    gpu::AccelerationStructureHandle hidden =
        gpu::create_acceleration_structure(&device, &desc)!!;
    gpu::destroy_acceleration_structure(&device, hidden)!!;

    gpu::AllocationDesc allocation_desc = {
        .size = requirements.storage_size,
        .alignment = requirements.storage_alignment,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access = { .compute },
        .debug_name = "ray_query_placed_storage",
    };
    gpu::GpuAllocation allocation =
        gpu::allocate_memory(&device, &allocation_desc)!!;
    gpu::AccelerationStructureHandle placed =
        gpu::create_placed_acceleration_structure(
            &device,
            &desc,
            allocation,
            0,
        )!!;
    assert(@catch(gpu::create_placed_acceleration_structure(
        &device,
        &desc,
        allocation,
        0,
    )) == gpu::INVALID_ARGUMENT,
        "overlapping placed acceleration structures publish nothing");
    assert(@catch(gpu::free_allocation(&device, &allocation))
        == gpu::RESOURCE_IN_USE && allocation.is_valid(),
        "a live placed structure retains caller-owned storage without freeing it");
    gpu::destroy_acceleration_structure(&device, placed)!!;
    gpu::free_allocation(&device, &allocation)!!;

    gpu::DedicatedAccelerationStructure dedicated =
        gpu::create_dedicated_acceleration_structure(&device, &desc)!!;
    assert(@catch(gpu::free_allocation(
        &device,
        &dedicated.allocation,
    )) == gpu::RESOURCE_IN_USE,
        "dedicated storage still has an explicit structure-before-allocation order");
    gpu::destroy_acceleration_structure(
        &device,
        dedicated.acceleration_structure,
    )!!;
    gpu::free_allocation(&device, &dedicated.allocation)!!;
}

fn OwnedSpan? allocate_ray_query_span(
    gpu::Device* device,
    usz size,
    usz alignment,
    gpu::MemoryClass memory_class,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size = size,
        .alignment = alignment,
        .memory_class = memory_class,
        .access = { .compute },
        .debug_name = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    return {
        .allocation = allocation,
        .span = gpu::get_allocation_span(device, allocation)!,
    };
}

fn gpu::CompletionPoint? submit_ray_query_commands(
    gpu::Queue queue,
    gpu::ExecutableCommandList executable,
    gpu::CompletionWait[] waits = {},
) {
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = {
        .command_lists = lists[..],
        .completion_waits = waits,
    };
    return gpu::submit(queue, &submit);
}

fn void lavapipe_triangle_blas_tlas_query_round_trip() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_query_test_device(&runtime)!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;

    OwnedSpan vertices = allocate_ray_query_span(
        device:       &device,
        size:         27 * float::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_vertices",
    )!!;
    defer free_owned_span(&device, &vertices)!!;
    float* vertex_data = (float*)test_span_mapping(
        &device,
        vertices.span,
    ).ptr;
    float[27] triangle = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
        4.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        4.0, 1.0, 0.0,
    };
    vertex_data[0:27] = triangle[..];
    gpu::flush_mapped_span(&device, vertices.span)!!;

    OwnedSpan indices_u16 = allocate_ray_query_span(
        device:       &device,
        size:         3 * ushort::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_indices_u16",
    )!!;
    defer free_owned_span(&device, &indices_u16)!!;
    ushort* u16_data = (ushort*)test_span_mapping(
        &device,
        indices_u16.span,
    ).ptr;
    u16_data[0] = 0;
    u16_data[1] = 1;
    u16_data[2] = 2;
    gpu::flush_mapped_span(&device, indices_u16.span)!!;
    OwnedSpan indices_u32 = allocate_ray_query_span(
        device:       &device,
        size:         3 * uint::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_indices_u32",
    )!!;
    defer free_owned_span(&device, &indices_u32)!!;
    uint* u32_data = (uint*)test_span_mapping(
        &device,
        indices_u32.span,
    ).ptr;
    u32_data[0] = 0;
    u32_data[1] = 1;
    u32_data[2] = 2;
    gpu::flush_mapped_span(&device, indices_u32.span)!!;

    gpu::AccelerationStructureGeometryDesc[3] blas_geometries = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque },
            .triangles = {
                .max_vertex_count = 3,
                .max_primitive_count = 1,
                .index_type = gpu::AccelerationStructureIndexType.NONE,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque },
            .triangles = {
                .max_vertex_count = 3,
                .max_primitive_count = 1,
                .index_type = gpu::AccelerationStructureIndexType.U16,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .flags = { .opaque },
            .triangles = {
                .max_vertex_count = 3,
                .max_primitive_count = 1,
                .index_type = gpu::AccelerationStructureIndexType.U32,
            },
        },
    };
    gpu::AccelerationStructureDesc blas_desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = blas_geometries[..],
        .flags = { .allow_update, .prefer_fast_trace },
        .debug_name = "ray_query_triangle_blas",
    };
    gpu::AccelerationStructureRequirements blas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &blas_desc,
        )!!;
    gpu::AccelerationStructureHandle blas =
        gpu::create_acceleration_structure(&device, &blas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, blas)!!;
    OwnedSpan blas_scratch = allocate_ray_query_span(
        device:       &device,
        size:         blas_requirements.build_scratch_size
            > blas_requirements.update_scratch_size
                ? blas_requirements.build_scratch_size
                : blas_requirements.update_scratch_size,
        alignment:    blas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        debug_name:   "ray_query_blas_scratch",
    )!!;
    defer free_owned_span(&device, &blas_scratch)!!;

    gpu::CommandAllocatorDesc allocator_desc = {
        .max_acceleration_structure_geometries_per_build = 3,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList blas_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&blas_commands)!!;
    gpu::AccelerationStructureGeometryBuildDesc[3] blas_inputs = {
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = vertices.span.checked_subspan(
                    0,
                    9 * float::size,
                )!!,
                .vertex_stride = 3 * float::size,
                .vertex_count = 3,
                .primitive_count = 1,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = vertices.span.checked_subspan(
                    9 * float::size,
                    9 * float::size,
                )!!,
                .vertex_stride = 3 * float::size,
                .vertex_count = 3,
                .indices = indices_u16.span,
                .primitive_count = 1,
            },
        },
        {
            .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
            .triangles = {
                .vertices = vertices.span.checked_subspan(
                    18 * float::size,
                    9 * float::size,
                )!!,
                .vertex_stride = 3 * float::size,
                .vertex_count = 3,
                .indices = indices_u32.span,
                .primitive_count = 1,
            },
        },
    };
    gpu::AccelerationStructureBuildDesc blas_build = {
        .destination = blas,
        .geometries = blas_inputs[..],
        .scratch = blas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&blas_commands, &blas_build)!!;
    gpu::ExecutableCommandList blas_executable =
        gpu::end_commands(&blas_commands)!!;
    gpu::CompletionPoint blas_completion =
        submit_ray_query_commands(queue, blas_executable)!!;

    gpu::AccelerationStructureInstanceDesc instance_desc = {
        .transform_row_0 = { 1.0, 0.0, 0.0, 0.0 },
        .transform_row_1 = { 0.0, 1.0, 0.0, 0.0 },
        .transform_row_2 = { 0.0, 0.0, 1.0, 0.0 },
        .bottom_level = blas,
        .custom_index = 0x1234,
        .mask = 0x0f,
        .flags = { .triangle_cull_disable },
    };
    gpu::AccelerationStructureInstance instance =
        gpu::make_acceleration_structure_instance(
            &device,
            &instance_desc,
        )!!;
    OwnedSpan instances = allocate_ray_query_span(
        device:       &device,
        size:         gpu::AccelerationStructureInstance::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_instances",
    )!!;
    defer free_owned_span(&device, &instances)!!;
    *(gpu::AccelerationStructureInstance*)test_span_mapping(
        &device,
        instances.span,
    ).ptr = instance;
    gpu::flush_mapped_span(&device, instances.span)!!;

    gpu::AccelerationStructureDesc tlas_desc = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 1,
        .flags = { .allow_update, .prefer_fast_trace },
        .debug_name = "ray_query_tlas",
    };
    gpu::AccelerationStructureRequirements tlas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &tlas_desc,
        )!!;
    gpu::AccelerationStructureHandle tlas =
        gpu::create_acceleration_structure(&device, &tlas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, tlas)!!;
    OwnedSpan tlas_scratch = allocate_ray_query_span(
        device:       &device,
        size:         tlas_requirements.build_scratch_size
            > tlas_requirements.update_scratch_size
                ? tlas_requirements.build_scratch_size
                : tlas_requirements.update_scratch_size,
        alignment:    tlas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        debug_name:   "ray_query_tlas_scratch",
    )!!;
    defer free_owned_span(&device, &tlas_scratch)!!;

    gpu::CommandList tlas_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&tlas_commands)!!;
    gpu::AccelerationStructureBuildDesc tlas_build = {
        .destination = tlas,
        .instances = instances.span,
        .instance_count = 1,
        .scratch = tlas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&tlas_commands, &tlas_build)!!;
    gpu::ExecutableCommandList tlas_executable =
        gpu::end_commands(&tlas_commands)!!;
    gpu::CompletionWait[1] blas_wait = {{
        .point = blas_completion,
        .before = { .acceleration_structure_build },
    }};
    gpu::CompletionPoint tlas_completion = submit_ray_query_commands(
        queue,
        tlas_executable,
        blas_wait[..],
    )!!;
    gpu::wait_completion(tlas_completion)!!;

    gpu::AccelerationStructureView tlas_view =
        gpu::create_acceleration_structure_view(&device, tlas)!!;
    defer gpu::destroy_acceleration_structure_view(&device, tlas_view)!!;
    OwnedSpan output = allocate_ray_query_span(
        device:       &device,
        size:         5 * RayQueryResult::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "ray_query_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    OwnedSpan root_owned = allocate_ray_query_span(
        device:       &device,
        size:         RayQueryRoot::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    RayQueryRoot* root = (RayQueryRoot*)test_span_mapping(
        &device,
        root_owned.span,
    ).ptr;
    root.acceleration_structure_index = tlas_view.index;
    root.output_gpu = test_span_address(&device, output.span);
    gpu::flush_mapped_span(&device, root_owned.span)!!;

    gpu::ShaderDesc shader = {
        .spirv = RAY_QUERY_TRIANGLE_SPIRV[..],
        .entry_point = "main",
        .debug_name = "ray_query_triangle",
    };
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::CommandList query_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&query_commands)!!;
    gpu::cmd_bind_pipeline(&query_commands, pipeline)!!;
    gpu::cmd_dispatch(
        commands: &query_commands,
        root:     test_span_address(&device, root_owned.span),
        groups:   { 5, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute },
        .after = { .host },
    };
    gpu::cmd_barrier(&query_commands, &host_read)!!;
    gpu::ExecutableCommandList query_executable =
        gpu::end_commands(&query_commands)!!;
    gpu::CompletionWait[1] tlas_wait = {{
        .point = tlas_completion,
        .before = { .compute },
    }};
    gpu::CompletionPoint query_completion = submit_ray_query_commands(
        queue,
        query_executable,
        tlas_wait[..],
    )!!;
    gpu::wait_completion(query_completion)!!;
    gpu::invalidate_mapped_span(&device, output.span)!!;
    RayQueryResult* results = (RayQueryResult*)test_span_mapping(
        &device,
        output.span,
    ).ptr;
    assert(results[0].hit == 1
            && results[0].instance_custom_index == 0x1234
            && results[0].primitive_index == 0
            && results[0].distance > 0.99 && results[0].distance < 1.01,
        "triangle BLAS/TLAS query returns the exact committed hit");
    assert(results[1].hit == 1 && results[2].hit == 1,
        "U16 and U32 indexed geometries produce committed hits");
    assert(results[3].hit == 0 && results[4].hit == 0,
        "spatial misses and instance-mask misses remain uncommitted");

    vertex_data[0] = 8.0;
    vertex_data[3] = 9.0;
    vertex_data[6] = 8.0;
    gpu::flush_mapped_span(&device, vertices.span)!!;
    gpu::CommandList update_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&update_commands)!!;
    gpu::Barrier host_to_update = {
        .before = { .host },
        .after = { .acceleration_structure_build },
    };
    gpu::cmd_barrier(&update_commands, &host_to_update)!!;
    gpu::cmd_update_acceleration_structure(
        &update_commands,
        &blas_build,
    )!!;
    gpu::Barrier update_to_update = {
        .before = { .acceleration_structure_build },
        .after = { .acceleration_structure_build },
    };
    gpu::cmd_barrier(&update_commands, &update_to_update)!!;
    gpu::cmd_update_acceleration_structure(
        &update_commands,
        &tlas_build,
    )!!;
    gpu::Barrier update_to_query = {
        .before = { .acceleration_structure_build },
        .after = { .compute },
    };
    gpu::cmd_barrier(&update_commands, &update_to_query)!!;
    gpu::cmd_bind_pipeline(&update_commands, pipeline)!!;
    gpu::cmd_dispatch(
        commands: &update_commands,
        root:     test_span_address(&device, root_owned.span),
        groups:   { 5, 1, 1 },
    )!!;
    gpu::cmd_barrier(&update_commands, &host_read)!!;
    gpu::ExecutableCommandList update_executable =
        gpu::end_commands(&update_commands)!!;
    gpu::CompletionPoint update_completion =
        submit_ray_query_commands(queue, update_executable)!!;
    gpu::wait_completion(update_completion)!!;
    gpu::invalidate_mapped_span(&device, output.span)!!;
    assert(results[0].hit == 0
            && results[1].hit == 1 && results[2].hit == 1,
        "in-place BLAS/TLAS updates and explicit barriers change query results");
}

fn void lavapipe_aabb_candidates_require_explicit_confirmation() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_query_test_device(&runtime)!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    OwnedSpan aabbs = allocate_ray_query_span(
        device:       &device,
        size:         6 * float::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_aabb_input",
    )!!;
    defer free_owned_span(&device, &aabbs)!!;
    float* aabb_data = (float*)test_span_mapping(&device, aabbs.span).ptr;
    float[6] bounds = { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    aabb_data[0:6] = bounds[..];
    gpu::flush_mapped_span(&device, aabbs.span)!!;
    OwnedSpan triangle_vertices = allocate_ray_query_span(
        device:       &device,
        size:         9 * float::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_mixed_triangle_input",
    )!!;
    defer free_owned_span(&device, &triangle_vertices)!!;
    float* mixed_vertex_data = (float*)test_span_mapping(
        &device,
        triangle_vertices.span,
    ).ptr;
    float[9] mixed_triangle = {
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
    };
    mixed_vertex_data[0:9] = mixed_triangle[..];
    gpu::flush_mapped_span(&device, triangle_vertices.span)!!;

    gpu::AccelerationStructureGeometryDesc[1] aabb_geometries = {{
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .aabbs.max_primitive_count = 1,
    }};
    gpu::AccelerationStructureDesc aabb_blas_desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = aabb_geometries[..],
        .flags = { .prefer_fast_trace },
        .debug_name = "ray_query_aabb_blas",
    };
    gpu::AccelerationStructureRequirements aabb_blas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &aabb_blas_desc,
        )!!;
    gpu::AccelerationStructureHandle aabb_blas =
        gpu::create_acceleration_structure(&device, &aabb_blas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, aabb_blas)!!;
    OwnedSpan aabb_blas_scratch = allocate_ray_query_span(
        device:       &device,
        size:         aabb_blas_requirements.build_scratch_size,
        alignment:    aabb_blas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        debug_name:   "ray_query_aabb_blas_scratch",
    )!!;
    defer free_owned_span(&device, &aabb_blas_scratch)!!;

    gpu::AccelerationStructureGeometryDesc[1] triangle_geometries = {{
        .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
        .flags = { .opaque },
        .triangles = {
            .max_vertex_count = 3,
            .max_primitive_count = 1,
            .index_type = gpu::AccelerationStructureIndexType.NONE,
        },
    }};
    gpu::AccelerationStructureDesc triangle_blas_desc = {
        .kind = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = triangle_geometries[..],
        .flags = { .prefer_fast_trace },
        .debug_name = "ray_query_mixed_triangle_blas",
    };
    gpu::AccelerationStructureRequirements triangle_blas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &triangle_blas_desc,
        )!!;
    gpu::AccelerationStructureHandle triangle_blas =
        gpu::create_acceleration_structure(&device, &triangle_blas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, triangle_blas)!!;
    OwnedSpan triangle_blas_scratch = allocate_ray_query_span(
        device:       &device,
        size:         triangle_blas_requirements.build_scratch_size,
        alignment:    triangle_blas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        debug_name:   "ray_query_mixed_triangle_blas_scratch",
    )!!;
    defer free_owned_span(&device, &triangle_blas_scratch)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .max_acceleration_structure_geometries_per_build = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList blas_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&blas_commands)!!;
    gpu::AccelerationStructureGeometryBuildDesc[1] aabb_inputs = {{
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .aabbs = {
            .aabbs = aabbs.span,
            .stride = 6 * float::size,
            .primitive_count = 1,
        },
    }};
    gpu::AccelerationStructureBuildDesc aabb_blas_build = {
        .destination = aabb_blas,
        .geometries = aabb_inputs[..],
        .scratch = aabb_blas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(
        &blas_commands,
        &aabb_blas_build,
    )!!;
    gpu::Barrier blas_build_barrier = {
        .before = { .acceleration_structure_build },
        .after = { .acceleration_structure_build },
    };
    gpu::cmd_barrier(&blas_commands, &blas_build_barrier)!!;
    gpu::AccelerationStructureGeometryBuildDesc[1] triangle_inputs = {{
        .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
        .triangles = {
            .vertices = triangle_vertices.span,
            .vertex_stride = 3 * float::size,
            .vertex_count = 3,
            .primitive_count = 1,
        },
    }};
    gpu::AccelerationStructureBuildDesc triangle_blas_build = {
        .destination = triangle_blas,
        .geometries = triangle_inputs[..],
        .scratch = triangle_blas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(
        &blas_commands,
        &triangle_blas_build,
    )!!;
    gpu::ExecutableCommandList blas_executable =
        gpu::end_commands(&blas_commands)!!;
    gpu::CompletionPoint blas_completion =
        submit_ray_query_commands(queue, blas_executable)!!;

    gpu::AccelerationStructureInstanceDesc aabb_instance_desc = {
        .transform_row_0 = { 1.0, 0.0, 0.0, 0.0 },
        .transform_row_1 = { 0.0, 1.0, 0.0, 0.0 },
        .transform_row_2 = { 0.0, 0.0, 1.0, 0.0 },
        .bottom_level = aabb_blas,
        .custom_index = 0xAABB,
        .mask = 0xff,
    };
    gpu::AccelerationStructureInstance aabb_instance =
        gpu::make_acceleration_structure_instance(
            &device,
            &aabb_instance_desc,
        )!!;
    gpu::AccelerationStructureInstanceDesc triangle_instance_desc = {
        .transform_row_0 = { 1.0, 0.0, 0.0, 0.0 },
        .transform_row_1 = { 0.0, 1.0, 0.0, 0.0 },
        .transform_row_2 = { 0.0, 0.0, 1.0, 0.0 },
        .bottom_level = triangle_blas,
        .custom_index = 0xBEEF,
        .mask = 0xff,
    };
    gpu::AccelerationStructureInstance triangle_instance =
        gpu::make_acceleration_structure_instance(
            &device,
            &triangle_instance_desc,
        )!!;
    OwnedSpan instances = allocate_ray_query_span(
        device:       &device,
        size:         2 * gpu::AccelerationStructureInstance::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_aabb_instance",
    )!!;
    defer free_owned_span(&device, &instances)!!;
    gpu::AccelerationStructureInstance* instance_data =
        (gpu::AccelerationStructureInstance*)test_span_mapping(
        &device,
        instances.span,
    ).ptr;
    instance_data[0] = aabb_instance;
    instance_data[1] = triangle_instance;
    gpu::flush_mapped_span(&device, instances.span)!!;

    gpu::AccelerationStructureDesc tlas_desc = {
        .kind = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 2,
        .flags = { .prefer_fast_trace },
        .debug_name = "ray_query_aabb_tlas",
    };
    gpu::AccelerationStructureRequirements tlas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &tlas_desc,
        )!!;
    gpu::AccelerationStructureHandle tlas =
        gpu::create_acceleration_structure(&device, &tlas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, tlas)!!;
    OwnedSpan tlas_scratch = allocate_ray_query_span(
        device:       &device,
        size:         tlas_requirements.build_scratch_size,
        alignment:    tlas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        debug_name:   "ray_query_aabb_tlas_scratch",
    )!!;
    defer free_owned_span(&device, &tlas_scratch)!!;
    gpu::CommandList tlas_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&tlas_commands)!!;
    gpu::AccelerationStructureBuildDesc tlas_build = {
        .destination = tlas,
        .instances = instances.span,
        .instance_count = 2,
        .scratch = tlas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&tlas_commands, &tlas_build)!!;
    gpu::ExecutableCommandList tlas_executable =
        gpu::end_commands(&tlas_commands)!!;
    gpu::CompletionWait[1] blas_wait = {{
        .point = blas_completion,
        .before = { .acceleration_structure_build },
    }};
    gpu::CompletionPoint tlas_completion = submit_ray_query_commands(
        queue,
        tlas_executable,
        blas_wait[..],
    )!!;
    gpu::wait_completion(tlas_completion)!!;

    gpu::AccelerationStructureView tlas_view =
        gpu::create_acceleration_structure_view(&device, tlas)!!;
    defer gpu::destroy_acceleration_structure_view(&device, tlas_view)!!;
    OwnedSpan output = allocate_ray_query_span(
        device:       &device,
        size:         3 * RayQueryResult::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "ray_query_aabb_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    OwnedSpan root_owned = allocate_ray_query_span(
        device:       &device,
        size:         RayQueryRoot::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "ray_query_aabb_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    RayQueryRoot* root = (RayQueryRoot*)test_span_mapping(
        &device,
        root_owned.span,
    ).ptr;
    root.acceleration_structure_index = tlas_view.index;
    root.output_gpu = test_span_address(&device, output.span);
    gpu::flush_mapped_span(&device, root_owned.span)!!;
    gpu::ShaderDesc shader = {
        .spirv = RAY_QUERY_AABB_SPIRV[..],
        .entry_point = "main",
        .debug_name = "ray_query_aabb",
    };
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::CommandList query_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&query_commands)!!;
    gpu::cmd_bind_pipeline(&query_commands, pipeline)!!;
    gpu::cmd_dispatch(
        commands: &query_commands,
        root:     test_span_address(&device, root_owned.span),
        groups:   { 3, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute },
        .after = { .host },
    };
    gpu::cmd_barrier(&query_commands, &host_read)!!;
    gpu::ExecutableCommandList query_executable =
        gpu::end_commands(&query_commands)!!;
    gpu::CompletionWait[1] tlas_wait = {{
        .point = tlas_completion,
        .before = { .compute },
    }};
    gpu::CompletionPoint query_completion = submit_ray_query_commands(
        queue,
        query_executable,
        tlas_wait[..],
    )!!;
    gpu::wait_completion(query_completion)!!;
    gpu::invalidate_mapped_span(&device, output.span)!!;
    RayQueryResult* results = (RayQueryResult*)test_span_mapping(
        &device,
        output.span,
    ).ptr;
    assert(results[0].hit == 1
            && results[0].instance_custom_index == 0xAABB
            && results[0].primitive_index == 0
            && results[0].distance > 1.24 && results[0].distance < 1.26,
        "accepted AABB candidate becomes an exact generated intersection");
    assert(results[1].hit == 0,
        "a broad-phase AABB candidate rejected by procedural math stays uncommitted");
    assert(results[2].hit == 1 && results[2].primitive_index == 0,
        "one TLAS traverses separate triangle and procedural AABB BLAS instances");
}
