module gpu_test;

import gpu;

const char[*] TRACE_FUNCTIONAL_RAYGEN_SPIRV =
    $embed("shaders/ray_trace_functional.rgen.spv");
const char[*] TRACE_FUNCTIONAL_PRIMARY_MISS_SPIRV =
    $embed("shaders/ray_trace_functional_primary.rmiss.spv");
const char[*] TRACE_FUNCTIONAL_SECONDARY_MISS_SPIRV =
    $embed("shaders/ray_trace_functional_secondary.rmiss.spv");
const char[*] TRACE_FUNCTIONAL_TRIANGLE_HIT_SPIRV =
    $embed("shaders/ray_trace_functional_triangle.rchit.spv");
const char[*] TRACE_FUNCTIONAL_PROCEDURAL_INTERSECTION_SPIRV =
    $embed("shaders/ray_trace_functional_procedural.rint.spv");
const char[*] TRACE_FUNCTIONAL_PROCEDURAL_HIT_SPIRV =
    $embed("shaders/ray_trace_functional_procedural.rchit.spv");
const char[*] TRACE_FUNCTIONAL_CALLABLE_SPIRV =
    $embed("shaders/ray_trace_functional.rcall.spv");

const uint TRACE_FUNCTIONAL_RAY_COUNT = 4;
const uint TRACE_FUNCTIONAL_GROUP_COUNT = 6;

struct TraceFunctionalRoot {
    gpu::AccelerationStructureIndex acceleration_structure_index;
    uint                            _pad0;
    gpu::GpuAddress                 output_gpu;
}
$assert TraceFunctionalRoot::size == 16;

fn usz trace_align_up(usz value, usz alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

fn OwnedSpan? allocate_trace_span(
    gpu::Device*      device,
    usz               size,
    usz               alignment,
    gpu::MemoryClass  memory_class,
    gpu::QueueRoles   access,
    ZString           debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = alignment,
        .memory_class = memory_class,
        .access       = access,
        .debug_name   = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    return {
        .allocation = allocation,
        .span       = gpu::get_allocation_span(device, allocation)!,
    };
}

fn gpu::Device? create_ray_tracing_test_device(
    gpu::Runtime* out_runtime,
) {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_ray_tracing_test",
        .contract_validation = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .acceleration_structure_heap_capacity = 32,
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceDesc device_desc = {
        .enable_ray_tracing_pipelines = true,
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

fn gpu::PipelineHandle? create_functional_ray_tracing_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc ray_generation_shader = {
        .spirv       = TRACE_FUNCTIONAL_RAYGEN_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_raygen",
    };
    gpu::ShaderDesc primary_miss_shader = {
        .spirv       = TRACE_FUNCTIONAL_PRIMARY_MISS_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_primary_miss",
    };
    gpu::ShaderDesc secondary_miss_shader = {
        .spirv       = TRACE_FUNCTIONAL_SECONDARY_MISS_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_secondary_miss",
    };
    gpu::ShaderDesc triangle_hit_shader = {
        .spirv       = TRACE_FUNCTIONAL_TRIANGLE_HIT_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_triangle_hit",
    };
    gpu::ShaderDesc procedural_intersection_shader = {
        .spirv       = TRACE_FUNCTIONAL_PROCEDURAL_INTERSECTION_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_procedural_intersection",
    };
    gpu::ShaderDesc procedural_hit_shader = {
        .spirv       = TRACE_FUNCTIONAL_PROCEDURAL_HIT_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_procedural_hit",
    };
    gpu::ShaderDesc callable_shader = {
        .spirv       = TRACE_FUNCTIONAL_CALLABLE_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_functional_callable",
    };
    gpu::ShaderDesc[1] ray_generation_shaders = { ray_generation_shader };
    gpu::ShaderDesc[2] miss_shaders = {
        primary_miss_shader,
        secondary_miss_shader,
    };
    gpu::RayTracingHitGroupDesc[2] hit_groups = {
        {
            .type               = gpu::RayTracingHitGroupType.TRIANGLES,
            .closest_hit_shader = &triangle_hit_shader,
        },
        {
            .type                = gpu::RayTracingHitGroupType.PROCEDURAL,
            .closest_hit_shader  = &procedural_hit_shader,
            .intersection_shader = &procedural_intersection_shader,
        },
    };
    gpu::ShaderDesc[1] callable_shaders = { callable_shader };
    gpu::RayTracingPipelineDesc desc = {
        .ray_generation_shaders = ray_generation_shaders[..],
        .miss_shaders           = miss_shaders[..],
        .hit_groups             = hit_groups[..],
        .callable_shaders       = callable_shaders[..],
        .max_recursion_depth    = 1,
        .debug_name             = "trace_functional_pipeline",
    };
    return gpu::create_ray_tracing_pipeline(device, &desc);
}

fn OwnedSpan? create_functional_sbt(
    gpu::Device*                       device,
    gpu::PipelineHandle                pipeline,
    gpu::RayTracingPipelineInfo*       info,
    gpu::RayTracingPipelineCaps*       caps,
    gpu::QueueRoles                    access,
    gpu::RayTracingShaderBindingTable* table,
) {
    usz stride = trace_align_up(
        caps.shader_group_handle_size,
        caps.shader_group_handle_alignment,
    );
    usz ray_generation_offset = 0;
    usz miss_offset = trace_align_up(
        ray_generation_offset + stride,
        caps.shader_group_base_alignment,
    );
    usz hit_offset = trace_align_up(
        miss_offset + 2 * stride,
        caps.shader_group_base_alignment,
    );
    usz callable_offset = trace_align_up(
        hit_offset + 2 * stride,
        caps.shader_group_base_alignment,
    );
    usz total_size = callable_offset + stride;
    OwnedSpan sbt = allocate_trace_span(
        device:       device,
        size:         total_size,
        alignment:    caps.shader_group_base_alignment,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_sbt",
    )!;
    defer catch (void)free_owned_span(device, &sbt);

    char[4096] handle_storage;
    usz handle_size = caps.shader_group_handle_size;
    usz handle_bytes = (usz)info.total_group_count * handle_size;
    if (handle_bytes > handle_storage.len
        || info.total_group_count != TRACE_FUNCTIONAL_GROUP_COUNT) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    char[] handles = handle_storage[0:handle_bytes];
    gpu::get_ray_tracing_shader_group_handles(
        device:      device,
        pipeline:    pipeline,
        first_group: 0,
        group_count: info.total_group_count,
        out:         handles,
    )!;
    char[] mapped = test_span_mapping(device, sbt.span);
    usz[6] offsets = {
        ray_generation_offset,
        miss_offset,
        miss_offset + stride,
        hit_offset,
        hit_offset + stride,
        callable_offset,
    };
    foreach (group, offset : offsets) {
        mapped[offset:handle_size] =
            handles[(usz)group * handle_size:handle_size];
    }
    gpu::flush_mapped_span(device, sbt.span)!;

    table.ray_generation = {
        .span   = sbt.span.checked_subspan(ray_generation_offset, stride)!,
        .stride = stride,
    };
    table.miss = {
        .span   = sbt.span.checked_subspan(miss_offset, 2 * stride)!,
        .stride = stride,
    };
    table.hit = {
        .span   = sbt.span.checked_subspan(hit_offset, 2 * stride)!,
        .stride = stride,
    };
    table.callable = {
        .span   = sbt.span.checked_subspan(callable_offset, stride)!,
        .stride = stride,
    };
    return sbt;
}

fn void direct_ray_tracing_triangle_aabb_and_miss_round_trip() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_tracing_test_device(&runtime)!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.ray_tracing_pipelines.enabled
            && caps.acceleration_structures.enabled,
        "the dedicated target requires actionable direct ray tracing");
    gpu::QueueKind queue_kind = caps.queues.graphics
        ? gpu::QueueKind.GRAPHICS
        : gpu::QueueKind.COMPUTE;
    gpu::QueueRoles access = caps.queues.graphics
        ? (gpu::QueueRoles){ .graphics }
        : (gpu::QueueRoles){ .compute };
    gpu::Queue queue = gpu::get_queue(&device, queue_kind)!!;

    OwnedSpan aabbs = allocate_trace_span(
        device:       &device,
        size:         6 * float::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_aabb",
    )!!;
    defer free_owned_span(&device, &aabbs)!!;
    float[6] bounds = { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    float* aabb_data = (float*)test_span_mapping(&device, aabbs.span).ptr;
    aabb_data[0:6] = bounds[..];
    gpu::flush_mapped_span(&device, aabbs.span)!!;

    OwnedSpan vertices = allocate_trace_span(
        device:       &device,
        size:         9 * float::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_triangle",
    )!!;
    defer free_owned_span(&device, &vertices)!!;
    float[9] triangle = {
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
    };
    float* vertex_data = (float*)test_span_mapping(&device, vertices.span).ptr;
    vertex_data[0:9] = triangle[..];
    gpu::flush_mapped_span(&device, vertices.span)!!;

    gpu::AccelerationStructureGeometryDesc[1] aabb_geometries = {{
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .aabbs.max_primitive_count = 1,
    }};
    gpu::AccelerationStructureDesc aabb_desc = {
        .kind       = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = aabb_geometries[..],
        .flags      = { .prefer_fast_trace },
        .debug_name = "trace_functional_aabb_blas",
    };
    gpu::AccelerationStructureRequirements aabb_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &aabb_desc,
        )!!;
    gpu::AccelerationStructureHandle aabb_blas =
        gpu::create_acceleration_structure(&device, &aabb_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, aabb_blas)!!;
    OwnedSpan aabb_scratch = allocate_trace_span(
        device:       &device,
        size:         aabb_requirements.build_scratch_size,
        alignment:    aabb_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       access,
        debug_name:   "trace_functional_aabb_scratch",
    )!!;
    defer free_owned_span(&device, &aabb_scratch)!!;

    gpu::AccelerationStructureGeometryDesc[1] triangle_geometries = {{
        .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
        .flags = { .opaque },
        .triangles = {
            .max_vertex_count = 3,
            .max_primitive_count = 1,
            .index_type = gpu::AccelerationStructureIndexType.NONE,
        },
    }};
    gpu::AccelerationStructureDesc triangle_desc = {
        .kind       = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        .geometries = triangle_geometries[..],
        .flags      = { .prefer_fast_trace },
        .debug_name = "trace_functional_triangle_blas",
    };
    gpu::AccelerationStructureRequirements triangle_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &triangle_desc,
        )!!;
    gpu::AccelerationStructureHandle triangle_blas =
        gpu::create_acceleration_structure(&device, &triangle_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, triangle_blas)!!;
    OwnedSpan triangle_scratch = allocate_trace_span(
        device:       &device,
        size:         triangle_requirements.build_scratch_size,
        alignment:    triangle_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       access,
        debug_name:   "trace_functional_triangle_scratch",
    )!!;
    defer free_owned_span(&device, &triangle_scratch)!!;

    gpu::AccelerationStructureInstanceDesc aabb_instance_desc = {
        .transform_row_0 = { 1.0, 0.0, 0.0, 0.0 },
        .transform_row_1 = { 0.0, 1.0, 0.0, 0.0 },
        .transform_row_2 = { 0.0, 0.0, 1.0, 0.0 },
        .bottom_level    = aabb_blas,
        .custom_index    = 0xAABB,
        .mask            = 0xff,
    };
    gpu::AccelerationStructureInstanceDesc triangle_instance_desc = {
        .transform_row_0 = { 1.0, 0.0, 0.0, 0.0 },
        .transform_row_1 = { 0.0, 1.0, 0.0, 0.0 },
        .transform_row_2 = { 0.0, 0.0, 1.0, 0.0 },
        .bottom_level    = triangle_blas,
        .custom_index    = 0xBEEF,
        .mask            = 0xff,
    };
    gpu::AccelerationStructureInstance[2] instance_values = {
        gpu::make_acceleration_structure_instance(
            &device,
            &aabb_instance_desc,
        )!!,
        gpu::make_acceleration_structure_instance(
            &device,
            &triangle_instance_desc,
        )!!,
    };
    OwnedSpan instances = allocate_trace_span(
        device:       &device,
        size:         2 * gpu::AccelerationStructureInstance::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_instances",
    )!!;
    defer free_owned_span(&device, &instances)!!;
    gpu::AccelerationStructureInstance* instance_data =
        (gpu::AccelerationStructureInstance*)test_span_mapping(
            &device,
            instances.span,
        ).ptr;
    instance_data[0:2] = instance_values[..];
    gpu::flush_mapped_span(&device, instances.span)!!;

    gpu::AccelerationStructureDesc tlas_desc = {
        .kind               = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 2,
        .flags              = { .prefer_fast_trace },
        .debug_name         = "trace_functional_tlas",
    };
    gpu::AccelerationStructureRequirements tlas_requirements =
        gpu::get_acceleration_structure_requirements(
            &device,
            &tlas_desc,
        )!!;
    gpu::AccelerationStructureHandle tlas =
        gpu::create_acceleration_structure(&device, &tlas_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, tlas)!!;
    OwnedSpan tlas_scratch = allocate_trace_span(
        device:       &device,
        size:         tlas_requirements.build_scratch_size,
        alignment:    tlas_requirements.scratch_alignment,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       access,
        debug_name:   "trace_functional_tlas_scratch",
    )!!;
    defer free_owned_span(&device, &tlas_scratch)!!;
    gpu::AccelerationStructureView tlas_view =
        gpu::create_acceleration_structure_view(&device, tlas)!!;
    defer gpu::destroy_acceleration_structure_view(&device, tlas_view)!!;

    OwnedSpan output = allocate_trace_span(
        device:       &device,
        size:         TRACE_FUNCTIONAL_RAY_COUNT * uint::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       access,
        debug_name:   "trace_functional_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    OwnedSpan root_owned = allocate_trace_span(
        device:       &device,
        size:         TraceFunctionalRoot::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    TraceFunctionalRoot* root = (TraceFunctionalRoot*)test_span_mapping(
        &device,
        root_owned.span,
    ).ptr;
    root.acceleration_structure_index = tlas_view.index;
    root.output_gpu = test_span_address(&device, output.span);
    gpu::flush_mapped_span(&device, root_owned.span)!!;

    gpu::PipelineHandle pipeline =
        create_functional_ray_tracing_pipeline(&device)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::RayTracingPipelineInfo pipeline_info =
        gpu::get_ray_tracing_pipeline_info(&device, pipeline)!!;
    assert(pipeline_info.ray_generation.group_count == 1
            && pipeline_info.miss.group_count == 2
            && pipeline_info.hit.group_count == 2
            && pipeline_info.callable.group_count == 1,
        "the functional SBT packs every declared group category");
    gpu::RayTracingShaderBindingTable table;
    OwnedSpan sbt = create_functional_sbt(
        device:   &device,
        pipeline: pipeline,
        info:     &pipeline_info,
        caps:     &caps.ray_tracing_pipelines,
        access:   access,
        table:    &table,
    )!!;
    defer free_owned_span(&device, &sbt)!!;

    gpu::CommandAllocatorDesc allocator_desc = {
        .max_acceleration_structure_geometries_per_build = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::AccelerationStructureGeometryBuildDesc[1] aabb_inputs = {{
        .kind = gpu::AccelerationStructureGeometryKind.AABBS,
        .aabbs = {
            .aabbs           = aabbs.span,
            .stride          = 6 * float::size,
            .primitive_count = 1,
        },
    }};
    gpu::AccelerationStructureBuildDesc aabb_build = {
        .destination = aabb_blas,
        .geometries  = aabb_inputs[..],
        .scratch     = aabb_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&commands, &aabb_build)!!;
    gpu::Barrier build_to_build = {
        .before = { .acceleration_structure_build },
        .after  = { .acceleration_structure_build },
    };
    gpu::cmd_barrier(&commands, &build_to_build)!!;
    gpu::AccelerationStructureGeometryBuildDesc[1] triangle_inputs = {{
        .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
        .triangles = {
            .vertices       = vertices.span,
            .vertex_stride  = 3 * float::size,
            .vertex_count   = 3,
            .primitive_count = 1,
        },
    }};
    gpu::AccelerationStructureBuildDesc triangle_build = {
        .destination = triangle_blas,
        .geometries  = triangle_inputs[..],
        .scratch     = triangle_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&commands, &triangle_build)!!;
    gpu::cmd_barrier(&commands, &build_to_build)!!;
    gpu::AccelerationStructureBuildDesc tlas_build = {
        .destination    = tlas,
        .instances      = instances.span,
        .instance_count = 2,
        .scratch        = tlas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&commands, &tlas_build)!!;
    gpu::Barrier build_to_trace = {
        .before = { .acceleration_structure_build },
        .after  = { .ray_tracing },
    };
    gpu::cmd_barrier(&commands, &build_to_trace)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_trace_rays(
        commands:   &commands,
        root:       test_span_address(&device, root_owned.span),
        shader_binding_table: &table,
        dimensions: { TRACE_FUNCTIONAL_RAY_COUNT, 1, 1 },
    )!!;
    gpu::Barrier trace_to_host = {
        .before = { .ray_tracing },
        .after  = { .host },
    };
    gpu::cmd_barrier(&commands, &trace_to_host)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit_desc)!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&device, output.span)!!;
    uint* results = (uint*)test_span_mapping(&device, output.span).ptr;
    assert(results[0] == 0x12,
        "the procedural AABB group consumes its selected SBT record");
    assert(results[1] == 0x11,
        "the triangle closest-hit group consumes its selected SBT record");
    assert(results[2] == 0x13,
        "the primary miss group reports its exact category");
    assert(results[3] == 0x14,
        "the secondary miss group and callable stage both execute");
}
