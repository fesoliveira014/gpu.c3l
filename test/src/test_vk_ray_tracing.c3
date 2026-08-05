module gpu_test;

import gpu;
import std::io;

const char[*] TRACE_INDIRECT_BUILD_SPIRV =
    $embed("shaders/ray_trace_indirect.comp.spv");

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

struct TraceIndirectBuildRoot {
    gpu::GpuAddress arguments_gpu;
}

struct TraceValidationCapture {
    uint errors;
}

fn void capture_trace_validation_error(
    gpu::DebugMessage* message,
    void* user_data,
) {
    if (message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.validation) {
        TraceValidationCapture* capture = (TraceValidationCapture*)user_data;
        capture.errors++;
    }
}

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
    TraceValidationCapture* validation,
) {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_ray_tracing_test",
        .contract_validation = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .debug_callback = &capture_trace_validation_error,
        .debug_user_data = validation,
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
        validation.errors = 0;
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
            .kind               = gpu::RayTracingHitGroupKind.TRIANGLES,
            .closest_hit_shader = &triangle_hit_shader,
        },
        {
            .kind                = gpu::RayTracingHitGroupKind.PROCEDURAL,
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
        .dynamic_stack_size     = true,
        .debug_name             = "trace_functional_pipeline",
    };
    return gpu::create_ray_tracing_pipeline(device, &desc);
}

fn ulong? functional_pipeline_stack_size(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::RayTracingPipelineInfo* info,
) {
    ulong ray_generation =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.ray_generation.first_group,
            group_shader: gpu::RayTracingGroupShader.GENERAL,
        )!;
    ulong primary_miss =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.miss.first_group,
            group_shader: gpu::RayTracingGroupShader.GENERAL,
        )!;
    ulong secondary_miss =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.miss.first_group + 1,
            group_shader: gpu::RayTracingGroupShader.GENERAL,
        )!;
    ulong triangle_hit =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.hit.first_group,
            group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
        )!;
    ulong procedural_hit =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.hit.first_group + 1,
            group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
        )!;
    ulong intersection =
        gpu::get_ray_tracing_shader_group_stack_size(
            device:       device,
            pipeline:     pipeline,
            group:        info.hit.first_group + 1,
            group_shader: gpu::RayTracingGroupShader.INTERSECTION,
        )!;
    ulong callable = gpu::get_ray_tracing_shader_group_stack_size(
        device:       device,
        pipeline:     pipeline,
        group:        info.callable.first_group,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
    )!;
    ulong miss = primary_miss > secondary_miss
        ? primary_miss
        : secondary_miss;
    ulong closest_hit = triangle_hit > procedural_hit
        ? triangle_hit
        : procedural_hit;
    ulong trace = miss > closest_hit ? miss : closest_hit;
    if (intersection > trace) trace = intersection;
    ulong stack_size = ray_generation + trace + 2 * callable;
    if (stack_size > uint::max) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return stack_size;
}

fn gpu::PipelineHandle? create_trace_indirect_build_pipeline(
    gpu::Device* device,
) {
    gpu::ShaderDesc shader = {
        .spirv       = TRACE_INDIRECT_BUILD_SPIRV[..],
        .entry_point = "main",
        .debug_name  = "trace_indirect_build",
    };
    gpu::ComputePipelineDesc desc = {
        .shader     = shader,
        .debug_name = "trace_indirect_build_pipeline",
    };
    return gpu::create_compute_pipeline(device, &desc);
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

fn void assert_trace_functional_results(
    gpu::Device* device,
    OwnedSpan*   output,
    String       dispatch,
) {
    gpu::invalidate_mapped_span(device, output.span)!!;
    uint* results = (uint*)test_span_mapping(device, output.span).ptr;
    assert(results[0] == 0x12,
        "the procedural AABB group consumes its selected SBT record under %s",
        dispatch);
    assert(results[1] == 0x11,
        "the triangle closest-hit group consumes its selected SBT record under %s",
        dispatch);
    assert(results[2] == 0x13,
        "the primary miss group reports its exact category under %s",
        dispatch);
    assert(results[3] == 0x14,
        "the secondary miss group and callable stage both execute under %s",
        dispatch);
}

fn void ray_tracing_triangle_aabb_and_miss_round_trip() @test {
    gpu::Runtime runtime;
    TraceValidationCapture validation;
    gpu::Device device = create_ray_tracing_test_device(
        &runtime,
        &validation,
    )!!;
    defer {
        destroy_test_device(&device)!!;
        runtime = {};
    }
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.ray_tracing_pipelines.enabled
            && caps.acceleration_structures.enabled,
        "the dedicated target requires actionable direct ray tracing");
    assert(caps.queues.compute,
        "ray-tracing pipelines require a selected compute queue");
    gpu::QueueKind queue_kind = gpu::QueueKind.COMPUTE;
    gpu::QueueRoles access = { .compute };
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
    gpu::AccelerationStructureHandle triangle_clone =
        gpu::create_acceleration_structure(&device, &triangle_desc)!!;
    defer gpu::destroy_acceleration_structure(&device, triangle_clone)!!;
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
        .bottom_level    = triangle_clone,
        .custom_index    = 0xBEEF,
        .mask            = 0xff,
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

    OwnedSpan indirect_args = allocate_trace_span(
        device:       &device,
        size:         gpu::TraceRaysIndirectCommand::size,
        alignment:    4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       access,
        debug_name:   "trace_functional_indirect_args",
    )!!;
    defer free_owned_span(&device, &indirect_args)!!;
    OwnedSpan indirect_root = allocate_trace_span(
        device:       &device,
        size:         TraceIndirectBuildRoot::size,
        alignment:    8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_indirect_root",
    )!!;
    defer free_owned_span(&device, &indirect_root)!!;
    TraceIndirectBuildRoot* indirect_root_data =
        (TraceIndirectBuildRoot*)test_span_mapping(
            &device,
            indirect_root.span,
        ).ptr;
    indirect_root_data.arguments_gpu = test_span_address(
        &device,
        indirect_args.span,
    );
    gpu::flush_mapped_span(&device, indirect_root.span)!!;

    OwnedSpan indirect_output = allocate_trace_span(
        device:       &device,
        size:         TRACE_FUNCTIONAL_RAY_COUNT * uint::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       access,
        debug_name:   "trace_functional_indirect_output",
    )!!;
    defer free_owned_span(&device, &indirect_output)!!;
    OwnedSpan indirect_trace_root = allocate_trace_span(
        device:       &device,
        size:         TraceFunctionalRoot::size,
        alignment:    16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "trace_functional_indirect_trace_root",
    )!!;
    defer free_owned_span(&device, &indirect_trace_root)!!;
    TraceFunctionalRoot* indirect_trace_root_data =
        (TraceFunctionalRoot*)test_span_mapping(
            &device,
            indirect_trace_root.span,
        ).ptr;
    indirect_trace_root_data.acceleration_structure_index = tlas_view.index;
    indirect_trace_root_data.output_gpu = test_span_address(
        &device,
        indirect_output.span,
    );
    gpu::flush_mapped_span(&device, indirect_trace_root.span)!!;

    gpu::PipelineHandle pipeline =
        create_functional_ray_tracing_pipeline(&device)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::PipelineHandle indirect_build_pipeline =
        create_trace_indirect_build_pipeline(&device)!!;
    defer gpu::destroy_pipeline(&device, indirect_build_pipeline)!!;
    gpu::RayTracingPipelineInfo pipeline_info =
        gpu::get_ray_tracing_pipeline_info(&device, pipeline)!!;
    assert(pipeline_info.ray_generation.group_count == 1
            && pipeline_info.miss.group_count == 2
            && pipeline_info.hit.group_count == 2
            && pipeline_info.callable.group_count == 1,
        "the functional SBT packs every declared group category");
    assert(pipeline_info.dynamic_stack_size
        && pipeline_info.max_recursion_depth == 1);
    ulong pipeline_stack_size = functional_pipeline_stack_size(
        &device,
        pipeline,
        &pipeline_info,
    )!!;
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
    gpu::TextureDesc texture_desc = {
        .width      = 1,
        .height     = 1,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .access     = access,
        .debug_name = "trace_functional_texture_state",
    };
    gpu::TextureHandle texture = gpu::create_texture(
        &device,
        &texture_desc,
    )!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::CommandList build_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&build_commands)!!;
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
    gpu::cmd_build_acceleration_structure(&build_commands, &aabb_build)!!;
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
    gpu::cmd_build_acceleration_structure(
        &build_commands,
        &triangle_build,
    )!!;
    gpu::ExecutableCommandList build_executable =
        gpu::end_commands(&build_commands)!!;
    gpu::ExecutableCommandList[1] build_lists = { build_executable };
    gpu::SubmitDesc build_submit = { .command_lists = build_lists[..] };
    gpu::CompletionPoint build_completion =
        gpu::submit(queue, &build_submit)!!;
    gpu::wait_completion(build_completion)!!;

    gpu::CommandList clone_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&clone_commands)!!;
    gpu::cmd_clone_acceleration_structure(
        &clone_commands,
        triangle_blas,
        triangle_clone,
    )!!;
    gpu::ExecutableCommandList clone_executable =
        gpu::end_commands(&clone_commands)!!;
    gpu::ExecutableCommandList[1] clone_lists = { clone_executable };
    gpu::SubmitDesc clone_submit = { .command_lists = clone_lists[..] };
    gpu::CompletionPoint clone_completion =
        gpu::submit(queue, &clone_submit)!!;
    gpu::wait_completion(clone_completion)!!;

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
    gpu::AccelerationStructureInstance* instance_data =
        (gpu::AccelerationStructureInstance*)test_span_mapping(
            &device,
            instances.span,
        ).ptr;
    instance_data[0:2] = instance_values[..];
    gpu::flush_mapped_span(&device, instances.span)!!;

    gpu::CommandList trace_commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&trace_commands)!!;
    gpu::TextureBarrier texture_to_storage = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .ray_tracing }, { .read, .write }),
    )!!;
    gpu::cmd_texture_barrier(&trace_commands, &texture_to_storage)!!;
    gpu::AccelerationStructureBuildDesc tlas_build = {
        .destination    = tlas,
        .instances      = instances.span,
        .instance_count = 2,
        .scratch        = tlas_scratch.span,
    };
    gpu::cmd_build_acceleration_structure(&trace_commands, &tlas_build)!!;
    gpu::Barrier build_to_trace = {
        .before = { .acceleration_structure_build },
        .after  = { .ray_tracing },
    };
    gpu::cmd_barrier(&trace_commands, &build_to_trace)!!;
    if (caps.ray_tracing_pipelines.indirect_dispatch) {
        gpu::cmd_bind_pipeline(
            &trace_commands,
            indirect_build_pipeline,
        )!!;
        gpu::cmd_dispatch(
            commands: &trace_commands,
            root:     test_span_address(&device, indirect_root.span),
            groups:   { 1, 1, 1 },
        )!!;
        gpu::Barrier args_ready = {
            .before = { .compute },
            .after  = { .indirect },
        };
        gpu::cmd_barrier(&trace_commands, &args_ready)!!;
    } else {
        io::printfn("basic indirect ray tracing unavailable; direct proof only");
    }
    gpu::cmd_bind_pipeline(&trace_commands, pipeline)!!;
    gpu::cmd_set_ray_tracing_pipeline_stack_size(
        &trace_commands,
        pipeline_stack_size,
    )!!;
    gpu::cmd_trace_rays(
        commands:             &trace_commands,
        root:                 test_span_address(&device, root_owned.span),
        shader_binding_table: &table,
        dimensions:           { TRACE_FUNCTIONAL_RAY_COUNT, 1, 1 },
    )!!;
    if (caps.ray_tracing_pipelines.indirect_dispatch) {
        gpu::Barrier direct_to_indirect = {
            .before = { .ray_tracing },
            .after  = { .ray_tracing },
        };
        gpu::cmd_barrier(&trace_commands, &direct_to_indirect)!!;
        gpu::cmd_trace_rays_indirect(
            commands:             &trace_commands,
            root:                 test_span_address(
                &device,
                indirect_trace_root.span,
            ),
            shader_binding_table: &table,
            args:                 indirect_args.span,
        )!!;
    }
    gpu::TextureBarrier texture_to_sampled = gpu::texture_transition(
        texture: texture,
        before:  texture_to_storage.after,
        after:   gpu::sampled_at({ .ray_tracing }),
    )!!;
    gpu::cmd_texture_barrier(&trace_commands, &texture_to_sampled)!!;
    gpu::Barrier trace_to_host = {
        .before = { .ray_tracing },
        .after  = { .host },
    };
    gpu::cmd_barrier(&trace_commands, &trace_to_host)!!;
    gpu::ExecutableCommandList trace_executable =
        gpu::end_commands(&trace_commands)!!;
    gpu::ExecutableCommandList[1] trace_lists = { trace_executable };
    gpu::SubmitDesc trace_submit = { .command_lists = trace_lists[..] };
    gpu::CompletionPoint trace_completion =
        gpu::submit(queue, &trace_submit)!!;
    gpu::wait_completion(trace_completion)!!;

    assert_trace_functional_results(&device, &output, "a direct dispatch");
    if (caps.ray_tracing_pipelines.indirect_dispatch) {
        assert_trace_functional_results(
            &device,
            &indirect_output,
            "an indirect dispatch",
        );
    }
    assert(validation.errors == 0,
        "the ray-tracing dispatch emits no Vulkan validation errors");
}
