module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

alias TraceRaysPushRootFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineLayout layout,
    gpu::GpuAddress root,
);

alias TraceRaysEmitFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    uint width,
    uint height,
    uint depth,
);

struct TraceRaysOps {
    void*                user_data;
    TraceRaysPushRootFn  push_root;
    TraceRaysEmitFn      trace;
}

struct PreparedTraceRays {
    BoundPipeline*                    pipeline;
    vk::StridedDeviceAddressRegionKHR ray_generation;
    vk::StridedDeviceAddressRegionKHR miss;
    vk::StridedDeviceAddressRegionKHR hit;
    vk::StridedDeviceAddressRegionKHR callable;
    gpu::GpuSpan                      ray_generation_span;
    gpu::GpuSpan                      miss_span;
    gpu::GpuSpan                      hit_span;
    gpu::GpuSpan                      callable_span;
    gpu::GpuAddress                   root;
    gpu::Vec3u                        dimensions;
}

fn bool trace_span_is_canonical_empty(gpu::GpuSpan span) {
    return span.owner == 0
        && span.index == 0
        && span.generation == 0
        && span.offset == 0
        && span.size == 0;
}

fn void? validate_trace_dimensions(
    VkDeviceState* state,
    gpu::Vec3u dimensions,
    bool validate_limit,
) {
    if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    ulong xy = (ulong)dimensions.x * dimensions.y;
    if (dimensions.z > 0 && xy > ulong::max / dimensions.z) {
        return gpu::INVALID_ARGUMENT~;
    }
    ulong invocations = xy * dimensions.z;
    if (validate_limit
        && invocations
            > state.caps.ray_tracing_pipelines
                .max_ray_dispatch_invocation_count) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn vk::StridedDeviceAddressRegionKHR? lower_trace_region(
    VkDeviceState* state,
    gpu::RayTracingShaderBindingTableRegion* region,
) {
    if (trace_span_is_canonical_empty(region.span)) {
        if (region.stride != 0) return gpu::INVALID_ARGUMENT~;
        return {};
    }
    if (region.span.size == 0) return gpu::INVALID_ARGUMENT~;
    ResolvedSpan resolved = resolve_span(state, &region.span)!;
    if ((ulong)resolved.address == 0
        || region.span.size > ulong::max
        || region.stride > ulong::max) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .device_address = (vk::DeviceAddress)(ulong)resolved.address,
        .stride         = (vk::DeviceSize)region.stride,
        .size           = (vk::DeviceSize)region.span.size,
    };
}

fn vk::StridedDeviceAddressRegionKHR? validate_and_lower_trace_region(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::RayTracingShaderBindingTableRegion* region,
    bool ray_generation,
) {
    bool empty = trace_span_is_canonical_empty(region.span);
    if (empty) {
        if (ray_generation || region.stride != 0) {
            return gpu::INVALID_ARGUMENT~;
        }
        return {};
    }
    if (region.span.size == 0 || region.stride == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::RayTracingPipelineCaps caps =
        state.caps.ray_tracing_pipelines;
    if (caps.shader_group_handle_size == 0
        || caps.shader_group_handle_alignment == 0
        || caps.shader_group_base_alignment == 0
        || caps.max_shader_group_stride == 0
        || region.stride < caps.shader_group_handle_size
        || region.stride % caps.shader_group_handle_alignment != 0
        || region.stride > caps.max_shader_group_stride
        || region.span.size % region.stride != 0
        || (ray_generation && region.span.size != region.stride)) {
        return gpu::INVALID_ARGUMENT~;
    }
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           &region.span,
        required:       region.span.size,
        operation:      "cmd_trace_rays",
        rejected_field: "shader_binding_table",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      &region.span,
        operation: "cmd_trace_rays",
    )!;
    if (!resolved.slot.usage.shader_binding_table
        || (ulong)resolved.address == 0
        || (ulong)resolved.address % caps.shader_group_base_alignment != 0
        || region.span.size > ulong::max
        || region.stride > ulong::max) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .device_address = (vk::DeviceAddress)(ulong)resolved.address,
        .stride         = (vk::DeviceSize)region.stride,
        .size           = (vk::DeviceSize)region.span.size,
    };
}

fn PreparedTraceRays? lower_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    if (record == null || table == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_trace_dimensions(record.device_state, dimensions, false)!;
    return {
        .pipeline            = borrow_bound_pipeline(record)!,
        .ray_generation      = lower_trace_region(
            record.device_state,
            &table.ray_generation,
        )!,
        .miss                = lower_trace_region(
            record.device_state,
            &table.miss,
        )!,
        .hit                 = lower_trace_region(
            record.device_state,
            &table.hit,
        )!,
        .callable            = lower_trace_region(
            record.device_state,
            &table.callable,
        )!,
        .ray_generation_span = table.ray_generation.span,
        .miss_span           = table.miss.span,
        .hit_span            = table.hit.span,
        .callable_span       = table.callable.span,
        .root                = root,
        .dimensions          = dimensions,
    };
}

fn PreparedTraceRays? full_prepare_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    if (record == null || table == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = record.device_state;
    if (state == null || !state.ray_tracing_pipelines_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (!record.queue.roles.graphics && !record.queue.roles.compute) {
        return gpu::INVALID_ARGUMENT~;
    }
    BoundPipeline* pipeline = &record.logical_pipeline;
    if (!pipeline.handle.is_valid()
        || pipeline.kind != PipelineKind.RAY_TRACING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_trace_dimensions(state, dimensions, true)!;
    return {
        .pipeline            = pipeline,
        .ray_generation      = validate_and_lower_trace_region(
            state,
            record,
            &table.ray_generation,
            true,
        )!,
        .miss                = validate_and_lower_trace_region(
            state,
            record,
            &table.miss,
            false,
        )!,
        .hit                 = validate_and_lower_trace_region(
            state,
            record,
            &table.hit,
            false,
        )!,
        .callable            = validate_and_lower_trace_region(
            state,
            record,
            &table.callable,
            false,
        )!,
        .ray_generation_span = table.ray_generation.span,
        .miss_span           = table.miss.span,
        .hit_span            = table.hit.span,
        .callable_span       = table.callable.span,
        .root                = root,
        .dimensions          = dimensions,
    };
}

fn void push_trace_root_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineLayout layout,
    gpu::GpuAddress root,
) {
    (void)user_data;
    gpu::RootPush push = { .root_gpu = root };
    vk::cmd_push_constants(
        command_buffer: command_buffer,
        layout:         layout,
        stage_flags:    ray_tracing_shader_stage_flags(),
        offset:         0,
        size:           gpu::RootPush::size,
        p_values:       &push,
    );
}

fn void emit_trace_rays_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    uint width,
    uint height,
    uint depth,
) {
    VkDeviceState* state = (VkDeviceState*)user_data;
    state.ray_tracing_pipeline_operations.cmd_trace_rays(
        command_buffer,
        ray_generation,
        miss,
        hit,
        callable,
        width,
        height,
        depth,
    );
}

fn TraceRaysOps trace_rays_ops_real(VkDeviceState* state) {
    return {
        .user_data = state,
        .push_root = &push_trace_root_real,
        .trace     = &emit_trace_rays_real,
    };
}

fn void record_trace_rays_with_ops(
    CommandRecord* record,
    PreparedTraceRays* trace,
    TraceRaysOps* ops,
) {
    ops.push_root(
        ops.user_data,
        record.command_buffer,
        trace.pipeline.layout,
        trace.root,
    );
    ops.trace(
        ops.user_data,
        record.command_buffer,
        &trace.ray_generation,
        &trace.miss,
        &trace.hit,
        &trace.callable,
        trace.dimensions.x,
        trace.dimensions.y,
        trace.dimensions.z,
    );
}

fn void? track_trace_rays(
    CommandRecord* record,
    PreparedTraceRays* trace,
) {
    track_bound_pipeline(record, trace.pipeline)!;
    if (!trace_span_is_canonical_empty(trace.ray_generation_span)) {
        track_span_command_reference(
            record.device_state,
            record,
            &trace.ray_generation_span,
        )!;
    }
    if (!trace_span_is_canonical_empty(trace.miss_span)) {
        track_span_command_reference(
            record.device_state,
            record,
            &trace.miss_span,
        )!;
    }
    if (!trace_span_is_canonical_empty(trace.hit_span)) {
        track_span_command_reference(
            record.device_state,
            record,
            &trace.hit_span,
        )!;
    }
    if (!trace_span_is_canonical_empty(trace.callable_span)) {
        track_span_command_reference(
            record.device_state,
            record,
            &trace.callable_span,
        )!;
    }
}

fn void? trusted_trace_rays_with_ops(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
    TraceRaysOps* ops,
) {
    if (ops == null || ops.push_root == null || ops.trace == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    PreparedTraceRays trace = lower_trace_rays(
        record,
        root,
        table,
        dimensions,
    )!;
    record_trace_rays_with_ops(record, &trace, ops);
}

fn void? full_trace_rays_with_ops(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
    TraceRaysOps* ops,
) {
    if (ops == null || ops.push_root == null || ops.trace == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    PreparedTraceRays trace = full_prepare_trace_rays(
        record,
        root,
        table,
        dimensions,
    )!;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    track_trace_rays(record, &trace)!;
    record_trace_rays_with_ops(record, &trace, ops);
}

fn void? trusted_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return trusted_trace_rays_with_ops(
        record,
        root,
        table,
        dimensions,
        &ops,
    );
}

fn void? full_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return full_trace_rays_with_ops(
        record,
        root,
        table,
        dimensions,
        &ops,
    );
}
