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

alias TraceRaysIndirectEmitFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    vk::DeviceAddress indirect_address,
);

struct TraceRaysOps {
    void*                   user_data;
    TraceRaysPushRootFn     push_root;
    TraceRaysEmitFn         trace;
    TraceRaysIndirectEmitFn trace_indirect;
}

struct PreparedTraceRaysState {
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
}

struct PreparedTraceRays {
    PreparedTraceRaysState state;
    gpu::Vec3u             dimensions;
}

struct PreparedTraceRaysIndirect {
    PreparedTraceRaysState state;
    ResolvedSpan           args;
    gpu::GpuSpan           args_span;
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
        && (invocations
                > state.caps.ray_tracing_pipelines
                    .max_ray_dispatch_invocation_count
            || dimensions.x
                > state.caps.ray_tracing_pipelines.max_ray_dispatch_dimensions.x
            || dimensions.y
                > state.caps.ray_tracing_pipelines.max_ray_dispatch_dimensions.y
            || dimensions.z
                > state.caps.ray_tracing_pipelines.max_ray_dispatch_dimensions.z)) {
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
    if ((ulong)resolved.address == 0) {
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
    ZString operation,
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
        operation:      operation,
        rejected_field: "shader_binding_table",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      &region.span,
        operation: operation,
    )!;
    if (!resolved.slot.usage.shader_binding_table
        || (ulong)resolved.address == 0
        || (ulong)resolved.address % caps.shader_group_base_alignment != 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .device_address = (vk::DeviceAddress)(ulong)resolved.address,
        .stride         = (vk::DeviceSize)region.stride,
        .size           = (vk::DeviceSize)region.span.size,
    };
}

fn PreparedTraceRaysState? lower_trace_rays_state(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
) {
    if (record == null || table == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (record.device_state == null
        || !record.device_state.ray_tracing_pipelines_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
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
    };
}

fn bool trace_rays_queue_supported(VkDeviceState* state, uint queue_id) {
    if (state == null
        || state.queue_completions.ptr == null
        || queue_id >= state.queue_completions.len) {
        return false;
    }
    QueueCompletionState* completion = &state.queue_completions[queue_id];
    return completion.bound
        && completion.timeline != null
        && queue_flags_include(
            completion.queue_flags,
            vk::QueueFlagBits.COMPUTE_BIT,
        );
}

fn PreparedTraceRaysState? full_prepare_trace_rays_state(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    ZString operation,
) {
    if (record == null || table == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = record.device_state;
    if (state == null || !state.ray_tracing_pipelines_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if ((!record.queue.roles.graphics && !record.queue.roles.compute)
        || !trace_rays_queue_supported(state, record.queue.id)) {
        return gpu::INVALID_ARGUMENT~;
    }
    BoundPipeline* pipeline = &record.logical_pipeline;
    if (!pipeline.handle.is_valid()
        || pipeline.kind != PipelineKind.RAY_TRACING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    return {
        .pipeline            = pipeline,
        .ray_generation      = validate_and_lower_trace_region(
            state:          state,
            record:         record,
            region:         &table.ray_generation,
            ray_generation: true,
            operation:      operation,
        )!,
        .miss                = validate_and_lower_trace_region(
            state:          state,
            record:         record,
            region:         &table.miss,
            ray_generation: false,
            operation:      operation,
        )!,
        .hit                 = validate_and_lower_trace_region(
            state:          state,
            record:         record,
            region:         &table.hit,
            ray_generation: false,
            operation:      operation,
        )!,
        .callable            = validate_and_lower_trace_region(
            state:          state,
            record:         record,
            region:         &table.callable,
            ray_generation: false,
            operation:      operation,
        )!,
        .ray_generation_span = table.ray_generation.span,
        .miss_span           = table.miss.span,
        .hit_span            = table.hit.span,
        .callable_span       = table.callable.span,
        .root                = root,
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
    if (record.device_state == null
        || !record.device_state.ray_tracing_pipelines_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    validate_trace_dimensions(record.device_state, dimensions, false)!;
    return {
        .state      = lower_trace_rays_state(record, root, table)!,
        .dimensions = dimensions,
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
    if ((!record.queue.roles.graphics && !record.queue.roles.compute)
        || !trace_rays_queue_supported(state, record.queue.id)) {
        return gpu::INVALID_ARGUMENT~;
    }
    BoundPipeline* pipeline = &record.logical_pipeline;
    if (!pipeline.handle.is_valid()
        || pipeline.kind != PipelineKind.RAY_TRACING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    validate_trace_dimensions(record.device_state, dimensions, true)!;
    return {
        .state = full_prepare_trace_rays_state(
            record:    record,
            root:      root,
            table:     table,
            operation: "cmd_trace_rays",
        )!,
        .dimensions = dimensions,
    };
}

fn ResolvedSpan? lower_trace_rays_indirect_args(
    VkDeviceState* state,
    gpu::GpuSpan* args,
) {
    ResolvedSpan resolved = resolve_span_slot(
        state,
        args,
        gpu::TraceRaysIndirectCommand::size,
    )!;
    if ((ulong)resolved.address % 4 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn ResolvedSpan? validate_and_lower_trace_rays_indirect_args(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* args,
) {
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           args,
        required:       gpu::TraceRaysIndirectCommand::size,
        operation:      "cmd_trace_rays_indirect",
        rejected_field: "args",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      args,
        operation: "cmd_trace_rays_indirect",
    )!;
    if (!resolved.slot.usage.indirect
        || (ulong)resolved.address % 4 != 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return resolved;
}

fn PreparedTraceRaysIndirect? lower_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    if (record == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = record.device_state;
    if (state == null
        || !state.caps.ray_tracing_pipelines.indirect_dispatch) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .state     = lower_trace_rays_state(record, root, table)!,
        .args      = lower_trace_rays_indirect_args(state, &args)!,
        .args_span = args,
    };
}

fn PreparedTraceRaysIndirect? full_prepare_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    if (record == null) return gpu::INVALID_ARGUMENT~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = record.device_state;
    if (state == null
        || !state.caps.ray_tracing_pipelines.indirect_dispatch) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .state = full_prepare_trace_rays_state(
            record:    record,
            root:      root,
            table:     table,
            operation: "cmd_trace_rays_indirect",
        )!,
        .args = validate_and_lower_trace_rays_indirect_args(
            state,
            record,
            &args,
        )!,
        .args_span = args,
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

fn void emit_trace_rays_indirect_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    vk::DeviceAddress indirect_address,
) {
    VkDeviceState* state = (VkDeviceState*)user_data;
    state.ray_tracing_pipeline_operations.cmd_trace_rays_indirect(
        command_buffer,
        ray_generation,
        miss,
        hit,
        callable,
        indirect_address,
    );
}

fn TraceRaysOps trace_rays_ops_real(VkDeviceState* state) {
    return {
        .user_data      = state,
        .push_root      = &push_trace_root_real,
        .trace          = &emit_trace_rays_real,
        .trace_indirect = &emit_trace_rays_indirect_real,
    };
}

fn void record_trace_rays_with_ops(
    CommandRecord* record,
    PreparedTraceRays* trace,
    TraceRaysOps* ops,
) {
    ops.push_root(
        user_data:      ops.user_data,
        command_buffer: record.command_buffer,
        layout:         trace.state.pipeline.layout,
        root:           trace.state.root,
    );
    ops.trace(
        user_data:      ops.user_data,
        command_buffer: record.command_buffer,
        ray_generation: &trace.state.ray_generation,
        miss:           &trace.state.miss,
        hit:            &trace.state.hit,
        callable:       &trace.state.callable,
        width:          trace.dimensions.x,
        height:         trace.dimensions.y,
        depth:          trace.dimensions.z,
    );
}

fn void record_trace_rays_indirect_with_ops(
    CommandRecord* record,
    PreparedTraceRaysIndirect* trace,
    TraceRaysOps* ops,
) {
    ops.push_root(
        user_data:      ops.user_data,
        command_buffer: record.command_buffer,
        layout:         trace.state.pipeline.layout,
        root:           trace.state.root,
    );
    ops.trace_indirect(
        user_data:       ops.user_data,
        command_buffer:  record.command_buffer,
        ray_generation:  &trace.state.ray_generation,
        miss:            &trace.state.miss,
        hit:             &trace.state.hit,
        callable:        &trace.state.callable,
        indirect_address: (vk::DeviceAddress)(ulong)trace.args.address,
    );
}

fn void? track_trace_rays_state(
    CommandRecord* record,
    PreparedTraceRaysState* trace,
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
        record:     record,
        root:       root,
        table:      table,
        dimensions: dimensions,
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
        record:     record,
        root:       root,
        table:      table,
        dimensions: dimensions,
    )!;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    track_trace_rays_state(record, &trace.state)!;
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
        record:     record,
        root:       root,
        table:      table,
        dimensions: dimensions,
        ops:        &ops,
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
        record:     record,
        root:       root,
        table:      table,
        dimensions: dimensions,
        ops:        &ops,
    );
}

fn void? trusted_trace_rays_indirect_with_ops(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
    TraceRaysOps* ops,
) {
    if (ops == null || ops.push_root == null
        || ops.trace_indirect == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    PreparedTraceRaysIndirect trace = lower_trace_rays_indirect(
        record: record,
        root:   root,
        table:  table,
        args:   args,
    )!;
    record_trace_rays_indirect_with_ops(record, &trace, ops);
}

fn void? full_trace_rays_indirect_with_ops(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
    TraceRaysOps* ops,
) {
    if (ops == null || ops.push_root == null
        || ops.trace_indirect == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    PreparedTraceRaysIndirect trace = full_prepare_trace_rays_indirect(
        record: record,
        root:   root,
        table:  table,
        args:   args,
    )!;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    track_trace_rays_state(record, &trace.state)!;
    track_span_command_reference(
        record.device_state,
        record,
        &trace.args_span,
    )!;
    record_trace_rays_indirect_with_ops(record, &trace, ops);
}

fn void? trusted_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return trusted_trace_rays_indirect_with_ops(
        record: record,
        root:   root,
        table:  table,
        args:   args,
        ops:    &ops,
    );
}

fn void? full_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return full_trace_rays_indirect_with_ops(
        record: record,
        root:   root,
        table:  table,
        args:   args,
        ops:    &ops,
    );
}
