module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

alias TraceRaysPushRootFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineLayout layout,
    gpu::GpuAddress root,
    char[] inline_root,
    uint inline_offset,
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

alias TraceRaysIndirect2EmitFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::DeviceAddress indirect_address,
);

alias RayTracingStackSizeSetFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint pipeline_stack_size,
);

struct TraceRaysOps {
    void*                    user_data;
    TraceRaysPushRootFn      push_root;
    TraceRaysEmitFn          trace;
    TraceRaysIndirectEmitFn  trace_indirect;
    TraceRaysIndirect2EmitFn trace_indirect2;
}

struct RayTracingStackSizeOps {
    void*                    user_data;
    RayTracingStackSizeSetFn set;
}

struct PreparedTraceRaysState {
    BoundPipeline*                    pipeline;
    vk::StridedDeviceAddressRegionKHR ray_generation;
    vk::StridedDeviceAddressRegionKHR miss;
    vk::StridedDeviceAddressRegionKHR hit;
    vk::StridedDeviceAddressRegionKHR callable;
    gpu::GpuAddress                   root;
}

struct PreparedTraceRays {
    PreparedTraceRaysState state;
    gpu::Vec3u             dimensions;
}

struct PreparedTraceRaysIndirect {
    PreparedTraceRaysState state;
    ResolvedSpan           args;
}

struct PreparedTraceRaysIndirect2 {
    BoundPipeline*  pipeline;
    gpu::GpuAddress root;
    ResolvedSpan    args;
}

fn bool trace_span_is_canonical_empty(gpu::GpuSpan span) {
    return span.owner == 0
        && span.index == 0
        && span.generation == 0
        && span.offset == 0
        && span.size == 0;
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

fn PreparedTraceRaysState? lower_trace_rays_state(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
) {
    if (record == null || table == null) return gpu::INVALID_ARGUMENT~;
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
    if (record.device_state == null
        || !record.device_state.ray_tracing_pipelines_enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .state      = lower_trace_rays_state(record, root, table)!,
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
    return resolved;
}

fn PreparedTraceRaysIndirect? lower_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    if (record == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = record.device_state;
    if (state == null
        || !state.caps.ray_tracing_pipelines.indirect_dispatch) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .state     = lower_trace_rays_state(record, root, table)!,
        .args      = lower_trace_rays_indirect_args(state, &args)!,
    };
}

fn ResolvedSpan? lower_trace_rays_indirect2_args(
    VkDeviceState* state,
    gpu::GpuSpan* args,
) {
    ResolvedSpan resolved = resolve_span_slot(
        state,
        args,
        gpu::TraceRaysIndirectCommand2::size,
    )!;
    return resolved;
}

fn PreparedTraceRaysIndirect2? lower_trace_rays_indirect2(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    if (record == null) return gpu::INVALID_ARGUMENT~;
    VkDeviceState* state = record.device_state;
    if (state == null
        || !state.caps.ray_tracing_pipelines.indirect2_dispatch) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .pipeline  = borrow_bound_pipeline(record)!,
        .root      = root,
        .args      = lower_trace_rays_indirect2_args(state, &args)!,
    };
}

fn void push_trace_root_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineLayout layout,
    gpu::GpuAddress root,
    char[] inline_root,
    uint inline_offset,
) {
    (void)user_data;
    gpu::RootPush push = { .root_gpu = root };
    push_root_block(
        command_buffer: command_buffer,
        inline_root:    inline_root,
        inline_offset:  inline_offset,
        layout:         layout,
        stages:         ray_tracing_shader_stage_flags(),
        header:         &push,
        header_size:    gpu::RootPush::size,
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

fn void emit_trace_rays_indirect2_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::DeviceAddress indirect_address,
) {
    VkDeviceState* state = (VkDeviceState*)user_data;
    state.ray_tracing_pipeline_operations.cmd_trace_rays_indirect2(
        command_buffer,
        indirect_address,
    );
}

fn TraceRaysOps trace_rays_ops_real(VkDeviceState* state) {
    return {
        .user_data       = state,
        .push_root       = &push_trace_root_real,
        .trace           = &emit_trace_rays_real,
        .trace_indirect  = &emit_trace_rays_indirect_real,
        .trace_indirect2 = &emit_trace_rays_indirect2_real,
    };
}

fn void set_ray_tracing_pipeline_stack_size_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    uint pipeline_stack_size,
) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    state.ray_tracing_pipeline_operations.cmd_set_stack_size(
        command_buffer,
        pipeline_stack_size,
    );
}

fn RayTracingStackSizeOps ray_tracing_stack_size_ops_real(
    VkDeviceState* state,
) @private {
    return {
        .user_data = state,
        .set       = &set_ray_tracing_pipeline_stack_size_real,
    };
}

fn void? vk_cmd_set_ray_tracing_pipeline_stack_size_with_ops(
    CommandRecord* record,
    ulong pipeline_stack_size,
    RayTracingStackSizeOps* ops,
) {
    if (record == null || ops == null || ops.set == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (pipeline_stack_size > uint::max) return gpu::INVALID_ARGUMENT~;
    ops.set(
        ops.user_data,
        record.command_buffer,
        (uint)pipeline_stack_size,
    );
}

fn void? vk_cmd_set_ray_tracing_pipeline_stack_size(
    CommandRecord* record,
    ulong pipeline_stack_size,
) {
    if (record.device_state == null
        || !record.device_state.ray_tracing_pipelines_enabled
        || record.device_state.ray_tracing_pipeline_operations.cmd_set_stack_size == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    RayTracingStackSizeOps ops =
        ray_tracing_stack_size_ops_real(record.device_state);
    return vk_cmd_set_ray_tracing_pipeline_stack_size_with_ops(
        record,
        pipeline_stack_size,
        &ops,
    );
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
        inline_root:    record.inline_root,
        inline_offset:  trace.state.pipeline.inline_root_offset,
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
        inline_root:    record.inline_root,
        inline_offset:  trace.state.pipeline.inline_root_offset,
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

fn void record_trace_rays_indirect2_with_ops(
    CommandRecord* record,
    PreparedTraceRaysIndirect2* trace,
    TraceRaysOps* ops,
) {
    ops.push_root(
        user_data:      ops.user_data,
        command_buffer: record.command_buffer,
        layout:         trace.pipeline.layout,
        root:           trace.root,
        inline_root:    record.inline_root,
        inline_offset:  trace.pipeline.inline_root_offset,
    );
    ops.trace_indirect2(
        ops.user_data,
        record.command_buffer,
        (vk::DeviceAddress)(ulong)trace.args.address,
    );
}

fn void? vk_cmd_trace_rays_with_ops(
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

fn void? vk_cmd_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    if (record.device_state == null
        || record.device_state.ray_tracing_pipeline_operations.cmd_trace_rays == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return vk_cmd_trace_rays_with_ops(
        record:     record,
        root:       root,
        table:      table,
        dimensions: dimensions,
        ops:        &ops,
    );
}

fn void? vk_cmd_trace_rays_indirect_with_ops(
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

fn void? vk_cmd_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    if (record.device_state == null
        || record.device_state.ray_tracing_pipeline_operations.cmd_trace_rays_indirect == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return vk_cmd_trace_rays_indirect_with_ops(
        record: record,
        root:   root,
        table:  table,
        args:   args,
        ops:    &ops,
    );
}

fn void? vk_cmd_trace_rays_indirect2_with_ops(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
    TraceRaysOps* ops,
) {
    if (ops == null || ops.push_root == null
        || ops.trace_indirect2 == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    PreparedTraceRaysIndirect2 trace = lower_trace_rays_indirect2(
        record: record,
        root:   root,
        args:   args,
    )!;
    record_trace_rays_indirect2_with_ops(record, &trace, ops);
}

fn void? vk_cmd_trace_rays_indirect2(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    if (record.device_state == null
        || record.device_state.ray_tracing_pipeline_operations.cmd_trace_rays_indirect2 == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    TraceRaysOps ops = trace_rays_ops_real(record.device_state);
    return vk_cmd_trace_rays_indirect2_with_ops(
        record: record,
        root:   root,
        args:   args,
        ops:    &ops,
    );
}
