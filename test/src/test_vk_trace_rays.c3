module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic::types;

struct TraceRaysCapture {
    gpu::GpuAddress expected_root;
    vk::DeviceAddress expected_args;
    uint            pushes;
    uint            traces;
    uint            indirect_traces;
    uint            indirect2_traces;
    bool            matched;
}

struct RayTracingBindCapture {
    uint calls;
    bool matched;
}

struct RayTracingStackSizeCapture {
    uint expected_size;
    uint calls;
    bool matched;
}

fn void capture_ray_tracing_bind(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    vk::PipelineBindPoint bind_point,
    vk::Pipeline pipeline,
) {
    (void)command_buffer;
    RayTracingBindCapture* capture =
        (RayTracingBindCapture*)raw_capture;
    capture.calls++;
    capture.matched = capture.matched
        && bind_point == vk::PipelineBindPoint.RAY_TRACING_KHR
        && pipeline == (vk::Pipeline)(uptr)0x55;
}

fn void capture_ray_tracing_stack_size(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    uint pipeline_stack_size,
) {
    (void)command_buffer;
    RayTracingStackSizeCapture* capture =
        (RayTracingStackSizeCapture*)raw_capture;
    capture.calls++;
    capture.matched = capture.matched
        && pipeline_stack_size == capture.expected_size;
}

fn void capture_trace_root(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    vk::PipelineLayout layout,
    gpu::GpuAddress root,
) {
    (void)command_buffer;
    TraceRaysCapture* capture = (TraceRaysCapture*)raw_capture;
    capture.pushes++;
    capture.matched = capture.matched
        && layout == (vk::PipelineLayout)(uptr)0x44
        && root == capture.expected_root;
}

fn void capture_trace_rays(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    uint width,
    uint height,
    uint depth,
) {
    (void)command_buffer;
    TraceRaysCapture* capture = (TraceRaysCapture*)raw_capture;
    capture.traces++;
    capture.matched = capture.matched
        && ray_generation.device_address == 0x1000
        && ray_generation.stride == 32
        && ray_generation.size == 32
        && miss.device_address == 0x1000
        && miss.stride == 32
        && miss.size == 64
        && hit.device_address == 0
        && hit.stride == 0
        && hit.size == 0
        && callable.device_address == 0
        && callable.stride == 0
        && callable.size == 0
        && width == 2
        && height == 3
        && depth == 4;
}

fn void capture_trace_rays_indirect(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    vk::StridedDeviceAddressRegionKHR* ray_generation,
    vk::StridedDeviceAddressRegionKHR* miss,
    vk::StridedDeviceAddressRegionKHR* hit,
    vk::StridedDeviceAddressRegionKHR* callable,
    vk::DeviceAddress indirect_address,
) {
    (void)command_buffer;
    TraceRaysCapture* capture = (TraceRaysCapture*)raw_capture;
    capture.indirect_traces++;
    capture.matched = capture.matched
        && ray_generation.device_address == 0x1000
        && ray_generation.stride == 32
        && ray_generation.size == 32
        && miss.device_address == 0x1000
        && miss.stride == 32
        && miss.size == 64
        && hit.device_address == 0
        && hit.stride == 0
        && hit.size == 0
        && callable.device_address == 0
        && callable.stride == 0
        && callable.size == 0
        && indirect_address == capture.expected_args;
}

fn void capture_trace_rays_indirect2(
    void* raw_capture,
    vk::CommandBuffer command_buffer,
    vk::DeviceAddress indirect_address,
) {
    (void)command_buffer;
    TraceRaysCapture* capture = (TraceRaysCapture*)raw_capture;
    capture.indirect2_traces++;
    capture.matched = capture.matched
        && indirect_address == capture.expected_args;
}

struct TraceRaysFixture {
    SpanResolverFixture    spans;
    vk::CommandRecord      record;
    gpu::RayTracingShaderBindingTable table;
    vk::PipelineCell[1]    pipeline_cells;
    vk::PipelineCacheEntry[1] pipeline_entries;
    vk::QueueCompletionState[1] completions;
    vk::CommandResourceReference[4] references;
    vk::CommandScratch     scratch;
}

fn gpu::GpuSpan fixture_trace_sbt_span(
    TraceRaysFixture* fixture,
    usz offset,
    usz size,
) {
    gpu::GpuSpan span = fixture_allocation_span(&fixture.spans, offset, size);
    span.index = 1;
    return span;
}

fn void init_trace_rays_fixture(TraceRaysFixture* fixture) {
    init_span_resolver_fixture(&fixture.spans);
    fixture.spans.state.allocations.slots = fixture.spans.allocation_cells[..];
    fixture.spans.state.allocations.next_new = 2;
    fixture.spans.allocation_cells[1] = fixture.spans.allocation_cells[0];
    fixture.spans.allocation_cells[1].value.backing.gpu_base = (gpu::GpuAddress)0x1000;
    fixture.spans.allocation_cells[1].value.backing.size = 128;
    fixture.spans.allocation_cells[0].value.backing.size = 128;
    fixture.spans.allocation_cells[0].value.backing.cpu_base = null;
    fixture.spans.allocation_cells[1].value.backing.usage.shader_binding_table = true;
    fixture.spans.state.ray_tracing_pipelines_enabled = true;
    fixture.spans.state.caps.ray_tracing_pipelines = {
        .enabled                           = true,
        .indirect_dispatch                 = true,
        .indirect2_dispatch                = true,
        .max_ray_dispatch_invocation_count = 128,
        .max_ray_dispatch_dimensions       = { 7, 6, 5 },
        .shader_group_handle_size          = 32,
        .shader_group_handle_alignment     = 32,
        .shader_group_base_alignment       = 64,
        .max_shader_group_stride           = 64,
    };
    fixture.spans.state.pipelines.owner =
        gpu::internal::resource_owner_tag(
            SPAN_RESOLVER_DEVICE_OWNER,
            gpu::internal::ResourceOwnerKind.PIPELINE,
        );
    fixture.spans.state.pipelines.slots = fixture.pipeline_cells[..];
    fixture.spans.state.pipelines.next_new = 1;
    fixture.spans.state.pipeline_cache.entries = fixture.pipeline_entries[..];
    fixture.spans.state.pipeline_cache.next_new = 1;
    gpu::PipelineHandle pipeline = {
        .owner      = fixture.spans.state.pipelines.owner,
        .index      = 0,
        .generation = 3,
    };
    fixture.pipeline_cells[0] = {
        .value = {
            .pipeline = (vk::Pipeline)(uptr)0x55,
            .layout   = (vk::PipelineLayout)(uptr)0x44,
            .entry    = 0,
        },
        .generation = pipeline.generation,
        .used       = true,
    };
    fixture.pipeline_entries[0] = {
        .key  = { .kind = (int)vk::PipelineKind.RAY_TRACING },
        .used = true,
    };
    fixture.scratch.references = fixture.references[..];
    fixture.completions[0] = {
        .timeline    = (vk::Semaphore)(uptr)0x66,
        .queue_flags = vk::QueueFlagBits.COMPUTE_BIT,
        .bound       = true,
    };
    fixture.spans.state.queue_completions = fixture.completions[..];
    fixture.record = {
        .device_state = &fixture.spans.state,
        .state        = gpu::internal::CommandState.RECORDING,
        .command_buffer = (vk::CommandBuffer)(uptr)0x33,
        .queue        = { .roles = { .compute } },
        .scratch      = &fixture.scratch,
        .logical_pipeline = {
            .handle       = pipeline,
            .pipeline     = (vk::Pipeline)(uptr)0x55,
            .layout       = (vk::PipelineLayout)(uptr)0x44,
            .cache_entry  = 0,
            .kind         = vk::PipelineKind.RAY_TRACING,
        },
    };
    fixture.table = {
        .ray_generation = {
            .span   = fixture_trace_sbt_span(fixture, 0, 32),
            .stride = 32,
        },
        .miss = {
            .span   = fixture_trace_sbt_span(fixture, 0, 64),
            .stride = 32,
        },
    };
    fixture.spans.allocation_cells[0].value.backing.usage.indirect = true;
}

fn gpu::GpuSpan fixture_trace_args(
    TraceRaysFixture* fixture,
    usz offset = 0,
    usz size = gpu::TraceRaysIndirectCommand::size,
) {
    return {
        .owner      = fixture.spans.state.allocations.owner,
        .index      = 0,
        .generation = 7,
        .offset     = offset,
        .size       = size,
    };
}

fn gpu::GpuSpan fixture_trace_indirect2_args(
    TraceRaysFixture* fixture,
    usz offset = 0,
    usz size = gpu::TraceRaysIndirectCommand2::size,
) {
    return fixture_trace_args(fixture, offset, size);
}

fn void trusted_trace_lowers_exact_regions_root_and_dimensions() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuAddress root = (gpu::GpuAddress)0x1234;
    TraceRaysCapture capture = {
        .expected_root = root,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data = &capture,
        .push_root = &capture_trace_root,
        .trace     = &capture_trace_rays,
    };

    vk::trusted_trace_rays_with_ops(
        &fixture.record,
        root,
        &fixture.table,
        { 2, 3, 4 },
        &ops,
    )!!;
    assert(capture.matched && capture.pushes == 1 && capture.traces == 1);
}

fn void trusted_trace_rejects_a_device_without_ray_tracing_enabled() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.ray_tracing_pipelines_enabled = false;
    fixture.spans.state.caps.ray_tracing_pipelines = {};
    TraceRaysCapture capture = { .matched = true };
    vk::TraceRaysOps ops = {
        .user_data = &capture,
        .push_root = &capture_trace_root,
        .trace     = &capture_trace_rays,
    };

    assert(@catch(vk::trusted_trace_rays_with_ops(
        record:     &fixture.record,
        root:       (gpu::GpuAddress)1,
        table:      &fixture.table,
        dimensions: { 1, 1, 1 },
        ops:        &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(capture.pushes == 0 && capture.traces == 0);
}

fn void trusted_indirect_trace_lowers_exact_regions_root_and_address() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_args(&fixture);
    gpu::GpuAddress root = (gpu::GpuAddress)0x5678;
    TraceRaysCapture capture = {
        .expected_root = root,
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data      = &capture,
        .push_root      = &capture_trace_root,
        .trace_indirect = &capture_trace_rays_indirect,
    };

    vk::trusted_trace_rays_indirect_with_ops(
        record: &fixture.record,
        root:   root,
        table:  &fixture.table,
        args:   args,
        ops:    &ops,
    )!!;
    assert(capture.matched
        && capture.pushes == 1
        && capture.indirect_traces == 1);
    assert(fixture.scratch.reference_count == 0);
}

fn void indirect_trace_capability_rejects_before_argument_resolution() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.caps.ray_tracing_pipelines.indirect_dispatch = false;
    TraceRaysCapture capture = { .matched = true };
    vk::TraceRaysOps ops = {
        .user_data      = &capture,
        .push_root      = &capture_trace_root,
        .trace_indirect = &capture_trace_rays_indirect,
    };

    assert(@catch(vk::trusted_trace_rays_indirect_with_ops(
        record: &fixture.record,
        root:   {},
        table:  null,
        args:   {},
        ops:    &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(capture.pushes == 0 && capture.indirect_traces == 0);
}

fn void trusted_indirect2_trace_pushes_root_and_exact_address() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_indirect2_args(&fixture);
    gpu::GpuAddress root = (gpu::GpuAddress)0x6789;
    TraceRaysCapture capture = {
        .expected_root = root,
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data       = &capture,
        .push_root       = &capture_trace_root,
        .trace_indirect2 = &capture_trace_rays_indirect2,
    };

    vk::trusted_trace_rays_indirect2_with_ops(
        record: &fixture.record,
        root:   root,
        args:   args,
        ops:    &ops,
    )!!;
    assert(capture.matched
        && capture.pushes == 1
        && capture.indirect2_traces == 1);
    assert(fixture.scratch.reference_count == 0);
}

fn void indirect2_capability_rejects_before_argument_resolution() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.caps.ray_tracing_pipelines.indirect2_dispatch = false;
    TraceRaysCapture capture = { .matched = true };
    vk::TraceRaysOps ops = {
        .user_data       = &capture,
        .push_root       = &capture_trace_root,
        .trace_indirect2 = &capture_trace_rays_indirect2,
    };

    assert(@catch(vk::trusted_trace_rays_indirect2_with_ops(
        record: &fixture.record,
        root:   {},
        args:   {},
        ops:    &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(capture.pushes == 0 && capture.indirect2_traces == 0);
}

fn void indirect2_packet_requires_complete_range() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_indirect2_args(
        &fixture,
        0,
        gpu::TraceRaysIndirectCommand2::size - 1,
    );

    assert(@catch(vk::lower_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_ARGUMENT);
    args = fixture_trace_indirect2_args(&fixture);
    vk::PreparedTraceRaysIndirect2 prepared =
        vk::lower_trace_rays_indirect2(&fixture.record, {}, args)!!;
    assert(prepared.args.address == (gpu::GpuAddress)0x2000);
    args = fixture_trace_indirect2_args(&fixture, 2);
    assert(@catch(vk::lower_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_ARGUMENT);
}

fn void full_indirect2_rejects_semantic_argument_misuse() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_indirect2_args(&fixture);

    fixture.spans.allocation_cells[0].value.backing.usage.indirect = false;
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_ARGUMENT);
    fixture.spans.allocation_cells[0].value.backing.usage.indirect = true;
    args.owner = gpu::internal::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER + 1,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_HANDLE);
    args = fixture_trace_indirect2_args(&fixture);
    fixture.spans.allocation_cells[0].value.backing.access = { .transfer };
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_ARGUMENT);
}

fn void full_indirect2_requires_valid_dynamic_stack_state() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_indirect2_args(&fixture);

    fixture.record.logical_pipeline.handle = {};
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.logical_pipeline.handle = {
        .owner      = fixture.spans.state.pipelines.owner,
        .generation = 3,
    };
    fixture.record.logical_pipeline.kind = vk::PipelineKind.COMPUTE;
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.logical_pipeline.kind = vk::PipelineKind.RAY_TRACING;
    fixture.record.state = gpu::internal::CommandState.RECORDING_RENDER_PASS;
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.state = gpu::internal::CommandState.RECORDING;
    fixture.record.queue.roles = { .graphics };
    fixture.completions[0].queue_flags = vk::QueueFlagBits.GRAPHICS_BIT;
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.queue.roles = { .compute };
    fixture.completions[0].queue_flags = vk::QueueFlagBits.COMPUTE_BIT;
    fixture.record.logical_pipeline.dynamic_ray_tracing_stack_size = true;
    assert(@catch(vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )) == gpu::INVALID_RESOURCE_STATE);
    fixture.record.ray_tracing_stack_size_set = true;
    (void)vk::full_prepare_trace_rays_indirect2(
        &fixture.record,
        {},
        args,
    )!!;
}

fn void full_indirect2_retains_only_pipeline_and_packet_transactionally() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.resource_mutex.init()!!;
    defer (void)fixture.spans.state.resource_mutex.destroy();
    gpu::GpuSpan args = fixture_trace_indirect2_args(&fixture);
    TraceRaysCapture capture = {
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data       = &capture,
        .push_root       = &capture_trace_root,
        .trace_indirect2 = &capture_trace_rays_indirect2,
    };

    vk::full_trace_rays_indirect2_with_ops(
        record: &fixture.record,
        root:   {},
        args:   args,
        ops:    &ops,
    )!!;
    assert(capture.matched && capture.indirect2_traces == 1);
    assert(fixture.scratch.reference_count == 2,
        "only the pipeline and packet allocation are retained");
    assert(fixture.pipeline_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
    assert(fixture.spans.allocation_cells[0].value
        .tracked_command_references.load(AtomicOrdering.RELAXED) == 1);
    assert(fixture.spans.allocation_cells[1].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "GPU-authored SBT addresses remain caller-owned");
    vk::rollback_command_references(&fixture.record, 0);

    fixture.scratch.references = fixture.references[:1];
    capture.pushes = 0;
    capture.indirect2_traces = 0;
    assert(@catch(vk::full_trace_rays_indirect2_with_ops(
        record: &fixture.record,
        root:   {},
        args:   args,
        ops:    &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(fixture.scratch.reference_count == 0);
    assert(fixture.pipeline_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
    assert(fixture.spans.allocation_cells[0].value
        .tracked_command_references.load(AtomicOrdering.RELAXED) == 0);
    assert(capture.pushes == 0 && capture.indirect2_traces == 0);
}

fn void ray_tracing_bind_uses_distinct_native_bind_point_and_cache() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.record.descriptor_heap_binding.ray_tracing_set_bound = true;
    RayTracingBindCapture capture = { .matched = true };
    vk::PipelineBindOps ops = {
        .user_data = &capture,
        .bind      = &capture_ray_tracing_bind,
    };
    vk::BoundPipeline bound = fixture.record.logical_pipeline;
    vk::record_bound_pipeline_with_ops(&fixture.record, &bound, &ops)!!;
    vk::record_bound_pipeline_with_ops(&fixture.record, &bound, &ops)!!;
    assert(capture.matched && capture.calls == 1);
    assert(fixture.record.native_ray_tracing_pipeline.pipeline
        == bound.pipeline);
}

fn void dynamic_stack_size_sets_only_after_full_validation() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.record.logical_pipeline.dynamic_ray_tracing_stack_size = true;
    RayTracingStackSizeCapture capture = {
        .expected_size = 37,
        .matched       = true,
    };
    vk::RayTracingStackSizeOps ops = {
        .user_data = &capture,
        .set       = &capture_ray_tracing_stack_size,
    };

    vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )!!;
    assert(capture.matched && capture.calls == 1);
    assert(fixture.record.ray_tracing_stack_size_set);

    fixture.record.ray_tracing_stack_size_set = false;
    capture.expected_size = 0;
    vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        0,
        &ops,
    )!!;
    assert(capture.matched && capture.calls == 2,
        "a pipeline whose groups report no stack accepts a zero size");
    assert(fixture.record.ray_tracing_stack_size_set);

    fixture.record.ray_tracing_stack_size_set = false;
    SyncDiagnosticCapture diagnostic = {
        .expected_operation      = "cmd_set_ray_tracing_pipeline_stack_size",
        .expected_rejected_field = "pipeline_stack_size",
        .expected_invariant      = "stack size must be representable by uint",
    };
    fixture.spans.state.debug_callback = &capture_sync_diagnostic;
    fixture.spans.state.debug_user_data = &diagnostic;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        (ulong)uint::max + 1,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(diagnostic.calls == 1
        && diagnostic.public_contract
        && diagnostic.public_fault == gpu::INVALID_ARGUMENT
        && diagnostic.operation_matches
        && diagnostic.rejected_field_matches
        && diagnostic.invariant_matches,
        "FULL stack-size misuse emits one structured diagnostic");
    fixture.spans.state.debug_callback = null;
    fixture.spans.state.debug_user_data = null;
    fixture.record.logical_pipeline.dynamic_ray_tracing_stack_size = false;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.logical_pipeline.handle = {};
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.logical_pipeline.handle = {
        .owner      = fixture.spans.state.pipelines.owner,
        .generation = 3,
    };
    fixture.record.logical_pipeline.kind = vk::PipelineKind.COMPUTE;
    fixture.record.logical_pipeline.dynamic_ray_tracing_stack_size = true;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.logical_pipeline.kind = vk::PipelineKind.GRAPHICS;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.logical_pipeline.kind = vk::PipelineKind.RAY_TRACING;
    fixture.record.state =
        gpu::internal::CommandState.RECORDING_RENDER_PASS;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.state = gpu::internal::CommandState.RECORDING;
    fixture.record.queue.roles = { .transfer };
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.queue.roles = { .graphics };
    fixture.completions[0].queue_flags = vk::QueueFlagBits.GRAPHICS_BIT;
    assert(@catch(vk::full_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        37,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(capture.calls == 2);
    assert(!fixture.record.ray_tracing_stack_size_set);
}

fn void trusted_dynamic_stack_size_lowers_without_reference_work() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    RayTracingStackSizeCapture capture = {
        .expected_size = 19,
        .matched       = true,
    };
    vk::RayTracingStackSizeOps ops = {
        .user_data = &capture,
        .set       = &capture_ray_tracing_stack_size,
    };

    vk::trusted_set_ray_tracing_pipeline_stack_size_with_ops(
        &fixture.record,
        19,
        &ops,
    )!!;
    assert(capture.matched && capture.calls == 1);
    assert(fixture.record.ray_tracing_stack_size_set);
    assert(fixture.scratch.reference_count == 0);
}

fn void static_ray_bind_invalidates_dynamic_stack_state() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.record.descriptor_heap_binding.ray_tracing_set_bound = true;
    RayTracingBindCapture capture = { .matched = true };
    vk::PipelineBindOps ops = {
        .user_data = &capture,
        .bind      = &capture_ray_tracing_bind,
    };
    vk::BoundPipeline dynamic = fixture.record.logical_pipeline;
    dynamic.dynamic_ray_tracing_stack_size = true;
    fixture.record.ray_tracing_stack_size_set = true;
    vk::record_bound_pipeline_with_ops(&fixture.record, &dynamic, &ops)!!;
    assert(fixture.record.ray_tracing_stack_size_set);

    vk::BoundPipeline static_pipeline = dynamic;
    static_pipeline.dynamic_ray_tracing_stack_size = false;
    vk::record_bound_pipeline_with_ops(
        &fixture.record,
        &static_pipeline,
        &ops,
    )!!;
    assert(capture.calls == 1,
        "logical static rebinding can suppress the native bind");
    assert(!fixture.record.ray_tracing_stack_size_set);
    fixture.record.ray_tracing_stack_size_set = true;
    vk::record_bound_pipeline_with_ops(&fixture.record, &dynamic, &ops)!!;
    assert(fixture.record.ray_tracing_stack_size_set);
}

fn void full_dynamic_trace_requires_current_stack_state() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.record.logical_pipeline.dynamic_ray_tracing_stack_size = true;
    gpu::GpuSpan args = fixture_trace_args(&fixture);
    SyncDiagnosticCapture diagnostic = {
        .expected_operation      = "cmd_trace_rays",
        .expected_rejected_field = "pipeline_stack_size",
        .expected_invariant      = "dynamic ray-tracing stack size must be set before tracing",
    };
    fixture.spans.state.debug_callback = &capture_sync_diagnostic;
    fixture.spans.state.debug_user_data = &diagnostic;

    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        {},
        &fixture.table,
        { 1, 1, 1 },
    )) == gpu::INVALID_RESOURCE_STATE);
    assert(diagnostic.calls == 1
        && diagnostic.public_contract
        && diagnostic.public_fault == gpu::INVALID_RESOURCE_STATE
        && diagnostic.operation_matches
        && diagnostic.rejected_field_matches
        && diagnostic.invariant_matches,
        "FULL direct trace reports missing dynamic stack state");
    diagnostic.expected_operation = "cmd_trace_rays_indirect";
    diagnostic.operation_matches = false;
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_RESOURCE_STATE);
    assert(diagnostic.calls == 2 && diagnostic.operation_matches,
        "FULL indirect trace reports missing dynamic stack state");
    fixture.spans.state.debug_callback = null;
    fixture.spans.state.debug_user_data = null;
    fixture.record.ray_tracing_stack_size_set = true;
    (void)vk::full_prepare_trace_rays(
        &fixture.record,
        {},
        &fixture.table,
        { 1, 1, 1 },
    )!!;
    (void)vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )!!;
}

fn void full_trace_tracks_pipeline_and_unique_sbt_backing() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.resource_mutex.init()!!;
    defer (void)fixture.spans.state.resource_mutex.destroy();
    TraceRaysCapture capture = {
        .expected_root = (gpu::GpuAddress)7,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data = &capture,
        .push_root = &capture_trace_root,
        .trace     = &capture_trace_rays,
    };
    vk::full_trace_rays_with_ops(
        &fixture.record,
        capture.expected_root,
        &fixture.table,
        { 2, 3, 4 },
        &ops,
    )!!;

    assert(capture.matched && capture.traces == 1);
    assert(fixture.scratch.reference_count == 2,
        "pipeline plus one deduplicated SBT buffer are retained");
    assert(fixture.pipeline_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
    assert(fixture.spans.allocation_cells[1].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
    vk::rollback_command_references(&fixture.record, 0);
    assert(fixture.scratch.reference_count == 0);
}

fn void full_indirect_trace_tracks_pipeline_sbt_and_arguments() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.resource_mutex.init()!!;
    defer (void)fixture.spans.state.resource_mutex.destroy();
    gpu::GpuSpan args = fixture_trace_args(&fixture);
    TraceRaysCapture capture = {
        .expected_root = (gpu::GpuAddress)9,
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data      = &capture,
        .push_root      = &capture_trace_root,
        .trace_indirect = &capture_trace_rays_indirect,
    };

    vk::full_trace_rays_indirect_with_ops(
        record: &fixture.record,
        root:   capture.expected_root,
        table:  &fixture.table,
        args:   args,
        ops:    &ops,
    )!!;
    assert(capture.matched && capture.indirect_traces == 1);
    assert(fixture.scratch.reference_count == 3,
        "pipeline, SBT buffer, and indirect allocation are retained");
    assert(fixture.pipeline_cells[0].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 1);
    assert(fixture.spans.allocation_cells[1].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 1);
    assert(fixture.spans.allocation_cells[0].value
        .tracked_command_references.load(AtomicOrdering.RELAXED) == 1);
    vk::rollback_command_references(&fixture.record, 0);
}

fn void indirect_trace_validates_range_alignment_usage_and_queue() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_args(&fixture);

    args.size = gpu::TraceRaysIndirectCommand::size - 1;
    assert(@catch(vk::lower_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);

    args = fixture_trace_args(&fixture, 2);
    assert(@catch(vk::lower_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);

    args = fixture_trace_args(&fixture);
    fixture.spans.allocation_cells[0].value.backing.usage.indirect = false;
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);
    fixture.spans.allocation_cells[0].value.backing.usage.indirect = true;
    fixture.spans.allocation_cells[0].value.backing.access = { .transfer };
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);
}

fn void indirect_trace_reuses_pipeline_queue_and_render_scope_validation() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_args(&fixture);

    fixture.record.logical_pipeline.kind = vk::PipelineKind.COMPUTE;
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::COMMAND_RECORDING_ERROR);
    fixture.record.logical_pipeline.kind = vk::PipelineKind.RAY_TRACING;
    fixture.record.queue.roles = { .transfer };
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);
    fixture.record.queue.roles = { .graphics };
    fixture.spans.allocation_cells[1].value.backing.access = { .graphics };
    fixture.spans.allocation_cells[0].value.backing.access = { .graphics };
    fixture.completions[0].queue_flags = vk::QueueFlagBits.GRAPHICS_BIT;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        {},
        &fixture.table,
        { 1, 1, 1 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::INVALID_ARGUMENT);
    fixture.completions[0].queue_flags = vk::QueueFlagBits.COMPUTE_BIT;
    fixture.record.queue.roles = { .compute };
    fixture.record.state = gpu::internal::CommandState.RECORDING_RENDER_PASS;
    assert(@catch(vk::full_prepare_trace_rays_indirect(
        &fixture.record,
        {},
        &fixture.table,
        args,
    )) == gpu::COMMAND_RECORDING_ERROR);
}

fn void trusted_indirect_trace_repeats_without_reference_work() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    gpu::GpuSpan args = fixture_trace_args(&fixture);
    TraceRaysCapture capture = {
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data      = &capture,
        .push_root      = &capture_trace_root,
        .trace_indirect = &capture_trace_rays_indirect,
    };

    for (uint i = 0; i < 64; i++) {
        vk::trusted_trace_rays_indirect_with_ops(
            record: &fixture.record,
            root:   {},
            table:  &fixture.table,
            args:   args,
            ops:    &ops,
        )!!;
    }
    assert(capture.matched
        && capture.pushes == 64
        && capture.indirect_traces == 64);
    assert(fixture.scratch.reference_count == 0,
        "trusted repeated recording performs no reference allocation");
}

fn void indirect_trace_reference_exhaustion_rolls_back_every_retain() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.spans.state.resource_mutex.init()!!;
    defer (void)fixture.spans.state.resource_mutex.destroy();
    fixture.scratch.references = fixture.references[:2];
    gpu::GpuSpan args = fixture_trace_args(&fixture);
    TraceRaysCapture capture = {
        .expected_args = 0x2000,
        .matched       = true,
    };
    vk::TraceRaysOps ops = {
        .user_data      = &capture,
        .push_root      = &capture_trace_root,
        .trace_indirect = &capture_trace_rays_indirect,
    };

    assert(@catch(vk::full_trace_rays_indirect_with_ops(
        record: &fixture.record,
        root:   {},
        table:  &fixture.table,
        args:   args,
        ops:    &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(fixture.scratch.reference_count == 0);
    assert(fixture.pipeline_cells[0].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 0);
    assert(fixture.spans.allocation_cells[1].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 0);
    assert(fixture.spans.allocation_cells[0].value
        .tracked_command_references.load(AtomicOrdering.RELAXED) == 0);
    assert(capture.pushes == 0 && capture.indirect_traces == 0);
}

fn void full_trace_validation_rejects_shape_usage_and_dimensions() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    vk::PreparedTraceRays trace = vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )!!;
    assert(trace.state.ray_generation.device_address == 0x1000);
    assert(trace.state.miss.device_address == 0x1000);

    fixture.table.ray_generation.stride = 16;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT);
    fixture.table.ray_generation.stride = 32;
    fixture.table.miss.span = fixture_trace_sbt_span(&fixture, 32, 64);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT,
        "SBT bases use base alignment rather than handle alignment");
    fixture.table.miss.span = fixture_trace_sbt_span(&fixture, 0, 64);
    fixture.table.ray_generation.span.size = 64;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT,
        "the ray-generation region contains exactly one record");
    fixture.table.ray_generation.span.size = 32;
    fixture.spans.allocation_cells[1].value.backing.usage.shader_binding_table = false;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT);
    fixture.spans.allocation_cells[1].value.backing.usage.shader_binding_table = true;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 0, 1, 1 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { uint::max, uint::max, 2 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 8, 8, 3 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 8, 1, 1 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 1, 7, 1 },
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 1, 1, 6 },
    )) == gpu::INVALID_ARGUMENT);
}

fn void trace_rejects_noncanonical_empty_and_render_pass() @test {
    TraceRaysFixture fixture;
    init_trace_rays_fixture(&fixture);
    fixture.table.hit.span.owner = 1;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 1, 1, 1 },
    )) == gpu::INVALID_ARGUMENT);
    fixture.table.hit = {};
    fixture.record.state =
        gpu::internal::CommandState.RECORDING_RENDER_PASS;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 1, 1, 1 },
    )) == gpu::COMMAND_RECORDING_ERROR);
}
