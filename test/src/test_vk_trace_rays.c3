module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic::types;

struct TraceRaysCapture {
    gpu::GpuAddress expected_root;
    uint            pushes;
    uint            traces;
    bool            matched;
}

struct RayTracingBindCapture {
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
        && miss.device_address == 0x1020
        && miss.stride == 32
        && miss.size == 32
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

struct TraceRaysFixture {
    SpanResolverFixture    spans;
    vk::CommandRecord      record;
    gpu::RayTracingShaderBindingTable table;
    vk::PipelineCell[1]    pipeline_cells;
    vk::PipelineCacheEntry[1] pipeline_entries;
    vk::CommandResourceReference[4] references;
    vk::CommandScratch     scratch;
}

fn void init_trace_rays_fixture(TraceRaysFixture* fixture) {
    init_span_resolver_fixture(&fixture.spans);
    fixture.spans.buffer_cells[0].value.usage.shader_binding_table = true;
    fixture.spans.state.ray_tracing_pipelines_enabled = true;
    fixture.spans.state.caps.ray_tracing_pipelines = {
        .enabled                           = true,
        .max_ray_dispatch_invocation_count = 128,
        .shader_group_handle_size          = 32,
        .shader_group_handle_alignment     = 32,
        .shader_group_base_alignment       = 32,
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
            .span   = fixture_buffer_span(&fixture.spans, 0, 32),
            .stride = 32,
        },
        .miss = {
            .span   = fixture_buffer_span(&fixture.spans, 32, 32),
            .stride = 32,
        },
    };
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
    assert(fixture.spans.buffer_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
    vk::rollback_command_references(&fixture.record, 0);
    assert(fixture.scratch.reference_count == 0);
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
    assert(trace.ray_generation.device_address == 0x1000);
    assert(trace.miss.device_address == 0x1020);

    fixture.table.ray_generation.stride = 16;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT);
    fixture.table.ray_generation.stride = 32;
    fixture.spans.buffer_cells[0].value.usage.shader_binding_table = false;
    assert(@catch(vk::full_prepare_trace_rays(
        &fixture.record,
        (gpu::GpuAddress)1,
        &fixture.table,
        { 2, 3, 4 },
    )) == gpu::INVALID_ARGUMENT);
    fixture.spans.buffer_cells[0].value.usage.shader_binding_table = true;
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
