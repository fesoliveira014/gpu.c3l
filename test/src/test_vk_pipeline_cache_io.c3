module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import std::core::mem;
import vk;

// Headless (lavapipe): driver pipeline-cache blob export/import. Correctness
// only — blob content usefulness is driver-dependent, so no timing asserts.

const char[*] CACHE_IO_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

struct CacheDiagnosticCapture {
    uint calls;
    bool matched;
}

struct CacheBackendCapture {
    uint calls;
    uint terminal_calls;
    fault expected_fault;
    ZString expected_operation;
    ZString expected_text;
    int expected_result;
    bool matched;
}

struct PipelineCacheCreateProbe {
    uint attempts;
    sz first_size;
    sz second_size;
}

fn vk::Result scripted_pipeline_cache_create(
    void* user_data,
    vk::Device device,
    vk::PipelineCacheCreateInfo* info,
    vk::PipelineCache* out_cache,
) {
    PipelineCacheCreateProbe* probe = (PipelineCacheCreateProbe*)user_data;
    probe.attempts++;
    if (probe.attempts == 1) {
        probe.first_size = info.initial_data_size;
        return vk::Result.ERROR_UNKNOWN;
    }
    probe.second_size = info.initial_data_size;
    *out_cache = (vk::PipelineCache)(void*)1;
    return vk::Result.SUCCESS;
}

fn void capture_cache_backend_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    CacheBackendCapture* capture = (CacheBackendCapture*)user_data;
    capture.calls++;
    if (message.category.backend
        && message.has_fault
        && message.operation != null
        && message.operation.str_view() == "create_device") capture.terminal_calls++;
    capture.matched = message.severity == gpu::compat::DebugMessageSeverity.ERROR
        && message.category.backend
        && message.has_fault
        && message.public_fault == capture.expected_fault
        && message.operation != null
        && message.operation.str_view() == capture.expected_operation.str_view()
        && message.backend_text != null
        && message.backend_text.str_view() == capture.expected_text.str_view()
        && message.validation_id_number == capture.expected_result
        && !message.resource.has_public_identity;
}

fn void capture_cache_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    CacheDiagnosticCapture* capture = (CacheDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::compat::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == gpu::compat::INVALID_ARGUMENT
        && message.resource.kind == gpu::compat::DebugResourceKind.DEVICE
        && !message.resource.has_public_identity
        && message.operation != null
        && message.operation.str_view() == "get_pipeline_cache_data"
        && message.rejected_field != null
        && message.rejected_field.str_view() == "out"
        && message.invariant != null
        && message.invariant.str_view() == "output buffer must fit the pipeline cache data";
}

fn gpu::compat::Device make_cache_io_device(char[] blob = {}, CacheBackendCapture* capture = null) {
    gpu::compat::DeviceDesc desc = {
        .backend             = gpu::compat::BackendKind.VULKAN,
        .enable_validation   = true,
        .frames_in_flight    = 2,
        .pipeline_cache_data = blob,
        .debug_callback      = capture == null ? null : &capture_cache_backend_diagnostic,
        .debug_user_data     = capture,
        .application_name    = "gpu_pipeline_cache_io_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::PipelineHandle build_cache_io_pipeline(gpu::compat::Device* device) {
    gpu::compat::ShaderDesc shader_desc = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = CACHE_IO_COMP_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle shader = gpu::compat::create_shader(device, &shader_desc)!!;
    defer gpu::compat::destroy_shader(device, shader)!!;
    gpu::compat::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::compat::RootPush::size };
    return gpu::compat::create_compute_pipeline(device, &pipe_desc)!!;
}

fn void run_cache_io_dispatch(gpu::compat::Device* device, gpu::compat::PipelineHandle pipeline) {
    const uint COUNT = 64;
    gpu::compat::BufferDesc io_desc = {
        .size = COUNT * float::size, .usage = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "cache_io",
    };
    gpu::compat::BufferHandle input = gpu::compat::create_buffer(device, &io_desc)!!;
    defer gpu::compat::destroy_buffer(device, input)!!;
    gpu::compat::BufferHandle output = gpu::compat::create_buffer(device, &io_desc)!!;
    defer gpu::compat::destroy_buffer(device, output)!!;
    gpu::compat::GpuSpan in_span = gpu::compat::get_buffer_span(device, input)!!;
    float* in_data = (float*)in_span.cpu;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(device)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::compat::get_buffer_address(device, input)!!;
    root.output_gpu = gpu::compat::get_buffer_address(device, output)!!;
    root.count      = COUNT;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        groups:   { (COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;
    gpu::compat::BufferBarrier bar = {
        .buffer = output, .offset = 0, .size = COUNT * float::size,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(device, &submit)!!;
    gpu::compat::wait_queue_idle(device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::GpuSpan out_span = gpu::compat::get_buffer_span(device, output)!!;
    float* out_data = (float*)out_span.cpu;
    for (uint i = 0; i < COUNT; i++) assert(out_data[i] == (float)i * 2.0, "warm pipeline computes correctly");
    gpu::compat::end_frame(&frame_token)!!;
}

fn char[] export_cache_blob(gpu::compat::Device* device) {
    usz size = gpu::compat::get_pipeline_cache_size(device)!!;
    assert(size > 0, "cache blob has non-zero size after a pipeline build");
    char[] blob = mem::new_array(char, (sz)size);
    usz written = gpu::compat::get_pipeline_cache_data(device, blob)!!;
    assert(written > 0 && written <= size, "export writes within the queried size");
    return blob[:written];
}

fn void cache_blob_round_trip() @test {
    gpu::compat::Device device_a = make_cache_io_device();
    gpu::compat::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    run_cache_io_dispatch(&device_a, pipe_a);

    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);

    // Idempotent: a second export without new pipelines returns the same bytes.
    char[] blob2 = export_cache_blob(&device_a);
    defer mem::free(blob2.ptr);
    assert(blob.len == blob2.len, "export is idempotent in size");
    assert(mem::equals(blob, blob2), "export is idempotent in content");

    gpu::compat::destroy_pipeline(&device_a, pipe_a)!!;
    gpu::compat::destroy_device(&device_a)!!;

    gpu::compat::Device device_b = make_cache_io_device(blob);
    defer gpu::compat::destroy_device(&device_b)!!;
    gpu::compat::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::compat::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void cache_export_undersized_buffer_faults() @test {
    gpu::compat::Device device = make_cache_io_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::PipelineHandle pipe = build_cache_io_pipeline(&device);
    defer gpu::compat::destroy_pipeline(&device, pipe)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    CacheDiagnosticCapture capture;
    state.debug_callback = &capture_cache_diagnostic;
    state.debug_user_data = &capture;
    char[1] tiny;
    fault f = @catch(gpu::compat::get_pipeline_cache_data(&device, tiny[..]));
    assert(f == gpu::compat::INVALID_ARGUMENT, "undersized export buffer faults INVALID_ARGUMENT");
    assert(capture.calls == 1 && capture.matched,
        "undersized export emits one exact callback-time diagnostic");

    state.debug_callback = null;
    fault fallback = @catch(gpu::compat::get_pipeline_cache_data(&device, tiny[..]));
    assert(fallback == f, "null callback preserves pipeline-cache export fault parity");
    assert(capture.calls == 1, "null callback emits no structured duplicate");
}

fn void cache_corrupt_blob_tolerated() @test {
    gpu::compat::Device device_a = make_cache_io_device();
    gpu::compat::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);

    // Flip bytes past the 32-byte header: the driver must reject or absorb
    // the payload; either way pipelines must still build and run.
    foreach (usz i, &b : blob) {
        if (i >= 32) *b ^= (char)0xA5;
    }
    vk::VkDeviceState* state_a = (vk::VkDeviceState*)gpu::compat::device_backend_state(device_a);
    assert(gpu::compat::vk::cache_blob_compatible(state_a, blob),
        "corrupt payload retains a compatible header and reaches the driver retry path");
    gpu::compat::destroy_pipeline(&device_a, pipe_a)!!;
    gpu::compat::destroy_device(&device_a)!!;

    CacheBackendCapture retry_capture;
    gpu::compat::Device device_b = make_cache_io_device(blob, &retry_capture);
    assert(retry_capture.terminal_calls == 0,
        "a successful empty-cache retry emits no terminal diagnostic for the rejected blob");
    defer gpu::compat::destroy_device(&device_b)!!;
    gpu::compat::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::compat::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void cache_foreign_uuid_blob_starts_empty() @test {
    gpu::compat::Device device_a = make_cache_io_device();
    gpu::compat::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);
    gpu::compat::destroy_pipeline(&device_a, pipe_a)!!;
    gpu::compat::destroy_device(&device_a)!!;

    // Flip the UUID bytes: a foreign-device blob must never break device
    // creation, whichever layer drops it (pre-screen, retry, or the driver).
    foreach (usz i, &b : blob) {
        if (i >= 16 && i < 32) *b ^= (char)0xA5;
    }
    gpu::compat::Device device_b = make_cache_io_device(blob);
    defer gpu::compat::destroy_device(&device_b)!!;
    gpu::compat::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::compat::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void present_mode_query_requires_valid_swapchain() @test {
    gpu::compat::Device device = make_cache_io_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault f = @catch(gpu::compat::get_present_mode_support(&device, bogus));
    assert(f == gpu::compat::INVALID_HANDLE, "query on a bogus swapchain faults INVALID_HANDLE");
}

fn void pipeline_cache_retry_seam_is_exact_and_silent_on_success() @test {
    vk::VkDeviceState state;
    CacheBackendCapture capture;
    state.debug_callback = &capture_cache_backend_diagnostic;
    state.debug_user_data = &capture;
    PipelineCacheCreateProbe probe;
    gpu::compat::vk::PipelineCacheCreateOps ops = {
        .user_data = &probe,
        .create    = &scripted_pipeline_cache_create,
    };
    char[4] compatible_blob = { 1, 2, 3, 4 };

    gpu::compat::vk::create_driver_pipeline_cache_with_ops(
        state:      &state,
        cache_data: compatible_blob[..],
        ops:        &ops,
    )!!;

    assert(probe.attempts == 2, "retry seam performs exactly two cache-create attempts");
    assert(probe.first_size == compatible_blob.len, "first attempt receives the compatible blob");
    assert(probe.second_size == 0, "retry attempt receives an empty cache payload");
    assert(state.driver_cache != null, "successful retry publishes the created cache");
    assert(capture.terminal_calls == 0, "retryable first failure and successful retry emit no terminal diagnostic");
}

fn void pipeline_cache_terminal_results_emit_once_and_preserve_faults() @test {
    vk::VkDeviceState state;
    CacheBackendCapture capture;
    state.debug_callback = &capture_cache_backend_diagnostic;
    state.debug_user_data = &capture;

    capture = {
        .expected_fault     = gpu::compat::INVALID_ARGUMENT,
        .expected_operation = "get_pipeline_cache_data",
        .expected_text      = "Vulkan returned incomplete pipeline cache data",
        .expected_result    = (int)vk::Result.INCOMPLETE,
    };
    fault incomplete = @catch(gpu::compat::vk::check_pipeline_cache_data_backend_result(
        state:     &state,
        operation: "get_pipeline_cache_data",
        result:    vk::Result.INCOMPLETE,
    ));
    assert(incomplete == gpu::compat::INVALID_ARGUMENT && capture.calls == 1 && capture.matched,
        "pipeline-cache INCOMPLETE result reports once without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::compat::OUT_OF_HOST_MEMORY,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan host memory exhausted",
        .expected_result    = (int)vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    };
    fault terminal = @catch(gpu::compat::vk::check_pipeline_cache_terminal_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    ));
    assert(terminal == gpu::compat::OUT_OF_HOST_MEMORY && capture.calls == 2 && capture.matched,
        "pipeline-cache terminal result uses operation-aware backend routing");

    gpu::compat::vk::check_pipeline_cache_terminal_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.SUCCESS,
    )!!;
    assert(capture.calls == 2, "successful retry completion emits no diagnostic");

    state.debug_callback = null;
    fault fallback = @catch(gpu::compat::vk::check_pipeline_cache_data_backend_result(
        state:     &state,
        operation: "get_pipeline_cache_data",
        result:    vk::Result.INCOMPLETE,
    ));
    assert(fallback == incomplete && capture.calls == 2,
        "null callback preserves cache-data fault without a duplicate");
}
