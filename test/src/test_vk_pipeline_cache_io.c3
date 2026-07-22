module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::core::mem;
import vk;

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

fn void capture_cache_backend_diagnostic(gpu::DebugMessage* message, void* user_data) {
    CacheBackendCapture* capture = (CacheBackendCapture*)user_data;
    capture.calls++;
    if (message.category.backend
        && message.has_fault
        && message.operation != null
        && message.operation.str_view() == "create_device") capture.terminal_calls++;
    capture.matched = message.severity == gpu::DebugMessageSeverity.ERROR
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

fn void capture_cache_diagnostic(gpu::DebugMessage* message, void* user_data) {
    CacheDiagnosticCapture* capture = (CacheDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == gpu::INVALID_ARGUMENT
        && message.resource.kind == gpu::DebugResourceKind.DEVICE
        && !message.resource.has_public_identity
        && message.operation != null
        && message.operation.str_view() == "get_pipeline_cache_data"
        && message.rejected_field != null
        && message.rejected_field.str_view() == "out"
        && message.invariant != null
        && message.invariant.str_view() == "output buffer must fit the pipeline cache data";
}

fn gpu::Device make_cache_io_device(char[] blob = {}, CacheBackendCapture* capture = null) {
    gpu::RuntimeDesc desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .pipeline_cache_data = blob,
        .debug_callback      = capture == null ? null : &capture_cache_backend_diagnostic,
        .debug_user_data     = capture,
        .application_name    = "gpu_pipeline_cache_io_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::PipelineHandle build_cache_io_pipeline(gpu::Device* device) {
    gpu::ShaderDesc shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_IO_COMP_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader };
    return gpu::create_compute_pipeline(device, &pipe_desc)!!;
}

fn void run_cache_io_dispatch(gpu::Device* device, gpu::PipelineHandle pipeline) {
    const uint COUNT = 64;
    OwnedSpan input = allocate_owned_span(
        device:       device,
        size:         COUNT * float::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_input",
    )!!;
    defer free_owned_span(device, &input)!!;
    OwnedSpan output = allocate_owned_span(
        device:       device,
        size:         COUNT * float::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_output",
    )!!;
    defer free_owned_span(device, &output)!!;

    float* in_data = (float*)test_span_mapping(device, input.span).ptr;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;
    gpu::flush_mapped_span(device, input.span)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "cache_dispatch_root",
    )!!;
    defer free_owned_span(device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    ComputeRoot* root = (ComputeRoot*)test_span_mapping(device, root_span).ptr;
    root.input_gpu  = test_span_address(device, input.span);
    root.output_gpu = test_span_address(device, output.span);
    root.count      = COUNT;
    gpu::flush_mapped_span(device, root_span)!!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(device, root_span),
        groups:   { (COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;
    gpu::Barrier bar = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &bar)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        device,
        gpu::QueueKind.COMPUTE,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(device, output.span)!!;
    float* out_data = (float*)test_span_mapping(device, output.span).ptr;
    for (uint i = 0; i < COUNT; i++) assert(out_data[i] == (float)i * 2.0, "warm pipeline computes correctly");
}

fn char[] export_cache_blob(gpu::Device* device) {
    usz size = gpu::get_pipeline_cache_size(device)!!;
    assert(size > 0, "cache blob has non-zero size after a pipeline build");
    char[] blob = mem::new_array(char, (sz)size);
    usz written = gpu::get_pipeline_cache_data(device, blob)!!;
    assert(written > 0 && written <= size, "export writes within the queried size");
    return blob[:written];
}

fn void cache_blob_round_trip() @test {
    gpu::Device device_a = make_cache_io_device();
    gpu::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    run_cache_io_dispatch(&device_a, pipe_a);

    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);

    // Idempotent: a second export without new pipelines returns the same bytes.
    char[] blob2 = export_cache_blob(&device_a);
    defer mem::free(blob2.ptr);
    assert(blob.len == blob2.len, "export is idempotent in size");
    assert(mem::equals(blob, blob2), "export is idempotent in content");

    gpu::destroy_pipeline(&device_a, pipe_a)!!;
    destroy_test_device(&device_a)!!;

    gpu::Device device_b = make_cache_io_device(blob);
    defer destroy_test_device(&device_b)!!;
    gpu::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void cache_export_undersized_buffer_faults() @test {
    gpu::Device device = make_cache_io_device();
    defer destroy_test_device(&device)!!;
    gpu::PipelineHandle pipe = build_cache_io_pipeline(&device);
    defer gpu::destroy_pipeline(&device, pipe)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    CacheDiagnosticCapture capture;
    state.debug_callback = &capture_cache_diagnostic;
    state.debug_user_data = &capture;
    char[1] tiny;
    fault f = @catch(gpu::get_pipeline_cache_data(&device, tiny[..]));
    assert(f == gpu::INVALID_ARGUMENT, "undersized export buffer faults INVALID_ARGUMENT");
    assert(capture.calls == 1 && capture.matched,
        "undersized export emits one exact callback-time diagnostic");

    state.debug_callback = null;
    fault fallback = @catch(gpu::get_pipeline_cache_data(&device, tiny[..]));
    assert(fallback == f, "null callback preserves pipeline-cache export fault parity");
    assert(capture.calls == 1, "null callback emits no structured duplicate");
}

fn void cache_corrupt_blob_tolerated() @test {
    gpu::Device device_a = make_cache_io_device();
    gpu::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);

    // Flip bytes past the 32-byte header: the driver must reject or absorb
    // the payload; either way pipelines must still build and run.
    foreach (usz i, &b : blob) {
        if (i >= 32) *b ^= (char)0xA5;
    }
    vk::VkDeviceState* state_a = (vk::VkDeviceState*)gpu::device_backend_state(device_a);
    assert(gpu::vk::cache_blob_compatible(state_a, blob),
        "corrupt payload retains a compatible header and reaches the driver retry path");
    gpu::destroy_pipeline(&device_a, pipe_a)!!;
    destroy_test_device(&device_a)!!;

    CacheBackendCapture retry_capture;
    gpu::Device device_b = make_cache_io_device(blob, &retry_capture);
    assert(retry_capture.terminal_calls == 0,
        "a successful empty-cache retry emits no terminal diagnostic for the rejected blob");
    defer destroy_test_device(&device_b)!!;
    gpu::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void cache_foreign_uuid_blob_starts_empty() @test {
    gpu::Device device_a = make_cache_io_device();
    gpu::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);
    gpu::destroy_pipeline(&device_a, pipe_a)!!;
    destroy_test_device(&device_a)!!;

    // Flip the UUID bytes: a foreign-device blob must never break device
    // creation, whichever layer drops it (pre-screen, retry, or the driver).
    foreach (usz i, &b : blob) {
        if (i >= 16 && i < 32) *b ^= (char)0xA5;
    }
    gpu::Device device_b = make_cache_io_device(blob);
    defer destroy_test_device(&device_b)!!;
    gpu::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void present_mode_query_requires_valid_swapchain() @test {
    gpu::Device device = make_cache_io_device();
    defer destroy_test_device(&device)!!;

    gpu::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault f = @catch(gpu::get_present_mode_support(&device, bogus));
    assert(f == gpu::INVALID_HANDLE, "query on a bogus swapchain faults INVALID_HANDLE");
}

fn void pipeline_cache_retry_seam_is_exact_and_silent_on_success() @test {
    vk::VkDeviceState state;
    CacheBackendCapture capture;
    state.debug_callback = &capture_cache_backend_diagnostic;
    state.debug_user_data = &capture;
    PipelineCacheCreateProbe probe;
    gpu::vk::PipelineCacheCreateOps ops = {
        .user_data = &probe,
        .create    = &scripted_pipeline_cache_create,
    };
    char[4] compatible_blob = { 1, 2, 3, 4 };

    gpu::vk::create_driver_pipeline_cache_with_ops(
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
        .expected_fault     = gpu::INVALID_ARGUMENT,
        .expected_operation = "get_pipeline_cache_data",
        .expected_text      = "Vulkan returned incomplete pipeline cache data",
        .expected_result    = (int)vk::Result.INCOMPLETE,
    };
    fault incomplete = @catch(gpu::vk::check_pipeline_cache_data_backend_result(
        state:     &state,
        operation: "get_pipeline_cache_data",
        result:    vk::Result.INCOMPLETE,
    ));
    assert(incomplete == gpu::INVALID_ARGUMENT && capture.calls == 1 && capture.matched,
        "pipeline-cache INCOMPLETE result reports once without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::OUT_OF_HOST_MEMORY,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan host memory exhausted",
        .expected_result    = (int)vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    };
    fault terminal = @catch(gpu::vk::check_pipeline_cache_terminal_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    ));
    assert(terminal == gpu::OUT_OF_HOST_MEMORY && capture.calls == 2 && capture.matched,
        "pipeline-cache terminal result uses operation-aware backend routing");

    gpu::vk::check_pipeline_cache_terminal_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.SUCCESS,
    )!!;
    assert(capture.calls == 2, "successful retry completion emits no diagnostic");

    state.debug_callback = null;
    fault fallback = @catch(gpu::vk::check_pipeline_cache_data_backend_result(
        state:     &state,
        operation: "get_pipeline_cache_data",
        result:    vk::Result.INCOMPLETE,
    ));
    assert(fallback == incomplete && capture.calls == 2,
        "null callback preserves cache-data fault without a duplicate");
}
