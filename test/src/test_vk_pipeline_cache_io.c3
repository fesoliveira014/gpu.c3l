module gpu_test;

import gpu;
import std::core::mem;

// Headless (lavapipe): driver pipeline-cache blob export/import. Correctness
// only — blob content usefulness is driver-dependent, so no timing asserts.

const char[*] CACHE_IO_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn gpu::Device make_cache_io_device(char[] blob = {}) {
    gpu::DeviceDesc desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .enable_validation   = true,
        .frames_in_flight    = 2,
        .pipeline_cache_data = blob,
        .application_name    = "gpu_pipeline_cache_io_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::PipelineHandle build_cache_io_pipeline(gpu::Device* device) {
    gpu::ShaderDesc shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_IO_COMP_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(device, &shader_desc)!!;
    defer gpu::destroy_shader(device, shader)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    return gpu::create_compute_pipeline(device, &pipe_desc)!!;
}

fn void run_cache_io_dispatch(gpu::Device* device, gpu::PipelineHandle pipeline) {
    const uint COUNT = 64;
    gpu::BufferDesc io_desc = {
        .size = COUNT * float::size, .usage = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "cache_io",
    };
    gpu::BufferHandle input = gpu::create_buffer(device, &io_desc)!!;
    defer gpu::destroy_buffer(device, input)!!;
    gpu::BufferHandle output = gpu::create_buffer(device, &io_desc)!!;
    defer gpu::destroy_buffer(device, output)!!;
    gpu::GpuSpan in_span = gpu::get_buffer_span(device, input)!!;
    float* in_data = (float*)in_span.cpu;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;

    gpu::begin_frame(device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(device, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::get_buffer_address(device, input)!!;
    root.output_gpu = gpu::get_buffer_address(device, output)!!;
    root.count      = COUNT;

    gpu::CommandList cmd = gpu::begin_commands(device, gpu::QueueKind.COMPUTE)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        groups:   { (COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;
    gpu::BufferBarrier bar = {
        .buffer = output, .offset = 0, .size = COUNT * float::size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::end_commands(device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(device, &submit)!!;
    gpu::wait_queue_idle(device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(device, output)!!;
    float* out_data = (float*)out_span.cpu;
    for (uint i = 0; i < COUNT; i++) assert(out_data[i] == (float)i * 2.0, "warm pipeline computes correctly");
    gpu::end_frame(device)!!;
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
    gpu::destroy_device(&device_a)!!;

    gpu::Device device_b = make_cache_io_device(blob);
    defer gpu::destroy_device(&device_b)!!;
    gpu::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void cache_export_undersized_buffer_faults() @test {
    gpu::Device device = make_cache_io_device();
    defer gpu::destroy_device(&device)!!;
    gpu::PipelineHandle pipe = build_cache_io_pipeline(&device);
    defer gpu::destroy_pipeline(&device, pipe)!!;

    char[1] tiny;
    fault f = @catch(gpu::get_pipeline_cache_data(&device, tiny[..]));
    assert(f == gpu::INVALID_ARGUMENT, "undersized export buffer faults INVALID_ARGUMENT");
}

fn void cache_corrupt_blob_tolerated() @test {
    gpu::Device device_a = make_cache_io_device();
    gpu::PipelineHandle pipe_a = build_cache_io_pipeline(&device_a);
    char[] blob = export_cache_blob(&device_a);
    defer mem::free(blob.ptr);
    gpu::destroy_pipeline(&device_a, pipe_a)!!;
    gpu::destroy_device(&device_a)!!;

    // Flip bytes past the 32-byte header: the driver must reject or absorb
    // the payload; either way pipelines must still build and run.
    foreach (usz i, &b : blob) {
        if (i >= 32) *b ^= (char)0xA5;
    }
    gpu::Device device_b = make_cache_io_device(blob);
    defer gpu::destroy_device(&device_b)!!;
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
    gpu::destroy_device(&device_a)!!;

    // Flip the UUID bytes: a foreign-device blob must never break device
    // creation, whichever layer drops it (pre-screen, retry, or the driver).
    foreach (usz i, &b : blob) {
        if (i >= 16 && i < 32) *b ^= (char)0xA5;
    }
    gpu::Device device_b = make_cache_io_device(blob);
    defer gpu::destroy_device(&device_b)!!;
    gpu::PipelineHandle pipe_b = build_cache_io_pipeline(&device_b);
    defer gpu::destroy_pipeline(&device_b, pipe_b)!!;
    run_cache_io_dispatch(&device_b, pipe_b);
}

fn void present_mode_query_requires_valid_swapchain() @test {
    gpu::Device device = make_cache_io_device();
    defer gpu::destroy_device(&device)!!;

    gpu::SwapchainHandle bogus = { .index = 3, .generation = 7 };
    fault f = @catch(gpu::get_present_mode_support(&device, bogus));
    assert(f == gpu::INVALID_HANDLE, "query on a bogus swapchain faults INVALID_HANDLE");
}
