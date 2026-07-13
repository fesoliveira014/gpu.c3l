module gpu_test;

import gpu::compat;

// Headless (lavapipe): the root-pointer compute ABI proof. A compute shader
// reads/writes buffers through GPU addresses carried in a root struct pushed as
// one 64-bit push constant, with no descriptor sets. SPIR-V is $embedded from
// test/src/shaders/ (placed there by scripts/build_shaders.sh).

const char[*] ROOT_POINTER_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn void root_pointer_compute_readback() @test {
    const uint COUNT = 256;
    gpu::compat::DeviceDesc device_desc = {
        .backend = gpu::compat::BackendKind.VULKAN, .enable_validation = true,
        .frames_in_flight = 2, .application_name = "gpu_root_pointer_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc io_desc = {
        .size = COUNT * float::size, .usage = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "io",
    };
    gpu::compat::BufferHandle input = gpu::compat::create_buffer(&device, &io_desc)!!;
    defer gpu::compat::destroy_buffer(&device, input)!!;
    gpu::compat::BufferHandle output = gpu::compat::create_buffer(&device, &io_desc)!!;
    defer gpu::compat::destroy_buffer(&device, output)!!;

    gpu::compat::GpuSpan in_span = gpu::compat::get_buffer_span(&device, input)!!;
    float* in_data = (float*)in_span.cpu;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;

    gpu::compat::ShaderDesc shader_desc = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = ROOT_POINTER_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle shader = gpu::compat::create_shader(&device, &shader_desc)!!;
    defer gpu::compat::destroy_shader(&device, shader)!!;

    gpu::compat::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::compat::get_buffer_address(&device, input)!!;
    root.output_gpu = gpu::compat::get_buffer_address(&device, output)!!;
    root.count      = COUNT;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;
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
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::GpuSpan out_span = gpu::compat::get_buffer_span(&device, output)!!;
    float* out_data = (float*)out_span.cpu;
    for (uint i = 0; i < COUNT; i++) assert(out_data[i] == (float)i * 2.0, "output[i] == input[i]*2");
    gpu::compat::end_frame(&frame_token)!!;
}
