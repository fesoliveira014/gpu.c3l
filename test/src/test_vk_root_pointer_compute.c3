module gpu_test;

import gpu;

// Headless (lavapipe): the root-pointer compute ABI proof. A compute shader
// reads/writes buffers through GPU addresses carried in a root struct pushed as
// one 64-bit push constant, with no descriptor sets. SPIR-V is $embedded from
// test/src/shaders/ (placed there by scripts/build_shaders.sh).

const char[*] ROOT_POINTER_SPIRV = $embed("shaders/root_pointer.comp.spv");

struct ComputeRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
    uint            _pad0;
    uint            _pad1;
    uint            _pad2;
}
$assert ComputeRoot::size == 32;

fn void root_pointer_compute_readback() @test {
    const uint COUNT = 256;
    gpu::DeviceDesc device_desc = {
        .backend = gpu::BackendKind.VULKAN, .enable_validation = true,
        .frames_in_flight = 2, .application_name = "gpu_root_pointer_test",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc io_desc = {
        .size = COUNT * float::size, .usage = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "io",
    };
    gpu::BufferHandle input = gpu::create_buffer(&device, &io_desc)!!;
    defer gpu::destroy_buffer(&device, input)!!;
    gpu::BufferHandle output = gpu::create_buffer(&device, &io_desc)!!;
    defer gpu::destroy_buffer(&device, output)!!;

    gpu::GpuSpan in_span = gpu::get_buffer_span(&device, input)!!;
    float* in_data = (float*)in_span.cpu;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;

    gpu::ShaderDesc shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = ROOT_POINTER_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!!;
    defer gpu::destroy_shader(&device, shader)!!;

    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = 8 };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::get_buffer_address(&device, input)!!;
    root.output_gpu = gpu::get_buffer_address(&device, output)!!;
    root.count      = COUNT;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        groups:   { (COUNT + 63) / 64, 1, 1 },
    )!!;
    gpu::BufferBarrier bar = {
        .buffer = output, .offset = 0, .size = COUNT * float::size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!!;
    float* out_data = (float*)out_span.cpu;
    for (uint i = 0; i < COUNT; i++) assert(out_data[i] == (float)i * 2.0, "output[i] == input[i]*2");
    gpu::end_frame(&device)!!;
}
