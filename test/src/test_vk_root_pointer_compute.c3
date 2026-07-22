module gpu_test;

import gpu;

const char[*] ROOT_POINTER_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn void root_pointer_compute_readback() @test {
    const uint COUNT = 256;
    gpu::RuntimeDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_root_pointer_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan input = allocate_owned_span(
        device:       &device,
        size:         COUNT * float::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "input",
    )!!;
    defer free_owned_span(&device, &input)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         COUNT * float::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "output",
    )!!;
    defer free_owned_span(&device, &output)!!;

    float* input_data = (float*)test_span_mapping(&device, input.span).ptr;
    for (uint i = 0; i < COUNT; i++) input_data[i] = (float)i;
    gpu::flush_mapped_span(&device, input.span)!!;

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = ROOT_POINTER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "compute_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    ComputeRoot* root = (ComputeRoot*)test_span_mapping(&device, root_span).ptr;
    root.input_gpu = test_span_address(&device, input.span);
    root.output_gpu = test_span_address(&device, output.span);
    root.count = COUNT;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(&device, root_span),
        groups:   { (COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)0,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(&device, output.span)!!;

    float* output_data = (float*)test_span_mapping(&device, output.span).ptr;
    for (uint i = 0; i < COUNT; i++) {
        assert(output_data[i] == (float)i * 2.0, "output equals doubled input");
    }
}
