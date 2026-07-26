module hello_gpu;

import gpu;
import std::io;

const char[*] DOUBLER_SPIRV = $embed("../shaders/doubler.comp.spv");
const uint COUNT = 256;

struct DoublerRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
}

fn int main() {
    if (catch err = run()) {
        io::printfn("hello_gpu: FAIL (%s)", err);
        return 1;
    }
    io::printn("hello_gpu: all 256 values doubled on the GPU");
    return 0;
}

fn void? run() {
    gpu::RuntimeDesc runtime_desc = gpu::full_validation_runtime_desc();
    runtime_desc.application_name = "hello_gpu";
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::Adapter adapter = adapters.get(0)!;
    gpu::Device device = gpu::create_device(&adapter)!;
    defer (void)gpu::destroy_device(&device);

    gpu::AllocationDesc input_desc = {
        .size         = COUNT * float::size,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .compute },
        .debug_name   = "input",
    };
    gpu::AllocationDesc output_desc = input_desc;
    output_desc.memory_class = gpu::MemoryClass.CPU_READ;
    output_desc.debug_name = "output";

    gpu::GpuAllocation input =
        gpu::allocate_memory(&device, &input_desc)!;
    defer (void)gpu::free_allocation(&device, &input);
    gpu::GpuAllocation output =
        gpu::allocate_memory(&device, &output_desc)!;
    defer (void)gpu::free_allocation(&device, &output);

    gpu::GpuSpan in_span = gpu::get_allocation_span(&device, input)!;
    gpu::GpuSpan out_span = gpu::get_allocation_span(&device, output)!;
    float* in_data = (float*)gpu::get_span_mapping(&device, in_span)!.ptr;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;
    gpu::flush_mapped_span(&device, in_span)!;

    gpu::ShaderDesc shader_desc = {
        .spirv       = DOUBLER_SPIRV[..],
        .entry_point = "main",
    };

    gpu::ComputePipelineDesc pipe_desc = { .shader = shader_desc };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::AllocationDesc root_desc = {
        .size         = DoublerRoot::size,
        .alignment    = DoublerRoot::alignment,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .compute },
        .debug_name   = "doubler_root",
    };
    gpu::GpuAllocation root_allocation =
        gpu::allocate_memory(&device, &root_desc)!;
    defer (void)gpu::free_allocation(&device, &root_allocation);
    gpu::GpuSpan root_span =
        gpu::get_allocation_span(&device, root_allocation)!;

    run_compute(
        device:      &device,
        pipeline:    pipeline,
        input_span:  in_span,
        output_span: out_span,
        root_span:   root_span,
    )!;
    gpu::invalidate_mapped_span(&device, out_span)!;
    float* out_data = (float*)gpu::get_span_mapping(&device, out_span)!.ptr;
    for (uint i = 0; i < COUNT; i++) {
        if (out_data[i] != (float)i * 2.0f) return gpu::INVALID_ARGUMENT~;
    }
}

fn void? run_compute(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuSpan input_span,
    gpu::GpuSpan output_span,
    gpu::GpuSpan root_span,
) {
    DoublerRoot* root =
        (DoublerRoot*)gpu::get_span_mapping(device, root_span)!.ptr;
    gpu::GpuAddress root_address =
        gpu::get_span_address(device, root_span)!;
    root.input_gpu  = gpu::get_span_address(device, input_span)!;
    root.output_gpu = gpu::get_span_address(device, output_span)!;
    root.count      = COUNT;
    gpu::flush_mapped_span(device, root_span)!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.COMPUTE)!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(device, queue)!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::cmd_bind_pipeline(&commands, pipeline)!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root_address,
        groups:   { (COUNT + 63) / 64, 1, 1 },
    )!;
    gpu::Barrier to_host = {
        .before = { .compute = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &to_host)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    defer (void)gpu::discard_executable_commands(&executable);

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;
}
