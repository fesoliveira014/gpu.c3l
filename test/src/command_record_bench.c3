module gpu_bench;

import gpu @public;
import std::io;
import std::time;

const char[*] COMMAND_BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint COMMAND_RECORD_ITERATIONS = 20_000;
const uint COMMAND_RECORD_REPETITIONS = 5;

fn double record_barriers(gpu::Device* device) {
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_hazard_barriers(gpu::Device* device) {
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
        .hazards = {
            .draw_arguments = true,
            .descriptors = true,
            .depth_stencil = true,
        },
    };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_indirect_dispatches(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_dispatch_indirect(
            commands: &commands,
            root:     root,
            args:     args,
        )!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double command_median(double[COMMAND_RECORD_REPETITIONS] values) {
    double[COMMAND_RECORD_REPETITIONS] sorted = values;
    for (uint i = 1; i < COMMAND_RECORD_REPETITIONS; i++) {
        double key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    return sorted[COMMAND_RECORD_REPETITIONS / 2];
}

fn int main() {
    gpu::DeviceDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_command_record_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_bench_device(&device)!!;

    gpu::AllocationDesc data_desc = {
        .size         = 64,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "command_record_data",
    };
    gpu::GpuAllocation data = gpu::allocate_memory(&device, &data_desc)!!;
    defer gpu::free_allocation(&device, &data)!!;
    gpu::GpuSpan buffer_span = gpu::get_allocation_span(&device, data)!!;

    gpu::AllocationDesc args_desc = {
        .size         = gpu::DispatchIndirectCommand::size,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "command_record_args",
    };
    gpu::GpuAllocation args_allocation = gpu::allocate_memory(&device, &args_desc)!!;
    defer gpu::free_allocation(&device, &args_allocation)!!;
    gpu::GpuSpan args = gpu::get_allocation_span(&device, args_allocation)!!;

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = COMMAND_BENCH_SHADER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    (void)record_barriers(&device);
    (void)record_hazard_barriers(&device);
    (void)record_indirect_dispatches(&device, pipeline, gpu::get_span_address(&device, buffer_span)!!, args);

    double[COMMAND_RECORD_REPETITIONS] barrier_results;
    double[COMMAND_RECORD_REPETITIONS] hazard_results;
    double[COMMAND_RECORD_REPETITIONS] indirect_results;
    for (uint i = 0; i < COMMAND_RECORD_REPETITIONS; i++) {
        barrier_results[i] = record_barriers(&device);
        hazard_results[i] = record_hazard_barriers(&device);
        indirect_results[i] = record_indirect_dispatches(&device, pipeline, gpu::get_span_address(&device, buffer_span)!!, args);
    }

    io::printfn(
        "barrier: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(barrier_results),
    );
    io::printfn(
        "hazard barrier: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(hazard_results),
    );
    io::printfn(
        "indirect dispatch: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(indirect_results),
    );
    return 0;
}
