module gpu_bench;

import gpu @public;
import std::io;
import std::time;

const char[*] COMMAND_BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint COMMAND_RECORD_ITERATIONS = 20_000;
const uint COMMAND_RECORD_REPETITIONS = 5;

fn double record_global_barriers(gpu::Device* device) {
    gpu::FrameToken frame = gpu::begin_frame(device)!!;
    gpu::CommandList commands = gpu::begin_commands(device, gpu::QueueKind.GRAPHICS)!!;
    gpu::GlobalBarrier barrier = {
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
    };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_global_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::end_commands(&commands)!!;
    gpu::end_frame(&frame)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_buffer_barriers(gpu::Device* device, gpu::BufferHandle buffer) {
    gpu::FrameToken frame = gpu::begin_frame(device)!!;
    gpu::CommandList commands = gpu::begin_commands(device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferBarrier barrier = {
        .buffer        = buffer,
        .size          = 64,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
    };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_buffer_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::end_commands(&commands)!!;
    gpu::end_frame(&frame)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_indirect_dispatches(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    gpu::FrameToken frame = gpu::begin_frame(device)!!;
    gpu::CommandList commands = gpu::begin_commands(device, gpu::QueueKind.GRAPHICS)!!;
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_dispatch_indirect(
            commands: &commands,
            pipeline: pipeline,
            root:     root,
            args:     args,
        )!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::end_commands(&commands)!!;
    gpu::end_frame(&frame)!!;
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
        .frames_in_flight = 1,
        .application_name = "gpu_command_record_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buffer_desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buffer_desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;
    gpu::GpuSpan buffer_span = gpu::get_buffer_span(&device, buffer)!!;

    gpu::BufferDesc args_desc = {
        .size        = gpu::DispatchIndirectCommand::size,
        .usage       = { .indirect },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle args_buffer = gpu::create_buffer(&device, &args_desc)!!;
    defer gpu::destroy_buffer(&device, args_buffer)!!;
    gpu::GpuSpan args = gpu::get_buffer_span(&device, args_buffer)!!;

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = COMMAND_BENCH_SHADER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!!;
    defer gpu::destroy_shader(&device, shader)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    (void)record_global_barriers(&device);
    (void)record_buffer_barriers(&device, buffer);
    (void)record_indirect_dispatches(&device, pipeline, buffer_span.gpu, args);

    double[COMMAND_RECORD_REPETITIONS] global_results;
    double[COMMAND_RECORD_REPETITIONS] buffer_results;
    double[COMMAND_RECORD_REPETITIONS] indirect_results;
    for (uint i = 0; i < COMMAND_RECORD_REPETITIONS; i++) {
        global_results[i] = record_global_barriers(&device);
        buffer_results[i] = record_buffer_barriers(&device, buffer);
        indirect_results[i] = record_indirect_dispatches(&device, pipeline, buffer_span.gpu, args);
    }

    io::printfn(
        "global barrier: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(global_results),
    );
    io::printfn(
        "buffer barrier: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(buffer_results),
    );
    io::printfn(
        "indirect dispatch: iterations=%d repetitions=%d median=%.1f ns/record",
        COMMAND_RECORD_ITERATIONS, COMMAND_RECORD_REPETITIONS, command_median(indirect_results),
    );
    return 0;
}
