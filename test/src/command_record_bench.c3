module gpu_bench;

import gpu @public;
import gpu::vk @public;
import std::io;
import std::time;

const char[*] COMMAND_BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint COMMAND_RECORD_ITERATIONS = 20_000;
const uint GENERATED_RECORD_ITERATIONS = 1_000;
const uint COMMAND_RECORD_REPETITIONS = 5;
const uint MEASURED_COMMAND_PATH_COUNT = 3;

fn void accumulate_resolution_stats(
    gpu::CommandResolutionStats* total,
    gpu::CommandResolutionStats sample,
) {
    total.device_registry_resolutions += sample.device_registry_resolutions;
    total.retained_pin_operations += sample.retained_pin_operations;
    total.lifecycle_vtable_dispatches += sample.lifecycle_vtable_dispatches;
    total.command_table_lookups += sample.command_table_lookups;
    total.pipeline_table_lookups += sample.pipeline_table_lookups;
    total.pipeline_cache_lookups += sample.pipeline_cache_lookups;
    total.command_policy_selections += sample.command_policy_selections;
    total.recording_commands += sample.recording_commands;
    total.native_commands += sample.native_commands;
}

fn double record_barriers(
    gpu::Device* device,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::reset_command_resolution_stats();
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    accumulate_resolution_stats(resolution, gpu::command_resolution_stats());
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_hazard_barriers(
    gpu::Device* device,
    gpu::CommandResolutionStats* resolution,
) {
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
    gpu::reset_command_resolution_stats();
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_barrier(&commands, &barrier)!!;
    }
    long elapsed_ns = (long)start.to_now();
    accumulate_resolution_stats(resolution, gpu::command_resolution_stats());
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_indirect_dispatches(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::reset_command_resolution_stats();
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_RECORD_ITERATIONS; i++) {
        gpu::cmd_dispatch_indirect(
            commands: &commands,
            root:     root,
            args:     args,
        )!!;
    }
    long elapsed_ns = (long)start.to_now();
    accumulate_resolution_stats(resolution, gpu::command_resolution_stats());
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&commands_executable)!!;
    return (double)elapsed_ns / (double)COMMAND_RECORD_ITERATIONS;
}

fn double record_generated_dispatches(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuSpan records,
    gpu::GpuSpan count,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::reset_command_resolution_stats();
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < GENERATED_RECORD_ITERATIONS; i++) {
        gpu::cmd_dispatch_generated(
            commands:           &commands,
            records:            records,
            count_span:         count,
            max_dispatch_count: 1,
        )!!;
    }
    long elapsed_ns = (long)start.to_now();
    accumulate_resolution_stats(resolution, gpu::command_resolution_stats());
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_bench_executable(&executable)!!;
    return (double)elapsed_ns / (double)GENERATED_RECORD_ITERATIONS;
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
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_command_record_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
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

    gpu::AllocationDesc generated_records_desc = {
        .size         = gpu::GeneratedDispatchRecord::size,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics },
        .debug_name   = "command_record_generated_records",
    };
    gpu::GpuAllocation generated_records_allocation = gpu::allocate_memory(
        &device,
        &generated_records_desc,
    )!!;
    defer gpu::free_allocation(&device, &generated_records_allocation)!!;
    gpu::GpuSpan generated_records = gpu::get_allocation_span(
        &device,
        generated_records_allocation,
    )!!;
    gpu::AllocationDesc generated_count_desc = {
        .size         = 4,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics },
        .debug_name   = "command_record_generated_count",
    };
    gpu::GpuAllocation generated_count_allocation = gpu::allocate_memory(
        &device,
        &generated_count_desc,
    )!!;
    defer gpu::free_allocation(&device, &generated_count_allocation)!!;
    gpu::GpuSpan generated_count = gpu::get_allocation_span(
        &device,
        generated_count_allocation,
    )!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;

    gpu::CommandResolutionStats warmup_resolution;
    (void)record_barriers(&device, &warmup_resolution);
    (void)record_hazard_barriers(&device, &warmup_resolution);
    (void)record_indirect_dispatches(
        &device,
        pipeline,
        gpu::get_span_address(&device, buffer_span)!!,
        args,
        &warmup_resolution,
    );
    if (caps.generated_work) {
        (void)record_generated_dispatches(
            &device,
            pipeline,
            generated_records,
            generated_count,
            &warmup_resolution,
        );
    }

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong registry_locks_before = gpu::device_registry_lock_count();
    vk::CommandRecordingStats recording_before =
        vk::command_recording_stats(state);
    ulong pipelines_before = vk::pipeline_native_create_count(state);

    double[COMMAND_RECORD_REPETITIONS] barrier_results;
    double[COMMAND_RECORD_REPETITIONS] hazard_results;
    double[COMMAND_RECORD_REPETITIONS] indirect_results;
    double[COMMAND_RECORD_REPETITIONS] generated_results;
    gpu::CommandResolutionStats resolution;
    for (uint i = 0; i < COMMAND_RECORD_REPETITIONS; i++) {
        barrier_results[i] = record_barriers(&device, &resolution);
        hazard_results[i] = record_hazard_barriers(&device, &resolution);
        indirect_results[i] = record_indirect_dispatches(
            &device,
            pipeline,
            gpu::get_span_address(&device, buffer_span)!!,
            args,
            &resolution,
        );
        if (caps.generated_work) {
            generated_results[i] = record_generated_dispatches(
                &device,
                pipeline,
                generated_records,
                generated_count,
                &resolution,
            );
        }
    }
    ulong expected_recording_commands = COMMAND_RECORD_ITERATIONS
        * COMMAND_RECORD_REPETITIONS * MEASURED_COMMAND_PATH_COUNT;
    if (caps.generated_work) {
        expected_recording_commands += GENERATED_RECORD_ITERATIONS
            * COMMAND_RECORD_REPETITIONS;
    }
    ulong registry_locks =
        gpu::device_registry_lock_count() - registry_locks_before;
    vk::CommandRecordingStats recording_after =
        vk::command_recording_stats(state);
    ulong recording_allocations = recording_after.recording_allocations
        - recording_before.recording_allocations;
    ulong preprocess_allocations = recording_after.preprocess_allocations
        - recording_before.preprocess_allocations;
    ulong preprocess_reuses = recording_after.preprocess_reuses
        - recording_before.preprocess_reuses;
    ulong draw_compilations = vk::pipeline_native_create_count(state)
        - pipelines_before;
    assert(registry_locks == 0,
        "recording does not acquire the device registry lock");
    assert(recording_allocations == 0,
        "warmed recording performs no backend allocation");
    assert(preprocess_allocations == 0,
        "generated recording uses distinct pooled preprocess storage");
    assert(draw_compilations == 0,
        "recording does not compile pipelines");
    assert(resolution.device_registry_resolutions == 0,
        "warm recording does not resolve the device registry");
    assert(resolution.retained_pin_operations == 0,
        "warm recording does not borrow retained pins");
    assert(resolution.lifecycle_vtable_dispatches == 0,
        "warm recording does not load the lifecycle vtable");
    assert(resolution.command_table_lookups == 0,
        "warm recording does not resolve the backend command table");
    assert(resolution.pipeline_table_lookups == 0,
        "warm recording does not resolve the pipeline table after bind");
    assert(resolution.pipeline_cache_lookups == 0,
        "warm recording does not resolve the pipeline cache after bind");
    assert(resolution.command_policy_selections == 0,
        "warm recording does not select a validation policy");
    assert(resolution.recording_commands == expected_recording_commands,
        "resolution evidence counts every measured recording call");
    assert(resolution.native_commands >= resolution.recording_commands,
        "measured recording calls emit native command work");
    if (caps.generated_work) {
        assert(preprocess_reuses
            >= GENERATED_RECORD_ITERATIONS * COMMAND_RECORD_REPETITIONS,
            "each measured execute uses distinct pooled preprocess storage");
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
    if (caps.generated_work) {
        io::printfn(
            "generated dispatch: iterations=%d repetitions=%d median=%.1f ns/record",
            GENERATED_RECORD_ITERATIONS,
            COMMAND_RECORD_REPETITIONS,
            command_median(generated_results),
        );
    } else {
        io::printfn("generated dispatch: unsupported");
    }
    io::printfn(
        "invariants: registry_locks=%d recording_allocations=%d draw_compilations=%d preprocess_allocations=%d",
        registry_locks,
        recording_allocations,
        draw_compilations,
        preprocess_allocations,
    );
    io::printfn(
        "resolution: recording_commands=%d native_commands=%d device_registry=%d retained_pins=%d lifecycle_vtable=%d command_table=%d pipeline_table=%d pipeline_cache=%d policy=%d",
        resolution.recording_commands,
        resolution.native_commands,
        resolution.device_registry_resolutions,
        resolution.retained_pin_operations,
        resolution.lifecycle_vtable_dispatches,
        resolution.command_table_lookups,
        resolution.pipeline_table_lookups,
        resolution.pipeline_cache_lookups,
        resolution.command_policy_selections,
    );
    io::printfn("generated preprocess: reuse_events=%d", preprocess_reuses);
    return 0;
}
