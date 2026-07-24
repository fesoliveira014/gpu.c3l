module gpu_bench;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::io;
import std::os::env;
import std::time;

const char[*] COMMAND_BENCH_SHADER_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint COMMAND_RECORD_ITERATIONS = 20_000;
const uint GENERATED_RECORD_ITERATIONS = 64;
const uint COMMAND_RECORD_REPETITIONS = 5;
const uint MEASURED_COMMAND_PATH_COUNT = 3;
const uint BENCHMARK_EXPECTATION_VERSION = 3;

struct CommandBenchPolicy {
    gpu::ContractValidation contract;
    bool                    track_lifetimes;
    bool                    vulkan_layers;
    ZString                 contract_name;
}

fn bool? command_bench_flag(ZString variable) {
    if (try value = env::tget_var(variable.str_view())) {
        if (value == "true") return true;
        if (value == "false") return false;
    }
    return gpu::INVALID_ARGUMENT~;
}

fn CommandBenchPolicy? command_bench_policy() {
    String contract_value = env::tget_var("GPU_C3L_BENCH_CONTRACT")!;
    gpu::ContractValidation contract;
    ZString contract_name;
    if (contract_value == "trusted") {
        contract = gpu::ContractValidation.TRUSTED;
        contract_name = "trusted";
    } else if (contract_value == "object_boundaries") {
        contract = gpu::ContractValidation.OBJECT_BOUNDARIES;
        contract_name = "object_boundaries";
    } else if (contract_value == "full") {
        contract = gpu::ContractValidation.FULL;
        contract_name = "full";
    } else {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .contract        = contract,
        .track_lifetimes = command_bench_flag("GPU_C3L_BENCH_TRACKING")!,
        .vulkan_layers    = command_bench_flag("GPU_C3L_BENCH_LAYERS")!,
        .contract_name    = contract_name,
    };
}

fn gpu::internal::RecordingWorkCounters recording_work_delta(
    gpu::internal::RecordingWorkCounters after,
    gpu::internal::RecordingWorkCounters before,
) {
    return {
        .host_allocations = after.host_allocations - before.host_allocations,
        .command_pool_creations = after.command_pool_creations
            - before.command_pool_creations,
        .command_buffer_allocations = after.command_buffer_allocations
            - before.command_buffer_allocations,
        .command_buffer_frees = after.command_buffer_frees
            - before.command_buffer_frees,
        .command_buffer_resets = after.command_buffer_resets
            - before.command_buffer_resets,
        .image_view_creations = after.image_view_creations
            - before.image_view_creations,
        .vma_allocations = after.vma_allocations - before.vma_allocations,
        .generated_scratch_misses = after.generated_scratch_misses
            - before.generated_scratch_misses,
    };
}

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
    total.encoder_cell_computations += sample.encoder_cell_computations;
    total.encoder_lease_comparisons += sample.encoder_lease_comparisons;
    total.recording_commands += sample.recording_commands;
    total.native_commands += sample.native_commands;
    total.pipeline_bind_commands += sample.pipeline_bind_commands;
    total.strict_heap_set_bind_commands += sample.strict_heap_set_bind_commands;
    total.strict_heap_buffer_bind_commands +=
        sample.strict_heap_buffer_bind_commands;
    total.strict_heap_offset_commands +=
        sample.strict_heap_offset_commands;
}

fn double record_barriers(
    gpu::CommandAllocator* allocator,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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

fn void verify_commands_per_list(gpu::CommandAllocator* allocator) {
    uint[4] command_counts = { 1, 16, 256, 4096 };
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    foreach (count : command_counts) {
        gpu::CommandList commands = gpu::begin_commands(allocator)!!;
        gpu::reset_command_resolution_stats();
        for (uint i = 0; i < count; i++) {
            gpu::cmd_barrier(&commands, &barrier)!!;
        }
        gpu::CommandResolutionStats resolution =
            gpu::command_resolution_stats();
        assert(resolution.recording_commands == count,
            "commands-per-list evidence counts every recording call");
        assert(resolution.encoder_cell_computations == 0
            && resolution.encoder_lease_comparisons == 0,
            "commands-per-list evidence observes no retired frontend proof");
        assert(resolution.command_table_lookups == 0,
            "commands-per-list evidence uses the stable record directly");
        gpu::ExecutableCommandList executable =
            gpu::end_commands(&commands)!!;
        gpu::discard_executable_commands(&executable)!!;
    }
}

fn double record_hazard_barriers(
    gpu::CommandAllocator* allocator,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GpuSpan records,
    gpu::GpuSpan count,
    gpu::CommandResolutionStats* resolution,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    CommandBenchPolicy policy = command_bench_policy()!!;
    gpu::RuntimeDesc desc = {
        .contract_validation      = policy.contract,
        .track_resource_lifetimes = policy.track_lifetimes,
        .enable_vulkan_validation = policy.vulkan_layers,
        .application_name         = "gpu_command_record_bench",
    };
    gpu::Device device = create_bench_device(&desc, true)!!;
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
        .shader = shader,
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
    gpu::Queue graphics_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    gpu::CommandAllocatorDesc allocator_desc;
    gpu::GeneratedScratchDesc scratch;
    if (caps.generated_work) {
        scratch = {
            .pipeline                = pipeline,
            .kind                    = gpu::GeneratedWorkKind.DISPATCH,
            .max_commands_per_list   = GENERATED_RECORD_ITERATIONS,
            .preprocess_buffer_count = GENERATED_RECORD_ITERATIONS,
        };
        vk::MemoryRequirements requirements = vk::generated_scratch_requirements(
            state,
            &scratch,
        )!!;
        allocator_desc.max_generated_preprocess_buffers_per_list =
            GENERATED_RECORD_ITERATIONS;
        allocator_desc.generated_preprocess_bytes =
            (usz)requirements.size * GENERATED_RECORD_ITERATIONS;
    }
    gpu::internal::RecordingWorkCounters cold_before =
        vk::command_recording_work_counters(state);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        graphics_queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    if (caps.generated_work) {
        gpu::reserve_generated_scratch(&allocator, &scratch)!!;
    }

    gpu::CommandResolutionStats warmup_resolution;
    (void)record_barriers(&allocator, &warmup_resolution);
    (void)record_hazard_barriers(&allocator, &warmup_resolution);
    (void)record_indirect_dispatches(
        &allocator,
        pipeline,
        gpu::get_span_address(&device, buffer_span)!!,
        args,
        &warmup_resolution,
    );
    if (caps.generated_work) {
        (void)record_generated_dispatches(
            &allocator,
            pipeline,
            generated_records,
            generated_count,
            &warmup_resolution,
        );
    }
    verify_commands_per_list(&allocator);

    gpu::internal::RecordingWorkCounters warm_before =
        vk::command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters cold_work = recording_work_delta(
        warm_before,
        cold_before,
    );
    ulong registry_locks_before = gpu::internal::device_registry_lock_count();
    vk::CommandRecordingStats recording_before =
        vk::command_recording_stats(state);
    ulong pipelines_before = vk::pipeline_native_create_count(state);

    double[COMMAND_RECORD_REPETITIONS] barrier_results;
    double[COMMAND_RECORD_REPETITIONS] hazard_results;
    double[COMMAND_RECORD_REPETITIONS] indirect_results;
    double[COMMAND_RECORD_REPETITIONS] generated_results;
    gpu::CommandResolutionStats resolution;
    vk::reset_command_policy_counters(state);
    for (uint i = 0; i < COMMAND_RECORD_REPETITIONS; i++) {
        barrier_results[i] = record_barriers(&allocator, &resolution);
        hazard_results[i] = record_hazard_barriers(&allocator, &resolution);
        indirect_results[i] = record_indirect_dispatches(
            &allocator,
            pipeline,
            gpu::get_span_address(&device, buffer_span)!!,
            args,
            &resolution,
        );
        if (caps.generated_work) {
            generated_results[i] = record_generated_dispatches(
                &allocator,
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
        gpu::internal::device_registry_lock_count() - registry_locks_before;
    vk::CommandRecordingStats recording_after =
        vk::command_recording_stats(state);
    gpu::internal::RecordingWorkCounters warm_work = recording_work_delta(
        vk::command_recording_work_counters(state),
        warm_before,
    );
    vk::CommandPolicyCounters policy_counters =
        vk::command_policy_counters(state);
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
        "generated recording uses distinct reserved preprocess storage");
    assert(draw_compilations == 0,
        "recording does not compile pipelines");
    assert(warm_work.host_allocations == 0,
        "warmed recording performs no host allocation");
    assert(warm_work.command_buffer_allocations == 0,
        "warmed recording performs no command-buffer allocation");
    assert(warm_work.command_pool_creations == 0,
        "warmed recording performs no command-pool creation");
    assert(warm_work.command_buffer_frees == 0,
        "warmed recording performs no command-buffer free");
    assert(warm_work.image_view_creations == 0,
        "warmed recording performs no image-view creation");
    assert(warm_work.vma_allocations == 0,
        "warmed recording performs no VMA allocation");
    assert(warm_work.generated_scratch_misses == 0,
        "sufficient generated reservation has no scratch miss");
    assert(resolution.device_registry_resolutions == 0,
        "warm recording does not resolve the device registry");
    assert(resolution.retained_pin_operations == 0,
        "warm recording does not borrow retained pins");
    assert(resolution.lifecycle_vtable_dispatches == 0,
        "warm recording does not load the lifecycle vtable");
    assert(resolution.command_table_lookups == 0,
        "warm recording uses the stable record directly");
    assert(resolution.pipeline_table_lookups == 0,
        "warm recording does not resolve the pipeline table after bind");
    assert(resolution.pipeline_cache_lookups == 0,
        "warm recording does not resolve the pipeline cache after bind");
    assert(resolution.pipeline_bind_commands == 0
        && resolution.strict_heap_set_bind_commands == 0
        && resolution.strict_heap_buffer_bind_commands == 0
        && resolution.strict_heap_offset_commands == 0,
        "post-bind recording emits no pipeline or descriptor heap binds");
    assert(resolution.command_policy_selections == 0,
        "warm recording does not select a validation policy");
    assert(resolution.encoder_cell_computations == 0,
        "warm recording performs no frontend encoder-cell computation");
    assert(resolution.encoder_lease_comparisons == 0,
        "warm recording performs no packed-lease comparison");
    assert(resolution.recording_commands == expected_recording_commands,
        "resolution evidence counts every measured recording call");
    assert(resolution.native_commands >= resolution.recording_commands,
        "measured recording calls emit native command work");
    if (policy.contract == gpu::ContractValidation.FULL) {
        assert(policy_counters.semantic_checks > 0,
            "full validation performs semantic checks");
    } else {
        assert(policy_counters.semantic_checks == 0,
            "trusted lowering performs no detailed semantic checks");
    }
    if (policy.track_lifetimes) {
        assert(policy_counters.tracking_calls > 0
            && policy_counters.reference_allocations == 0
            && policy_counters.reference_increments > 0
            && policy_counters.reference_releases
                == policy_counters.reference_increments,
            "tracking reuses storage and retains/releases references exactly");
    } else {
        assert(policy_counters.tracking_calls == 0
            && policy_counters.reference_allocations == 0
            && policy_counters.reference_increments == 0
            && policy_counters.reference_releases == 0,
            "tracking-off recording performs no reference work");
    }
    if (caps.generated_work) {
        assert(preprocess_reuses
            >= GENERATED_RECORD_ITERATIONS * COMMAND_RECORD_REPETITIONS,
            "each measured execute uses distinct reserved preprocess storage");
    }

    io::printfn("expectation_version=%d", BENCHMARK_EXPECTATION_VERSION);
    io::printfn(
        "command_tokens: representation=direct recording_token_bytes=%d executable_token_bytes=%d record_bytes=%d cell_bytes=%d fixed_storage_bytes=%d commands_per_list=1,16,256,4096",
        gpu::CommandList::size,
        gpu::ExecutableCommandList::size,
        gpu::internal::CommandRecord::size,
        vk::CommandCell::size,
        state.commands.slots.len * vk::CommandCell::size,
    );
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
        "resolution: recording_commands=%d native_commands=%d pipeline_binds=%d descriptor_set_binds=%d descriptor_buffer_binds=%d descriptor_buffer_offsets=%d device_registry=%d retained_pins=%d lifecycle_vtable=%d command_table=%d pipeline_table=%d pipeline_cache=%d policy=%d encoder_cells=%d encoder_leases=%d",
        resolution.recording_commands,
        resolution.native_commands,
        resolution.pipeline_bind_commands,
        resolution.strict_heap_set_bind_commands,
        resolution.strict_heap_buffer_bind_commands,
        resolution.strict_heap_offset_commands,
        resolution.device_registry_resolutions,
        resolution.retained_pin_operations,
        resolution.lifecycle_vtable_dispatches,
        resolution.command_table_lookups,
        resolution.pipeline_table_lookups,
        resolution.pipeline_cache_lookups,
        resolution.command_policy_selections,
        resolution.encoder_cell_computations,
        resolution.encoder_lease_comparisons,
    );
    io::printfn(
        "validation policy=%s tracking=%s layers=%s semantic_checks=%d tracking_calls=%d reference_allocations=%d reference_increments=%d reference_releases=%d",
        policy.contract_name,
        policy.track_lifetimes ? "true" : "false",
        policy.vulkan_layers ? "true" : "false",
        policy_counters.semantic_checks,
        policy_counters.tracking_calls,
        policy_counters.reference_allocations,
        policy_counters.reference_increments,
        policy_counters.reference_releases,
    );
    io::printfn(
        "cold work: host_allocations=%d command_pool_creations=%d command_buffer_allocations=%d command_buffer_frees=%d command_buffer_resets=%d image_view_creations=%d vma_allocations=%d generated_scratch_misses=%d",
        cold_work.host_allocations,
        cold_work.command_pool_creations,
        cold_work.command_buffer_allocations,
        cold_work.command_buffer_frees,
        cold_work.command_buffer_resets,
        cold_work.image_view_creations,
        cold_work.vma_allocations,
        cold_work.generated_scratch_misses,
    );
    io::printfn(
        "warm work: host_allocations=%d command_pool_creations=%d command_buffer_allocations=%d command_buffer_frees=%d command_buffer_resets=%d image_view_creations=%d vma_allocations=%d generated_scratch_misses=%d",
        warm_work.host_allocations,
        warm_work.command_pool_creations,
        warm_work.command_buffer_allocations,
        warm_work.command_buffer_frees,
        warm_work.command_buffer_resets,
        warm_work.image_view_creations,
        warm_work.vma_allocations,
        warm_work.generated_scratch_misses,
    );
    io::printfn("generated preprocess: reuse_events=%d", preprocess_reuses);
    if (caps.generated_work) {
        gpu::release_generated_scratch(
            &allocator,
            pipeline,
            gpu::GeneratedWorkKind.DISPATCH,
        )!!;
    }
    return 0;
}
