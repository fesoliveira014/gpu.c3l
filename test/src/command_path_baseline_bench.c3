module gpu_bench;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::io;
import std::os::env;
import std::time;

const char[*] COMMAND_PATH_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const char[*] COMMAND_PATH_VERT_SPIRV =
    $embed("shaders/offscreen.vert.spv");
const char[*] COMMAND_PATH_FRAG_SPIRV =
    $embed("shaders/offscreen.frag.spv");

const uint COMMAND_PATH_ITERATIONS = 20_000;
const uint COMMAND_PATH_REPETITIONS = 5;
const uint COMMAND_PATH_OPERATION_LOOPS = COMMAND_PATH_REPETITIONS * 2;
const uint COMMAND_PATH_EQUIVALENCE_ELEMENTS = 64;
const uint COMMAND_PATH_RENDER_SIZE = 8;
const uint COMMAND_PATH_LIFECYCLE_REGISTRY_LOCKS_PER_SAMPLE = 0;
const ulong COMMAND_PATH_DISPATCH_NATIVE_CALLS = 2;
const ulong COMMAND_PATH_DRAW_NATIVE_CALLS = 2;
const ulong COMMAND_PATH_SINGLE_NATIVE_CALL = 1;

struct BenchSpan {
    gpu::GpuAllocation allocation;
    gpu::GpuSpan       span;
}

struct CommandPathPolicy {
    gpu::ContractValidation contract;
    bool                    vulkan_layers;
    ZString                 contract_name;
}

struct ComputeRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
    uint            _pad0;
    uint            _pad1;
    uint            _pad2;
}

$assert ComputeRoot::size == 32;
$assert $reflect(ComputeRoot.input_gpu).offset == 0;
$assert $reflect(ComputeRoot.output_gpu).offset == 8;
$assert $reflect(ComputeRoot.count).offset == 16;
$assert $reflect(ComputeRoot._pad0).offset == 20;
$assert $reflect(ComputeRoot._pad1).offset == 24;
$assert $reflect(ComputeRoot._pad2).offset == 28;

struct CommandPathWork {
    gpu::internal::RecordingWorkCounters recording;
    ulong                      shader_module_creations;
    ulong                      pipeline_creations;
}

struct OperationResult {
    double[COMMAND_PATH_REPETITIONS] direct;
    double[COMMAND_PATH_REPETITIONS] library;
    CommandPathWork                   work;
}

struct LifecycleResult {
    double[COMMAND_PATH_REPETITIONS] elapsed;
    CommandPathWork                  work;
}

struct LifecycleSample {
    double          elapsed;
    CommandPathWork work;
}

struct EquivalenceResult {
    uint expected;
    uint direct;
    uint library;
}

fn bool? command_path_flag(ZString variable) {
    if (try value = env::tget_var(variable.str_view())) {
        if (value == "true") return true;
        if (value == "false") return false;
        return gpu::INVALID_ARGUMENT~;
    }
    return false;
}

fn CommandPathPolicy? command_path_policy() {
    String contract_value = "trusted";
    if (try value = env::tget_var("GPU_C3L_BENCH_CONTRACT")) {
        contract_value = value;
    }
    gpu::ContractValidation contract;
    ZString contract_name;
    if (contract_value == "trusted") {
        contract = gpu::ContractValidation.TRUSTED;
        contract_name = "trusted";
    } else if (contract_value == "full") {
        contract = gpu::ContractValidation.FULL;
        contract_name = "full";
    } else {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .contract      = contract,
        .vulkan_layers = command_path_flag("GPU_C3L_BENCH_LAYERS")!,
        .contract_name = contract_name,
    };
}

fn BenchSpan? allocate_bench_span(
    gpu::Device* device,
    usz size,
    gpu::MemoryClass memory_class,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = 16,
        .memory_class = memory_class,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    return {
        .allocation = allocation,
        .span       = gpu::get_allocation_span(device, allocation)!,
    };
}

fn void? free_bench_span(gpu::Device* device, BenchSpan* owned) {
    if (owned == null || !owned.allocation.is_valid()) return;
    gpu::free_allocation(device, &owned.allocation)!;
    owned.span = {};
}

fn gpu::PipelineHandle create_compute_pipeline(gpu::Device* device) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = COMMAND_PATH_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = gpu::prepare_shader_code(&shader_desc)!!,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::PipelineHandle create_graphics_pipeline(gpu::Device* device) {
    gpu::ShaderDesc vertex_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = COMMAND_PATH_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = COMMAND_PATH_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = gpu::prepare_shader_code(&vertex_desc)!!,
        .fragment_shader = gpu::prepare_shader_code(&fragment_desc)!!,
        .color_formats   = color_formats[..],
    };
    return gpu::create_graphics_pipeline(device, &pipeline_desc)!!;
}

fn void add_work(
    CommandPathWork* total,
    CommandPathWork delta,
) {
    total.recording.host_allocations += delta.recording.host_allocations;
    total.recording.command_pool_creations +=
        delta.recording.command_pool_creations;
    total.recording.command_buffer_allocations +=
        delta.recording.command_buffer_allocations;
    total.recording.command_buffer_frees += delta.recording.command_buffer_frees;
    total.recording.command_buffer_resets += delta.recording.command_buffer_resets;
    total.recording.image_view_creations += delta.recording.image_view_creations;
    total.recording.vma_allocations += delta.recording.vma_allocations;
    total.recording.registry_lock_acquisitions +=
        delta.recording.registry_lock_acquisitions;
    total.shader_module_creations += delta.shader_module_creations;
    total.pipeline_creations += delta.pipeline_creations;
}

fn CommandPathWork work_snapshot(vk::VkDeviceState* state) {
    return {
        .recording = vk::command_recording_work_counters(state),
        .shader_module_creations =
            vk::pipeline_shader_module_create_count(state),
        .pipeline_creations = vk::pipeline_native_create_count(state),
    };
}

fn CommandPathWork measured_work(
    vk::VkDeviceState* state,
    CommandPathWork before,
) {
    CommandPathWork after = work_snapshot(state);
    return {
        .recording = vk::command_recording_work_delta(
            after.recording,
            before.recording,
        ),
        .shader_module_creations =
            after.shader_module_creations - before.shader_module_creations,
        .pipeline_creations =
            after.pipeline_creations - before.pipeline_creations,
    };
}

fn void require_zero_operation_work(CommandPathWork work) {
    assert(work.recording.host_allocations == 0);
    assert(work.recording.command_pool_creations == 0);
    assert(work.recording.command_buffer_allocations == 0);
    assert(work.recording.command_buffer_frees == 0);
    assert(work.recording.command_buffer_resets == 0);
    assert(work.recording.image_view_creations == 0);
    assert(work.recording.vma_allocations == 0);
    assert(work.recording.registry_lock_acquisitions == 0);
    assert(work.shader_module_creations == 0);
    assert(work.pipeline_creations == 0);
}

fn double elapsed_per_operation(std::time::Clock start) {
    return (double)(long)start.to_now() / (double)COMMAND_PATH_ITERATIONS;
}

fn void finish_recording(gpu::CommandList* commands) {
    gpu::ExecutableCommandList executable = gpu::end_commands(commands)!!;
    gpu::discard_executable_commands(&executable)!!;
}

fn void prepare_compute_commands(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::CommandList* direct,
    gpu::CommandList* library,
) {
    *direct = gpu::begin_commands(allocator)!!;
    *library = gpu::begin_commands(allocator)!!;
    gpu::cmd_bind_pipeline(direct, pipeline)!!;
    gpu::cmd_bind_pipeline(library, pipeline)!!;
}

fn void prepare_render_commands(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
    gpu::CommandList* direct,
    gpu::CommandList* library,
) {
    *direct = gpu::begin_commands(allocator)!!;
    *library = gpu::begin_commands(allocator)!!;
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics.color = { .targets = color_targets[..] };
    gpu::cmd_bind_pipeline(direct, pipeline)!!;
    gpu::cmd_bind_pipeline(library, pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(direct, pass, &graphics)!!;
    gpu::cmd_begin_render_pass_with_state(library, pass, &graphics)!!;
}

fn OperationResult measure_dispatch(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    vk::VkDeviceState* state,
) {
    OperationResult result;
    gpu::Vec3u groups = { 1, 1, 1 };
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        gpu::CommandList direct;
        gpu::CommandList library;
        prepare_compute_commands(
            allocator: allocator,
            pipeline:  pipeline,
            direct:    &direct,
            library:   &library,
        );
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        ulong warm_calls;
        vk::command_path_direct_dispatch(
            context:      &native,
            root:         (gpu::GpuAddress)0,
            groups:       groups,
            native_calls: &warm_calls,
        );
        gpu::cmd_dispatch(&library, (gpu::GpuAddress)0, groups)!!;
        assert(warm_calls == COMMAND_PATH_DISPATCH_NATIVE_CALLS);
        ulong direct_calls;
        ulong library_calls;
        if (repetition % 2 == 0) {
            CommandPathWork before = work_snapshot(state);
            std::time::Clock start = std::time::clock::now();
            for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
                vk::command_path_direct_dispatch(
                    context:      &native,
                    root:         (gpu::GpuAddress)0,
                    groups:       groups,
                    native_calls: &direct_calls,
                );
            }
            result.direct[repetition] = elapsed_per_operation(start);
            CommandPathWork work = measured_work(state, before);
            require_zero_operation_work(work);
            add_work(&result.work, work);

            before = work_snapshot(state);
            start = std::time::clock::now();
            for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
                gpu::cmd_dispatch(&library, (gpu::GpuAddress)0, groups)!!;
                library_calls += COMMAND_PATH_DISPATCH_NATIVE_CALLS;
            }
            result.library[repetition] = elapsed_per_operation(start);
            work = measured_work(state, before);
            require_zero_operation_work(work);
            add_work(&result.work, work);
        } else {
            CommandPathWork before = work_snapshot(state);
            std::time::Clock start = std::time::clock::now();
            for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
                gpu::cmd_dispatch(&library, (gpu::GpuAddress)0, groups)!!;
                library_calls += COMMAND_PATH_DISPATCH_NATIVE_CALLS;
            }
            result.library[repetition] = elapsed_per_operation(start);
            CommandPathWork work = measured_work(state, before);
            require_zero_operation_work(work);
            add_work(&result.work, work);

            before = work_snapshot(state);
            start = std::time::clock::now();
            for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
                vk::command_path_direct_dispatch(
                    context:      &native,
                    root:         (gpu::GpuAddress)0,
                    groups:       groups,
                    native_calls: &direct_calls,
                );
            }
            result.direct[repetition] = elapsed_per_operation(start);
            work = measured_work(state, before);
            require_zero_operation_work(work);
            add_work(&result.work, work);
        }
        ulong expected = (ulong)COMMAND_PATH_ITERATIONS
            * COMMAND_PATH_DISPATCH_NATIVE_CALLS;
        assert(direct_calls == expected && library_calls == expected);
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double measure_direct_draw_loop(
    vk::CommandPathNativeContext* native,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_draw(
            context:      native,
            vertex_root:  (gpu::GpuAddress)1,
            fragment_root: (gpu::GpuAddress)1,
            vertex_count: 3,
            instance_count: 1,
            native_calls: native_calls,
        );
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn double measure_library_draw_loop(
    gpu::CommandList* commands,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_draw(
            commands:       commands,
            vertex_root:    (gpu::GpuAddress)1,
            fragment_root:  (gpu::GpuAddress)1,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        *native_calls += COMMAND_PATH_DRAW_NATIVE_CALLS;
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn OperationResult measure_draw(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
    vk::VkDeviceState* state,
) {
    OperationResult result;
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        gpu::CommandList direct;
        gpu::CommandList library;
        prepare_render_commands(
            allocator: allocator,
            pipeline:  pipeline,
            pass:      pass,
            direct:    &direct,
            library:   &library,
        );
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        ulong warm_calls;
        vk::command_path_direct_draw(
            context:       &native,
            vertex_root:   (gpu::GpuAddress)1,
            fragment_root: (gpu::GpuAddress)1,
            vertex_count:  3,
            instance_count: 1,
            native_calls:  &warm_calls,
        );
        gpu::cmd_draw(
            commands:       &library,
            vertex_root:    (gpu::GpuAddress)1,
            fragment_root:  (gpu::GpuAddress)1,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        assert(warm_calls == COMMAND_PATH_DRAW_NATIVE_CALLS);
        ulong direct_calls;
        ulong library_calls;
        if (repetition % 2 == 0) {
            result.direct[repetition] = measure_direct_draw_loop(
                native:       &native,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
            result.library[repetition] = measure_library_draw_loop(
                commands:     &library,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
        } else {
            result.library[repetition] = measure_library_draw_loop(
                commands:     &library,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
            result.direct[repetition] = measure_direct_draw_loop(
                native:       &native,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
        }
        ulong expected = (ulong)COMMAND_PATH_ITERATIONS
            * COMMAND_PATH_DRAW_NATIVE_CALLS;
        assert(direct_calls == expected && library_calls == expected);
        gpu::cmd_end_render_pass(&direct)!!;
        gpu::cmd_end_render_pass(&library)!!;
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double measure_direct_barrier_loop(
    vk::CommandPathNativeContext* native,
    gpu::Barrier* barrier,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_barrier(native, barrier, native_calls);
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn double measure_library_barrier_loop(
    gpu::CommandList* commands,
    gpu::Barrier* barrier,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_barrier(commands, barrier)!!;
        *native_calls += COMMAND_PATH_SINGLE_NATIVE_CALL;
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn OperationResult measure_barrier(
    gpu::CommandAllocator* allocator,
    vk::VkDeviceState* state,
) {
    OperationResult result;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        gpu::CommandList direct = gpu::begin_commands(allocator)!!;
        gpu::CommandList library = gpu::begin_commands(allocator)!!;
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        ulong warm_calls;
        vk::command_path_direct_barrier(&native, &barrier, &warm_calls);
        gpu::cmd_barrier(&library, &barrier)!!;
        assert(warm_calls == COMMAND_PATH_SINGLE_NATIVE_CALL);
        ulong direct_calls;
        ulong library_calls;
        if (repetition % 2 == 0) {
            result.direct[repetition] = measure_direct_barrier_loop(
                native:       &native,
                barrier:      &barrier,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
            result.library[repetition] = measure_library_barrier_loop(
                commands:     &library,
                barrier:      &barrier,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
        } else {
            result.library[repetition] = measure_library_barrier_loop(
                commands:     &library,
                barrier:      &barrier,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
            result.direct[repetition] = measure_direct_barrier_loop(
                native:       &native,
                barrier:      &barrier,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
        }
        ulong expected = (ulong)COMMAND_PATH_ITERATIONS;
        assert(direct_calls == expected && library_calls == expected);
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double measure_direct_viewport_loop(
    vk::CommandPathNativeContext* native,
    gpu::Viewport* viewport,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_viewport(native, viewport, native_calls);
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn double measure_library_viewport_loop(
    gpu::CommandList* commands,
    gpu::Viewport* viewport,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_set_viewport(commands, viewport)!!;
        *native_calls += COMMAND_PATH_SINGLE_NATIVE_CALL;
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn OperationResult measure_viewport(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::RenderPassDesc* pass,
    vk::VkDeviceState* state,
) {
    OperationResult result;
    gpu::Viewport viewport = {
        .width     = COMMAND_PATH_RENDER_SIZE,
        .height    = COMMAND_PATH_RENDER_SIZE,
        .max_depth = 1.0f,
    };
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        gpu::CommandList direct;
        gpu::CommandList library;
        prepare_render_commands(
            allocator: allocator,
            pipeline:  pipeline,
            pass:      pass,
            direct:    &direct,
            library:   &library,
        );
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        ulong warm_calls;
        vk::command_path_direct_viewport(&native, &viewport, &warm_calls);
        gpu::cmd_set_viewport(&library, &viewport)!!;
        assert(warm_calls == COMMAND_PATH_SINGLE_NATIVE_CALL);
        ulong direct_calls;
        ulong library_calls;
        if (repetition % 2 == 0) {
            result.direct[repetition] = measure_direct_viewport_loop(
                native:       &native,
                viewport:     &viewport,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
            result.library[repetition] = measure_library_viewport_loop(
                commands:     &library,
                viewport:     &viewport,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
        } else {
            result.library[repetition] = measure_library_viewport_loop(
                commands:     &library,
                viewport:     &viewport,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
            result.direct[repetition] = measure_direct_viewport_loop(
                native:       &native,
                viewport:     &viewport,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
        }
        ulong expected = (ulong)COMMAND_PATH_ITERATIONS;
        assert(direct_calls == expected && library_calls == expected);
        gpu::cmd_end_render_pass(&direct)!!;
        gpu::cmd_end_render_pass(&library)!!;
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double measure_direct_copy_loop(
    vk::CommandPathNativeContext* native,
    vk::CommandPathNativeCopy* copy,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_copy_buffer(native, copy, native_calls);
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn double measure_library_copy_loop(
    gpu::CommandList* commands,
    gpu::BufferCopyDesc* copy,
    vk::VkDeviceState* state,
    CommandPathWork* total,
    ulong* native_calls,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_copy_buffer(commands, copy)!!;
        *native_calls += COMMAND_PATH_SINGLE_NATIVE_CALL;
    }
    double elapsed = elapsed_per_operation(start);
    CommandPathWork work = measured_work(state, before);
    require_zero_operation_work(work);
    add_work(total, work);
    return elapsed;
}

fn OperationResult measure_copy(
    gpu::Device* device,
    gpu::CommandAllocator* allocator,
    gpu::BufferCopyDesc* copy,
    vk::VkDeviceState* state,
) {
    OperationResult result;
    vk::CommandPathNativeCopy native_copy = vk::command_path_native_copy(
        device,
        copy.src,
        copy.dst,
    )!!;
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        gpu::CommandList direct = gpu::begin_commands(allocator)!!;
        gpu::CommandList library = gpu::begin_commands(allocator)!!;
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        ulong warm_calls;
        vk::command_path_direct_copy_buffer(
            &native,
            &native_copy,
            &warm_calls,
        );
        gpu::cmd_copy_buffer(&library, copy)!!;
        assert(warm_calls == COMMAND_PATH_SINGLE_NATIVE_CALL);
        ulong direct_calls;
        ulong library_calls;
        if (repetition % 2 == 0) {
            result.direct[repetition] = measure_direct_copy_loop(
                native:       &native,
                copy:         &native_copy,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
            result.library[repetition] = measure_library_copy_loop(
                commands:     &library,
                copy:         copy,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
        } else {
            result.library[repetition] = measure_library_copy_loop(
                commands:     &library,
                copy:         copy,
                state:        state,
                total:        &result.work,
                native_calls: &library_calls,
            );
            result.direct[repetition] = measure_direct_copy_loop(
                native:       &native,
                copy:         &native_copy,
                state:        state,
                total:        &result.work,
                native_calls: &direct_calls,
            );
        }
        ulong expected = (ulong)COMMAND_PATH_ITERATIONS;
        assert(direct_calls == expected && library_calls == expected);
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double[COMMAND_PATH_REPETITIONS] sorted_samples(
    double[COMMAND_PATH_REPETITIONS] values,
) {
    double[COMMAND_PATH_REPETITIONS] sorted = values;
    for (uint i = 1; i < COMMAND_PATH_REPETITIONS; i++) {
        double key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    return sorted;
}

fn double sample_median(double[COMMAND_PATH_REPETITIONS] values) {
    double[COMMAND_PATH_REPETITIONS] sorted = sorted_samples(values);
    return sorted[COMMAND_PATH_REPETITIONS / 2];
}

fn void print_operation(
    ZString operation,
    ulong native_calls_per_iteration,
    OperationResult* result,
) {
    double[COMMAND_PATH_REPETITIONS] direct = sorted_samples(result.direct);
    double[COMMAND_PATH_REPETITIONS] library = sorted_samples(result.library);
    double ratio = library[COMMAND_PATH_REPETITIONS / 2]
        / direct[COMMAND_PATH_REPETITIONS / 2];
    io::printfn(
        "command_path_vk operation=%s iterations=%d repetitions=%d native_calls_per_iteration=%d direct_min_ns=%.6f direct_median_ns=%.6f direct_max_ns=%.6f public_min_ns=%.6f public_median_ns=%.6f public_max_ns=%.6f ratio=%.6f",
        operation,
        COMMAND_PATH_ITERATIONS,
        COMMAND_PATH_REPETITIONS,
        native_calls_per_iteration,
        direct[0],
        direct[COMMAND_PATH_REPETITIONS / 2],
        direct[COMMAND_PATH_REPETITIONS - 1],
        library[0],
        library[COMMAND_PATH_REPETITIONS / 2],
        library[COMMAND_PATH_REPETITIONS - 1],
        ratio,
    );
}

fn void print_operation_work(
    ZString operation,
    CommandPathWork work,
) {
    io::printfn(
        "command_path_vk_work operation=%s loops=%d host_allocations=%d command_pool_creations=%d command_buffer_allocations=%d command_buffer_frees=%d command_buffer_resets=%d image_view_creations=%d vma_allocations=%d registry_lock_acquisitions=%d shader_module_creations=%d pipeline_creations=%d status=pass",
        operation,
        COMMAND_PATH_OPERATION_LOOPS,
        work.recording.host_allocations,
        work.recording.command_pool_creations,
        work.recording.command_buffer_allocations,
        work.recording.command_buffer_frees,
        work.recording.command_buffer_resets,
        work.recording.image_view_creations,
        work.recording.vma_allocations,
        work.recording.registry_lock_acquisitions,
        work.shader_module_creations,
        work.pipeline_creations,
    );
}

fn gpu::CompletionPoint submit_commands(
    gpu::Queue queue,
    gpu::ExecutableCommandList[] lists,
) {
    gpu::SubmitDesc submit = { .command_lists = lists };
    return gpu::submit(queue, &submit)!!;
}

fn EquivalenceResult execute_dispatch_equivalence(
    gpu::Device* device,
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
) {
    const usz DATA_BYTES = COMMAND_PATH_EQUIVALENCE_ELEMENTS * float::size;
    BenchSpan input = allocate_bench_span(
        device:       device,
        size:         DATA_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "command_path_dispatch_input",
    )!!;
    defer free_bench_span(device, &input)!!;
    BenchSpan direct_output = allocate_bench_span(
        device:       device,
        size:         DATA_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "command_path_dispatch_direct_output",
    )!!;
    defer free_bench_span(device, &direct_output)!!;
    BenchSpan public_output = allocate_bench_span(
        device:       device,
        size:         DATA_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "command_path_dispatch_public_output",
    )!!;
    defer free_bench_span(device, &public_output)!!;
    BenchSpan direct_root = allocate_bench_span(
        device:       device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "command_path_dispatch_direct_root",
    )!!;
    defer free_bench_span(device, &direct_root)!!;
    BenchSpan public_root = allocate_bench_span(
        device:       device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "command_path_dispatch_public_root",
    )!!;
    defer free_bench_span(device, &public_root)!!;

    float* input_values = (float*)gpu::get_span_mapping(device, input.span)!!.ptr;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        input_values[i] = (float)(i + 1);
    }
    ComputeRoot* direct_data =
        (ComputeRoot*)gpu::get_span_mapping(device, direct_root.span)!!.ptr;
    *direct_data = {
        .input_gpu  = gpu::get_span_address(device, input.span)!!,
        .output_gpu = gpu::get_span_address(device, direct_output.span)!!,
        .count      = COMMAND_PATH_EQUIVALENCE_ELEMENTS,
    };
    ComputeRoot* public_data =
        (ComputeRoot*)gpu::get_span_mapping(device, public_root.span)!!.ptr;
    *public_data = {
        .input_gpu  = gpu::get_span_address(device, input.span)!!,
        .output_gpu = gpu::get_span_address(device, public_output.span)!!,
        .count      = COMMAND_PATH_EQUIVALENCE_ELEMENTS,
    };
    gpu::flush_mapped_span(device, input.span)!!;
    gpu::flush_mapped_span(device, direct_root.span)!!;
    gpu::flush_mapped_span(device, public_root.span)!!;

    gpu::CommandList direct;
    gpu::CommandList library;
    prepare_compute_commands(
        allocator: allocator,
        pipeline:  pipeline,
        direct:    &direct,
        library:   &library,
    );
    gpu::cmd_fill_buffer(&direct, direct_output.span, 0x11111111)!!;
    gpu::cmd_fill_buffer(&library, public_output.span, 0x22222222)!!;
    gpu::Barrier to_compute = {
        .before = { .transfer = true },
        .after  = { .compute = true },
    };
    gpu::cmd_barrier(&direct, &to_compute)!!;
    gpu::cmd_barrier(&library, &to_compute)!!;
    vk::CommandPathNativeContext native =
        vk::command_path_native_context(&direct)!!;
    ulong native_calls;
    gpu::Vec3u groups = { 1, 1, 1 };
    vk::command_path_direct_dispatch(
        context:      &native,
        root:         gpu::get_span_address(device, direct_root.span)!!,
        groups:       groups,
        native_calls: &native_calls,
    );
    gpu::cmd_dispatch(
        &library,
        gpu::get_span_address(device, public_root.span)!!,
        groups,
    )!!;
    assert(native_calls == COMMAND_PATH_DISPATCH_NATIVE_CALLS);
    gpu::Barrier to_host = {
        .before = { .compute = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&direct, &to_host)!!;
    gpu::cmd_barrier(&library, &to_host)!!;
    gpu::ExecutableCommandList[2] lists = {
        gpu::end_commands(&direct)!!,
        gpu::end_commands(&library)!!,
    };
    gpu::CompletionPoint completion = submit_commands(queue, lists[..]);
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(device, direct_output.span)!!;
    gpu::invalidate_mapped_span(device, public_output.span)!!;

    float* direct_values =
        (float*)gpu::get_span_mapping(device, direct_output.span)!!.ptr;
    float* public_values =
        (float*)gpu::get_span_mapping(device, public_output.span)!!.ptr;
    EquivalenceResult result;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        float expected = (float)(i + 1) * 2.0f;
        assert(direct_values[i] == expected);
        assert(public_values[i] == expected);
        assert(direct_values[i] == public_values[i]);
        result.expected += (uint)expected;
        result.direct += (uint)direct_values[i];
        result.library += (uint)public_values[i];
    }
    assert(result.expected != 0);
    assert(result.direct == result.expected);
    assert(result.library == result.expected);
    return result;
}

fn void preseed_copy_outputs(
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
    gpu::GpuSpan direct_output,
    gpu::GpuSpan public_output,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_fill_buffer(&commands, direct_output, 0x33333333)!!;
    gpu::cmd_fill_buffer(&commands, public_output, 0x44444444)!!;
    gpu::ExecutableCommandList[1] lists = {
        gpu::end_commands(&commands)!!,
    };
    gpu::CompletionPoint completion = submit_commands(queue, lists[..]);
    gpu::wait_completion(completion)!!;
}

fn EquivalenceResult execute_copy_equivalence(
    gpu::Device* device,
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
) {
    const usz COPY_BYTES = COMMAND_PATH_EQUIVALENCE_ELEMENTS;
    BenchSpan source = allocate_bench_span(
        device:       device,
        size:         COPY_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "command_path_copy_source",
    )!!;
    defer free_bench_span(device, &source)!!;
    BenchSpan direct_output = allocate_bench_span(
        device:       device,
        size:         COPY_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "command_path_copy_direct_output",
    )!!;
    defer free_bench_span(device, &direct_output)!!;
    BenchSpan public_output = allocate_bench_span(
        device:       device,
        size:         COPY_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "command_path_copy_public_output",
    )!!;
    defer free_bench_span(device, &public_output)!!;
    char* source_values = (char*)gpu::get_span_mapping(device, source.span)!!.ptr;
    EquivalenceResult result;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        source_values[i] = (char)(i + 1);
        result.expected += i + 1;
    }
    gpu::flush_mapped_span(device, source.span)!!;
    preseed_copy_outputs(
        queue:         queue,
        allocator:     allocator,
        direct_output: direct_output.span,
        public_output: public_output.span,
    );

    gpu::CommandList direct = gpu::begin_commands(allocator)!!;
    gpu::CommandList library = gpu::begin_commands(allocator)!!;
    vk::CommandPathNativeContext native =
        vk::command_path_native_context(&direct)!!;
    vk::CommandPathNativeCopy direct_copy = vk::command_path_native_copy(
        device,
        source.span,
        direct_output.span,
    )!!;
    ulong native_calls;
    vk::command_path_direct_copy_buffer(&native, &direct_copy, &native_calls);
    gpu::BufferCopyDesc public_copy = {
        .src = source.span,
        .dst = public_output.span,
    };
    gpu::cmd_copy_buffer(&library, &public_copy)!!;
    assert(native_calls == COMMAND_PATH_SINGLE_NATIVE_CALL);
    gpu::Barrier to_host = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&direct, &to_host)!!;
    gpu::cmd_barrier(&library, &to_host)!!;
    gpu::ExecutableCommandList[2] lists = {
        gpu::end_commands(&direct)!!,
        gpu::end_commands(&library)!!,
    };
    gpu::CompletionPoint completion = submit_commands(queue, lists[..]);
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(device, direct_output.span)!!;
    gpu::invalidate_mapped_span(device, public_output.span)!!;
    char* direct_values =
        (char*)gpu::get_span_mapping(device, direct_output.span)!!.ptr;
    char* public_values =
        (char*)gpu::get_span_mapping(device, public_output.span)!!.ptr;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        assert(direct_values[i] == source_values[i]);
        assert(public_values[i] == source_values[i]);
        assert(direct_values[i] == public_values[i]);
        result.direct += (uint)direct_values[i];
        result.library += (uint)public_values[i];
    }
    assert(result.direct == result.expected);
    assert(result.library == result.expected);
    return result;
}

fn LifecycleSample measure_lifecycle_sample(
    gpu::CommandAllocator* allocator,
    gpu::Queue queue,
    gpu::PipelineHandle pipeline,
    vk::VkDeviceState* state,
    uint command_count,
) {
    CommandPathWork before = work_snapshot(state);
    std::time::Clock start = std::time::clock::now();
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    for (uint i = 0; i < command_count; i++) {
        gpu::cmd_dispatch(
            &commands,
            (gpu::GpuAddress)0,
            (gpu::Vec3u){ 1, 1, 1 },
        )!!;
    }
    gpu::ExecutableCommandList[1] lists = {
        gpu::end_commands(&commands)!!,
    };
    gpu::CompletionPoint completion = submit_commands(queue, lists[..]);
    gpu::wait_completion(completion)!!;
    double elapsed = (double)(long)start.to_now();
    return {
        .elapsed = elapsed,
        .work    = measured_work(state, before),
    };
}

fn void require_lifecycle_sample_work(CommandPathWork work) {
    assert(work.recording.host_allocations == 0);
    assert(work.recording.command_pool_creations == 0);
    assert(work.recording.command_buffer_allocations == 0);
    assert(work.recording.command_buffer_frees == 0);
    assert(work.recording.command_buffer_resets == 1);
    assert(work.recording.image_view_creations == 0);
    assert(work.recording.vma_allocations == 0);
    assert(work.recording.registry_lock_acquisitions
        == COMMAND_PATH_LIFECYCLE_REGISTRY_LOCKS_PER_SAMPLE);
    assert(work.shader_module_creations == 0);
    assert(work.pipeline_creations == 0);
}

fn void add_lifecycle_sample(
    LifecycleResult* result,
    uint repetition,
    LifecycleSample sample,
) {
    require_lifecycle_sample_work(sample.work);
    result.elapsed[repetition] = sample.elapsed;
    add_work(&result.work, sample.work);
}

fn void print_lifecycle(
    uint commands,
    LifecycleResult* result,
    LifecycleResult* zero,
) {
    double[COMMAND_PATH_REPETITIONS] sorted = sorted_samples(result.elapsed);
    double paired_delta_median;
    double incremental;
    if (commands != 0) {
        double[COMMAND_PATH_REPETITIONS] paired;
        for (uint i = 0; i < COMMAND_PATH_REPETITIONS; i++) {
            paired[i] = result.elapsed[i] - zero.elapsed[i];
        }
        paired_delta_median = sample_median(paired);
        incremental = paired_delta_median / (double)commands;
    }
    io::printfn(
        "command_path_vk_lifecycle commands=%d repetitions=%d min_ns=%.3f median_ns=%.3f max_ns=%.3f paired_delta_median_ns=%.3f incremental_ns_per_command=%.3f",
        commands,
        COMMAND_PATH_REPETITIONS,
        sorted[0],
        sorted[COMMAND_PATH_REPETITIONS / 2],
        sorted[COMMAND_PATH_REPETITIONS - 1],
        paired_delta_median,
        incremental,
    );
}

fn void print_lifecycle_work(
    uint commands,
    CommandPathWork work,
) {
    io::printfn(
        "command_path_vk_lifecycle_work commands=%d samples=%d host_allocations=%d command_pool_creations=%d command_buffer_allocations=%d command_buffer_frees=%d command_buffer_resets=%d image_view_creations=%d vma_allocations=%d registry_lock_acquisitions=%d shader_module_creations=%d pipeline_creations=%d status=pass",
        commands,
        COMMAND_PATH_REPETITIONS,
        work.recording.host_allocations,
        work.recording.command_pool_creations,
        work.recording.command_buffer_allocations,
        work.recording.command_buffer_frees,
        work.recording.command_buffer_resets,
        work.recording.image_view_creations,
        work.recording.vma_allocations,
        work.recording.registry_lock_acquisitions,
        work.shader_module_creations,
        work.pipeline_creations,
    );
}

fn int main() {
    CommandPathPolicy policy = command_path_policy()!!;
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = policy.contract,
        .enable_vulkan_validation = policy.vulkan_layers,
        .application_name         = "gpu_command_path_baseline",
    };
    gpu::Device device =
        create_bench_device(&runtime_desc, explicit_policy: true)!!;
    defer destroy_bench_device(&device)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    assert(state.validation_policy.contract == policy.contract);
    assert(state.validation_policy.vulkan_layers == policy.vulkan_layers);

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::PipelineHandle compute_pipeline = create_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, compute_pipeline)!!;
    gpu::PipelineHandle graphics_pipeline = create_graphics_pipeline(&device);
    defer gpu::destroy_pipeline(&device, graphics_pipeline)!!;

    gpu::TextureDesc target_desc = {
        .access = { .graphics },
        .width  = COMMAND_PATH_RENDER_SIZE,
        .height = COMMAND_PATH_RENDER_SIZE,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = target };
    gpu::AttachmentViewHandle view =
        gpu::create_attachment_view(&device, &view_desc)!!;
    defer gpu::destroy_attachment_view(&device, view)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view     = view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = COMMAND_PATH_RENDER_SIZE,
        .height = COMMAND_PATH_RENDER_SIZE,
    };

    BenchSpan copy_source = allocate_bench_span(
        device:       &device,
        size:         COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        debug_name:   "command_path_timing_copy_source",
    )!!;
    defer free_bench_span(&device, &copy_source)!!;
    BenchSpan copy_destination = allocate_bench_span(
        device:       &device,
        size:         COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        memory_class: gpu::MemoryClass.CPU_READ,
        debug_name:   "command_path_timing_copy_destination",
    )!!;
    defer free_bench_span(&device, &copy_destination)!!;
    gpu::BufferCopyDesc timing_copy = {
        .src = copy_source.span,
        .dst = copy_destination.span,
    };

    vk::reset_command_recording_work_counters(state);
    OperationResult dispatch = measure_dispatch(
        &allocator,
        compute_pipeline,
        state,
    );
    OperationResult draw = measure_draw(
        allocator: &allocator,
        pipeline:  graphics_pipeline,
        pass:      &pass,
        state:     state,
    );
    OperationResult barrier = measure_barrier(&allocator, state);
    OperationResult viewport = measure_viewport(
        allocator: &allocator,
        pipeline:  graphics_pipeline,
        pass:      &pass,
        state:     state,
    );
    OperationResult copy = measure_copy(
        device:    &device,
        allocator: &allocator,
        copy:      &timing_copy,
        state:     state,
    );

    EquivalenceResult dispatch_equivalence = execute_dispatch_equivalence(
        device:    &device,
        queue:     queue,
        allocator: &allocator,
        pipeline:  compute_pipeline,
    );
    EquivalenceResult copy_equivalence = execute_copy_equivalence(
        &device,
        queue,
        &allocator,
    );

    (void)measure_lifecycle_sample(
        allocator:     &allocator,
        queue:         queue,
        pipeline:      compute_pipeline,
        state:         state,
        command_count: 0,
    );
    LifecycleResult zero;
    LifecycleResult one;
    LifecycleResult sixteen;
    LifecycleResult two_fifty_six;
    for (uint repetition = 0; repetition < COMMAND_PATH_REPETITIONS; repetition++) {
        add_lifecycle_sample(
            &zero,
            repetition,
            measure_lifecycle_sample(
                allocator:     &allocator,
                queue:         queue,
                pipeline:      compute_pipeline,
                state:         state,
                command_count: 0,
            ),
        );
        add_lifecycle_sample(
            &one,
            repetition,
            measure_lifecycle_sample(
                allocator:     &allocator,
                queue:         queue,
                pipeline:      compute_pipeline,
                state:         state,
                command_count: 1,
            ),
        );
        add_lifecycle_sample(
            &sixteen,
            repetition,
            measure_lifecycle_sample(
                allocator:     &allocator,
                queue:         queue,
                pipeline:      compute_pipeline,
                state:         state,
                command_count: 16,
            ),
        );
        add_lifecycle_sample(
            &two_fifty_six,
            repetition,
            measure_lifecycle_sample(
                allocator:     &allocator,
                queue:         queue,
                pipeline:      compute_pipeline,
                state:         state,
                command_count: 256,
            ),
        );
    }

    io::printfn(
        "command_path_vk_policy validation=%s layers=%s resolution_stats=false recording_work_stats=true",
        policy.contract_name,
        policy.vulkan_layers ? "true" : "false",
    );
    print_operation("dispatch", COMMAND_PATH_DISPATCH_NATIVE_CALLS, &dispatch);
    print_operation("draw", COMMAND_PATH_DRAW_NATIVE_CALLS, &draw);
    print_operation("barrier", COMMAND_PATH_SINGLE_NATIVE_CALL, &barrier);
    print_operation("viewport", COMMAND_PATH_SINGLE_NATIVE_CALL, &viewport);
    print_operation("copy_buffer", COMMAND_PATH_SINGLE_NATIVE_CALL, &copy);
    print_operation_work("dispatch", dispatch.work);
    print_operation_work("draw", draw.work);
    print_operation_work("barrier", barrier.work);
    print_operation_work("viewport", viewport.work);
    print_operation_work("copy_buffer", copy.work);
    io::printfn(
        "command_path_vk_equivalence operation=dispatch elements=%d expected_checksum=%d direct_checksum=%d public_checksum=%d pairwise=true status=pass",
        COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        dispatch_equivalence.expected,
        dispatch_equivalence.direct,
        dispatch_equivalence.library,
    );
    io::printfn(
        "command_path_vk_equivalence operation=copy_buffer elements=%d expected_checksum=%d direct_checksum=%d public_checksum=%d pairwise=true status=pass",
        COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        copy_equivalence.expected,
        copy_equivalence.direct,
        copy_equivalence.library,
    );
    print_lifecycle(0, &zero, &zero);
    print_lifecycle(1, &one, &zero);
    print_lifecycle(16, &sixteen, &zero);
    print_lifecycle(256, &two_fifty_six, &zero);
    print_lifecycle_work(0, zero.work);
    print_lifecycle_work(1, one.work);
    print_lifecycle_work(16, sixteen.work);
    print_lifecycle_work(256, two_fifty_six.work);
    return 0;
}
