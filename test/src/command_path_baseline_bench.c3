module gpu_bench;

import gpu @public;
import gpu::internal::vk @public;
import std::io;
import std::time;

const char[*] COMMAND_PATH_COMP_SPIRV =
    $embed("shaders/root_pointer.comp.spv");

const uint COMMAND_PATH_ITERATIONS = 20_000;
const uint COMMAND_PATH_REPETITIONS = 5;
const uint COMMAND_PATH_EQUIVALENCE_ELEMENTS = 64;
const usz BENCH_SPAN_ALIGNMENT = 16;
const uint DISPATCH_DIRECT_SENTINEL = 0x11111111;
const uint DISPATCH_PUBLIC_SENTINEL = 0x22222222;
const uint COPY_DIRECT_SENTINEL = 0x33333333;
const uint COPY_PUBLIC_SENTINEL = 0x44444444;

struct BenchSpan {
    gpu::GpuAllocation allocation;
    gpu::GpuSpan       span;
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

struct OperationSamples {
    double[COMMAND_PATH_REPETITIONS] direct;
    double[COMMAND_PATH_REPETITIONS] library;
}

struct EquivalenceResult {
    uint checksum;
}

fn BenchSpan? allocate_bench_span(
    gpu::Device* device,
    usz size,
    gpu::MemoryClass memory_class,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = BENCH_SPAN_ALIGNMENT,
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
        .spirv       = COMMAND_PATH_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader_desc,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
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

fn double measure_direct_dispatch(
    vk::CommandPathNativeContext* native,
) {
    gpu::Vec3u groups = { 1, 1, 1 };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_dispatch(
            native,
            (gpu::GpuAddress)0,
            groups,
        );
    }
    return elapsed_per_operation(start);
}

fn double measure_public_dispatch(gpu::CommandList* commands) {
    gpu::Vec3u groups = { 1, 1, 1 };
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_dispatch(
            commands,
            (gpu::GpuAddress)0,
            groups,
        )!!;
    }
    return elapsed_per_operation(start);
}

fn OperationSamples measure_dispatch(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
) {
    OperationSamples result;
    gpu::Vec3u groups = { 1, 1, 1 };
    for (uint repetition = 0;
        repetition < COMMAND_PATH_REPETITIONS;
        repetition++
    ) {
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
        vk::command_path_direct_dispatch(
            &native,
            (gpu::GpuAddress)0,
            groups,
        );
        gpu::cmd_dispatch(
            &library,
            (gpu::GpuAddress)0,
            groups,
        )!!;

        if (repetition % 2 == 0) {
            result.direct[repetition] = measure_direct_dispatch(&native);
            result.library[repetition] = measure_public_dispatch(&library);
        } else {
            result.library[repetition] = measure_public_dispatch(&library);
            result.direct[repetition] = measure_direct_dispatch(&native);
        }
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double measure_direct_copy(
    vk::CommandPathNativeContext* native,
    vk::CommandPathNativeCopy* copy,
) {
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        vk::command_path_direct_copy_buffer(native, copy);
    }
    return elapsed_per_operation(start);
}

fn double measure_public_copy(
    gpu::CommandList* commands,
    gpu::BufferCopyDesc* copy,
) {
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_PATH_ITERATIONS; i++) {
        gpu::cmd_copy_buffer(commands, copy)!!;
    }
    return elapsed_per_operation(start);
}

fn OperationSamples measure_copy(
    gpu::Device* device,
    gpu::CommandAllocator* allocator,
    gpu::BufferCopyDesc* copy,
) {
    OperationSamples result;
    vk::CommandPathNativeCopy direct_copy = vk::command_path_native_copy(
        device,
        copy.src,
        copy.dst,
    )!!;
    for (uint repetition = 0;
        repetition < COMMAND_PATH_REPETITIONS;
        repetition++
    ) {
        gpu::CommandList direct = gpu::begin_commands(allocator)!!;
        gpu::CommandList library = gpu::begin_commands(allocator)!!;
        vk::CommandPathNativeContext native =
            vk::command_path_native_context(&direct)!!;
        vk::command_path_direct_copy_buffer(&native, &direct_copy);
        gpu::cmd_copy_buffer(&library, copy)!!;

        if (repetition % 2 == 0) {
            result.direct[repetition] =
                measure_direct_copy(&native, &direct_copy);
            result.library[repetition] =
                measure_public_copy(&library, copy);
        } else {
            result.library[repetition] =
                measure_public_copy(&library, copy);
            result.direct[repetition] =
                measure_direct_copy(&native, &direct_copy);
        }
        finish_recording(&direct);
        finish_recording(&library);
    }
    return result;
}

fn double sample_median(
    double[COMMAND_PATH_REPETITIONS] samples,
) {
    double[COMMAND_PATH_REPETITIONS] sorted = samples;
    for (uint i = 1; i < COMMAND_PATH_REPETITIONS; i++) {
        double key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    return sorted[COMMAND_PATH_REPETITIONS / 2];
}

fn void print_operation(
    ZString operation,
    OperationSamples* result,
) {
    double direct = sample_median(result.direct);
    double library = sample_median(result.library);
    io::printfn(
        "%s recording: iterations=%d repetitions=%d "
        "direct=%.3f ns/op public=%.3f ns/op public/direct=%.3fx",
        operation,
        COMMAND_PATH_ITERATIONS,
        COMMAND_PATH_REPETITIONS,
        direct,
        library,
        library / direct,
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
    const usz DATA_BYTES =
        COMMAND_PATH_EQUIVALENCE_ELEMENTS * float::size;
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

    float* input_values =
        (float*)gpu::get_span_mapping(device, input.span)!!.ptr;
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
    gpu::cmd_fill_buffer(
        &direct,
        direct_output.span,
        DISPATCH_DIRECT_SENTINEL,
    )!!;
    gpu::cmd_fill_buffer(
        &library,
        public_output.span,
        DISPATCH_PUBLIC_SENTINEL,
    )!!;
    gpu::Barrier to_compute = {
        .before = { .transfer = true },
        .after  = { .compute = true },
    };
    gpu::cmd_barrier(&direct, &to_compute)!!;
    gpu::cmd_barrier(&library, &to_compute)!!;
    vk::CommandPathNativeContext native =
        vk::command_path_native_context(&direct)!!;
    gpu::Vec3u groups = { 1, 1, 1 };
    vk::command_path_direct_dispatch(
        &native,
        gpu::get_span_address(device, direct_root.span)!!,
        groups,
    );
    gpu::cmd_dispatch(
        &library,
        gpu::get_span_address(device, public_root.span)!!,
        groups,
    )!!;
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
        result.checksum += (uint)direct_values[i];
    }
    assert(result.checksum != 0);
    return result;
}

fn void preseed_copy_outputs(
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
    gpu::GpuSpan direct_output,
    gpu::GpuSpan public_output,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_fill_buffer(
        &commands,
        direct_output,
        COPY_DIRECT_SENTINEL,
    )!!;
    gpu::cmd_fill_buffer(
        &commands,
        public_output,
        COPY_PUBLIC_SENTINEL,
    )!!;
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

    char* source_values =
        (char*)gpu::get_span_mapping(device, source.span)!!.ptr;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        source_values[i] = (char)(i + 1);
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
    vk::command_path_direct_copy_buffer(&native, &direct_copy);
    gpu::BufferCopyDesc public_copy = {
        .src = source.span,
        .dst = public_output.span,
    };
    gpu::cmd_copy_buffer(&library, &public_copy)!!;
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
    EquivalenceResult result;
    for (uint i = 0; i < COMMAND_PATH_EQUIVALENCE_ELEMENTS; i++) {
        assert(direct_values[i] == source_values[i]);
        assert(public_values[i] == source_values[i]);
        assert(direct_values[i] == public_values[i]);
        result.checksum += (uint)direct_values[i];
    }
    assert(result.checksum != 0);
    return result;
}

fn int main() {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_command_path_baseline",
    };
    gpu::Device device = create_bench_device(&runtime_desc)!!;
    defer destroy_bench_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::PipelineHandle pipeline = create_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, pipeline)!!;

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

    OperationSamples dispatch = measure_dispatch(&allocator, pipeline);
    OperationSamples copy =
        measure_copy(&device, &allocator, &timing_copy);
    EquivalenceResult dispatch_equivalence =
        execute_dispatch_equivalence(
            device:    &device,
            queue:     queue,
            allocator: &allocator,
            pipeline:  pipeline,
        );
    EquivalenceResult copy_equivalence =
        execute_copy_equivalence(&device, queue, &allocator);

    io::printfn(
        "command path baseline: iterations=%d repetitions=%d",
        COMMAND_PATH_ITERATIONS,
        COMMAND_PATH_REPETITIONS,
    );
    print_operation("dispatch", &dispatch);
    print_operation("copy buffer", &copy);
    io::printfn(
        "dispatch equivalence: elements=%d nonzero output matched checksum=%d",
        COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        dispatch_equivalence.checksum,
    );
    io::printfn(
        "copy equivalence: elements=%d nonzero output matched checksum=%d",
        COMMAND_PATH_EQUIVALENCE_ELEMENTS,
        copy_equivalence.checksum,
    );
    return 0;
}
