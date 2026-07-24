module gpu_bench;

import gpu;
import gpu_test;
import std::io;
import std::time;

const char[*] BUILD_DISPATCH_SPIRV =
    $embed("shaders/build_dispatch.comp.spv");
const char[*] CONSUME_DISPATCH_SPIRV =
    $embed("shaders/root_pointer.comp.spv");

const uint WAIT_SCOPE_GROUPS = 4096;
const uint WAIT_SCOPE_ITEMS = WAIT_SCOPE_GROUPS * gpu_test::ROOT_POINTER_WORKGROUP;
const uint WAIT_SCOPE_REPETITIONS = 16;
const uint WAIT_SCOPE_WARMUP_REPETITIONS = 2;
const ulong WAIT_SCOPE_TIMEOUT_NS = 10_000_000_000;

struct DispatchBuildRoot {
    gpu::GpuAddress arguments_gpu;
    uint            group_count;
    uint            _pad0;
}
$assert DispatchBuildRoot::size == 16;

fn gpu::PipelineHandle create_bench_pipeline(
    gpu::Device* device,
    char[] spirv,
) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = spirv,
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::ExecutableCommandList record_dispatch_builder(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 1, 1, 1 },
    )!!;
    return gpu::end_commands(&commands)!!;
}

fn gpu::ExecutableCommandList record_indirect_consumer(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress independent_root,
    gpu::GpuAddress indirect_root,
    gpu::GpuSpan arguments,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     independent_root,
        groups:   { WAIT_SCOPE_GROUPS, 1, 1 },
    )!!;
    gpu::cmd_dispatch_indirect(
        commands: &commands,
        root:     indirect_root,
        args:     arguments,
    )!!;
    return gpu::end_commands(&commands)!!;
}

fn double run_wait_scope_iteration(
    gpu::Queue producer_queue,
    gpu::Queue consumer_queue,
    gpu::CommandAllocator* producer_allocator,
    gpu::CommandAllocator* consumer_allocator,
    gpu::PipelineHandle producer_pipeline,
    gpu::PipelineHandle consumer_pipeline,
    gpu::GpuAddress producer_root,
    gpu::GpuAddress independent_root,
    gpu::GpuAddress indirect_root,
    gpu::GpuSpan arguments,
    bool narrow,
) {
    std::time::Clock start = std::time::clock::now();
    gpu::ExecutableCommandList producer = record_dispatch_builder(
        producer_allocator,
        producer_pipeline,
        producer_root,
    );
    gpu::ExecutableCommandList[1] producer_lists = { producer };
    gpu::SubmitDesc producer_submit = {
        .command_lists = producer_lists[..],
    };
    gpu::CompletionPoint produced = gpu::submit(
        producer_queue,
        &producer_submit,
    )!!;

    gpu::ExecutableCommandList consumer = record_indirect_consumer(
        consumer_allocator,
        consumer_pipeline,
        independent_root,
        indirect_root,
        arguments,
    );
    gpu::ExecutableCommandList[1] consumer_lists = { consumer };
    gpu::CompletionWait[1] waits;
    waits[0].point = produced;
    if (narrow) {
        waits[0].consumers = { .draw_arguments };
    } else {
        waits[0].before = { .all };
    }
    gpu::SubmitDesc consumer_submit = {
        .command_lists    = consumer_lists[..],
        .completion_waits = waits[..],
    };
    gpu::CompletionPoint consumed = gpu::submit(
        consumer_queue,
        &consumer_submit,
    )!!;
    gpu::wait_completion(consumed, WAIT_SCOPE_TIMEOUT_NS)!!;
    return (double)(long)start.to_now() / 1.0e6;
}

fn int main() {
    io::printfn(
        "iterations=%d units=ms",
        WAIT_SCOPE_REPETITIONS,
    );
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_completion_wait_scope_bench",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 2 },
        .distinct = { .compute },
    };
    gpu::DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    for (uint adapter_index = 0;
        adapter_index < adapters.count;
        adapter_index++) {
        gpu::Adapter adapter = adapters.get(adapter_index)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) {
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        defer gpu::destroy_device(&device)!!;
        gpu::Queue producer_queue = gpu::get_queue(
            &device,
            gpu::QueueKind.COMPUTE,
            0,
        )!!;
        gpu::Queue consumer_queue = gpu::get_queue(
            &device,
            gpu::QueueKind.COMPUTE,
            1,
        )!!;
        gpu::CommandAllocator producer_allocator =
            gpu::create_command_allocator(&device, producer_queue)!!;
        defer gpu::destroy_command_allocator(&producer_allocator)!!;
        gpu::CommandAllocator consumer_allocator =
            gpu::create_command_allocator(&device, consumer_queue)!!;
        defer gpu::destroy_command_allocator(&consumer_allocator)!!;
        gpu::PipelineHandle producer_pipeline = create_bench_pipeline(
            &device,
            BUILD_DISPATCH_SPIRV[..],
        );
        defer gpu::destroy_pipeline(&device, producer_pipeline)!!;
        gpu::PipelineHandle consumer_pipeline = create_bench_pipeline(
            &device,
            CONSUME_DISPATCH_SPIRV[..],
        );
        defer gpu::destroy_pipeline(&device, consumer_pipeline)!!;

        gpu_test::OwnedSpan arguments = gpu_test::allocate_owned_span(
            device:       &device,
            size:         gpu::DispatchIndirectCommand::size,
            memory_class: gpu::MemoryClass.GPU_PRIVATE,
            access:       { .compute },
            debug_name:   "wait_scope_arguments",
        )!!;
        defer gpu_test::free_owned_span(&device, &arguments)!!;
        gpu_test::OwnedSpan build_root = gpu_test::allocate_owned_span(
            device:       &device,
            size:         DispatchBuildRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "wait_scope_build_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &build_root)!!;
        DispatchBuildRoot* build = (DispatchBuildRoot*)gpu::get_span_mapping(
            &device,
            build_root.span,
        )!!.ptr;
        *build = {
            .arguments_gpu = gpu::get_span_address(&device, arguments.span)!!,
            .group_count   = WAIT_SCOPE_GROUPS,
        };
        gpu::flush_mapped_span(&device, build_root.span)!!;

        gpu_test::OwnedSpan input = gpu_test::allocate_owned_span(
            device:       &device,
            size:         WAIT_SCOPE_ITEMS * float::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "wait_scope_input",
        )!!;
        defer gpu_test::free_owned_span(&device, &input)!!;
        float* input_values = (float*)gpu::get_span_mapping(
            &device,
            input.span,
        )!!.ptr;
        for (uint value_index = 0;
            value_index < WAIT_SCOPE_ITEMS;
            value_index++) {
            input_values[value_index] = 1.0f;
        }
        gpu::flush_mapped_span(&device, input.span)!!;
        gpu_test::OwnedSpan all_output = gpu_test::allocate_owned_span(
            device:       &device,
            size:         WAIT_SCOPE_ITEMS * float::size,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .compute },
            debug_name:   "wait_scope_all_output",
        )!!;
        defer gpu_test::free_owned_span(&device, &all_output)!!;
        gpu_test::OwnedSpan narrow_output = gpu_test::allocate_owned_span(
            device:       &device,
            size:         WAIT_SCOPE_ITEMS * float::size,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .compute },
            debug_name:   "wait_scope_narrow_output",
        )!!;
        defer gpu_test::free_owned_span(&device, &narrow_output)!!;
        gpu_test::OwnedSpan independent_output = gpu_test::allocate_owned_span(
            device:       &device,
            size:         WAIT_SCOPE_ITEMS * float::size,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .compute },
            debug_name:   "wait_scope_independent_output",
        )!!;
        defer gpu_test::free_owned_span(&device, &independent_output)!!;
        gpu_test::OwnedSpan all_root = gpu_test::allocate_owned_span(
            device:       &device,
            size:         gpu_test::ComputeRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "wait_scope_all_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &all_root)!!;
        gpu_test::OwnedSpan narrow_root = gpu_test::allocate_owned_span(
            device:       &device,
            size:         gpu_test::ComputeRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "wait_scope_narrow_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &narrow_root)!!;
        gpu_test::OwnedSpan independent_root = gpu_test::allocate_owned_span(
            device:       &device,
            size:         gpu_test::ComputeRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .compute },
            debug_name:   "wait_scope_independent_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &independent_root)!!;

        gpu_test::ComputeRoot* all_consume =
            (gpu_test::ComputeRoot*)gpu::get_span_mapping(
                &device,
                all_root.span,
            )!!.ptr;
        *all_consume = {
            .input_gpu  = gpu::get_span_address(&device, input.span)!!,
            .output_gpu = gpu::get_span_address(&device, all_output.span)!!,
            .count      = WAIT_SCOPE_ITEMS,
        };
        gpu_test::ComputeRoot* narrow_consume =
            (gpu_test::ComputeRoot*)gpu::get_span_mapping(
                &device,
                narrow_root.span,
            )!!.ptr;
        *narrow_consume = {
            .input_gpu  = gpu::get_span_address(&device, input.span)!!,
            .output_gpu = gpu::get_span_address(&device, narrow_output.span)!!,
            .count      = WAIT_SCOPE_ITEMS,
        };
        gpu_test::ComputeRoot* independent_consume =
            (gpu_test::ComputeRoot*)gpu::get_span_mapping(
                &device,
                independent_root.span,
            )!!.ptr;
        *independent_consume = {
            .input_gpu  = gpu::get_span_address(&device, input.span)!!,
            .output_gpu = gpu::get_span_address(
                &device,
                independent_output.span,
            )!!,
            .count      = WAIT_SCOPE_ITEMS,
        };
        gpu::flush_mapped_span(&device, all_root.span)!!;
        gpu::flush_mapped_span(&device, narrow_root.span)!!;
        gpu::flush_mapped_span(&device, independent_root.span)!!;

        gpu::GpuAddress producer_root_gpu = gpu::get_span_address(
            &device,
            build_root.span,
        )!!;
        gpu::GpuAddress independent_root_gpu = gpu::get_span_address(
            &device,
            independent_root.span,
        )!!;
        gpu::GpuAddress all_root_gpu = gpu::get_span_address(
            &device,
            all_root.span,
        )!!;
        gpu::GpuAddress narrow_root_gpu = gpu::get_span_address(
            &device,
            narrow_root.span,
        )!!;
        for (uint i = 0; i < WAIT_SCOPE_WARMUP_REPETITIONS; i++) {
            (void)run_wait_scope_iteration(
                producer_queue:     producer_queue,
                consumer_queue:     consumer_queue,
                producer_allocator: &producer_allocator,
                consumer_allocator: &consumer_allocator,
                producer_pipeline:  producer_pipeline,
                consumer_pipeline:  consumer_pipeline,
                producer_root:      producer_root_gpu,
                independent_root:   independent_root_gpu,
                indirect_root:      all_root_gpu,
                arguments:          arguments.span,
                narrow:             false,
            );
            (void)run_wait_scope_iteration(
                producer_queue:     producer_queue,
                consumer_queue:     consumer_queue,
                producer_allocator: &producer_allocator,
                consumer_allocator: &consumer_allocator,
                producer_pipeline:  producer_pipeline,
                consumer_pipeline:  consumer_pipeline,
                producer_root:      producer_root_gpu,
                independent_root:   independent_root_gpu,
                indirect_root:      narrow_root_gpu,
                arguments:          arguments.span,
                narrow:             true,
            );
        }

        double all_ms = 0.0;
        double draw_arguments_ms = 0.0;
        for (uint i = 0; i < WAIT_SCOPE_REPETITIONS; i++) {
            if ((i & 1) == 0) {
                all_ms += run_wait_scope_iteration(
                    producer_queue:     producer_queue,
                    consumer_queue:     consumer_queue,
                    producer_allocator: &producer_allocator,
                    consumer_allocator: &consumer_allocator,
                    producer_pipeline:  producer_pipeline,
                    consumer_pipeline:  consumer_pipeline,
                    producer_root:      producer_root_gpu,
                    independent_root:   independent_root_gpu,
                    indirect_root:      all_root_gpu,
                    arguments:          arguments.span,
                    narrow:             false,
                );
                draw_arguments_ms += run_wait_scope_iteration(
                    producer_queue:     producer_queue,
                    consumer_queue:     consumer_queue,
                    producer_allocator: &producer_allocator,
                    consumer_allocator: &consumer_allocator,
                    producer_pipeline:  producer_pipeline,
                    consumer_pipeline:  consumer_pipeline,
                    producer_root:      producer_root_gpu,
                    independent_root:   independent_root_gpu,
                    indirect_root:      narrow_root_gpu,
                    arguments:          arguments.span,
                    narrow:             true,
                );
            } else {
                draw_arguments_ms += run_wait_scope_iteration(
                    producer_queue:     producer_queue,
                    consumer_queue:     consumer_queue,
                    producer_allocator: &producer_allocator,
                    consumer_allocator: &consumer_allocator,
                    producer_pipeline:  producer_pipeline,
                    consumer_pipeline:  consumer_pipeline,
                    producer_root:      producer_root_gpu,
                    independent_root:   independent_root_gpu,
                    indirect_root:      narrow_root_gpu,
                    arguments:          arguments.span,
                    narrow:             true,
                );
                all_ms += run_wait_scope_iteration(
                    producer_queue:     producer_queue,
                    consumer_queue:     consumer_queue,
                    producer_allocator: &producer_allocator,
                    consumer_allocator: &consumer_allocator,
                    producer_pipeline:  producer_pipeline,
                    consumer_pipeline:  consumer_pipeline,
                    producer_root:      producer_root_gpu,
                    independent_root:   independent_root_gpu,
                    indirect_root:      all_root_gpu,
                    arguments:          arguments.span,
                    narrow:             false,
                );
            }
        }
        all_ms /= (double)WAIT_SCOPE_REPETITIONS;
        draw_arguments_ms /= (double)WAIT_SCOPE_REPETITIONS;

        gpu::invalidate_mapped_span(&device, all_output.span)!!;
        gpu::invalidate_mapped_span(&device, narrow_output.span)!!;
        gpu::invalidate_mapped_span(&device, independent_output.span)!!;
        float* all_output_values = (float*)gpu::get_span_mapping(
            &device,
            all_output.span,
        )!!.ptr;
        float* narrow_output_values = (float*)gpu::get_span_mapping(
            &device,
            narrow_output.span,
        )!!.ptr;
        float* independent_output_values = (float*)gpu::get_span_mapping(
            &device,
            independent_output.span,
        )!!.ptr;
        if (all_output_values[0] != 2.0f
            || all_output_values[WAIT_SCOPE_ITEMS - 1] != 2.0f
            || narrow_output_values[0] != 2.0f
            || narrow_output_values[WAIT_SCOPE_ITEMS - 1] != 2.0f
            || independent_output_values[0] != 2.0f
            || independent_output_values[WAIT_SCOPE_ITEMS - 1] != 2.0f) {
            io::eprintfn("completion wait scope output mismatch");
            return 1;
        }
        io::printfn(
            "status=advisory groups=%d all=%.6f ms draw_arguments=%.6f ms ratio=%.6f",
            WAIT_SCOPE_GROUPS,
            all_ms,
            draw_arguments_ms,
            draw_arguments_ms / all_ms,
        );
        return 0;
    }

    io::printfn(
        "status=not_applicable reason=two_distinct_compute_queues_unavailable",
    );
    io::printfn("all=0.000000 ms draw_arguments=0.000000 ms ratio=0.000000");
    return 0;
}
