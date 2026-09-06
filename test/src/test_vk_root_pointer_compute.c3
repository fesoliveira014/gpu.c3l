module gpu_test;

import gpu::internal @public;
import gpu;
import gpu::internal::vk @public;

const char[*] ROOT_POINTER_SPIRV = $embed("shaders/root_pointer.comp.spv");
const char[*] ROOT_REFERENCE_SPIRV = $embed("shaders/root_ref.comp.spv");
const char[*] ROOT_SIGNED_SPIRV = $embed("shaders/root_signed.comp.spv");
const char[*] ROOT_MATRIX_ARRAY_SPIRV = $embed("shaders/root_matrix_array.comp.spv");
const uint MATRIX_ROOT_COUNT = 16;

struct DirectReferenceRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
}

fn void dispatch_root_probe(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
) {
    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)0,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}

fn void root_pointer_compute_readback() @test {
    const uint COUNT = 256;
    gpu::RuntimeDesc device_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
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
        .spirv       = ROOT_POINTER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
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
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
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

fn Vec4f transform_column_major(Mat4f* m, Vec4f v) {
    Vec4f result;
    for (uint c = 0; c < 4; c++) result += (*m)[c] * v[c];
    return result;
}

fn void matrix_and_array_root_round_trips() @test {
    gpu::RuntimeDesc device_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_matrix_root_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         MATRIX_ROOT_COUNT * Vec4f::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "matrix_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         MatrixRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "matrix_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    MatrixRoot* root = (MatrixRoot*)test_span_mapping(&device, root_owned.span).ptr;
    *root = {
        .output_gpu = test_span_address(&device, output.span),
        .count      = MATRIX_ROOT_COUNT,
        .transform  = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } },
    };
    for (uint k = 0; k < 4; k++) root.planes[k] = { (float)k + 1, (float)k + 2, (float)k + 3, 1 };
    for (uint k = 0; k < 8; k++) root.ids[k] = 10 * k;
    gpu::flush_mapped_span(&device, root_owned.span)!!;

    gpu::ShaderDesc shader = { .spirv = ROOT_MATRIX_ARRAY_SPIRV[..], .entry_point = "main" };
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(&commands, test_span_address(&device, root_owned.span), { 1, 1, 1 })!!;
    gpu::Barrier host_read = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::wait_completion(gpu::submit(queue, &submit)!!)!!;
    gpu::invalidate_mapped_span(&device, output.span)!!;

    Vec4f* results = (Vec4f*)test_span_mapping(&device, output.span).ptr;
    for (uint i = 0; i < MATRIX_ROOT_COUNT; i++) {
        Vec4f expected = transform_column_major(&root.transform, root.planes[i % 4]);
        expected[3] += (float)root.ids[i % 8];
        for (uint c = 0; c < 4; c++) {
            assert(results[i][c] == expected[c], "matrix and array root values round-trip through the shader");
        }
    }
}

fn void direct_reference_root_runtime_evidence() @test {
    const uint INPUT_VALUE = 0x1357_9bdf;
    const uint XOR_VALUE = 0xa5a5_5a5a;
    gpu::RuntimeDesc device_desc = {
        .contract_validation       = gpu::ContractValidation.FULL,
        .enable_vulkan_validation  = true,
        .application_name          = "gpu_root_reference_evidence",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    OwnedSpan input = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "reference_input",
    )!!;
    defer free_owned_span(&device, &input)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "reference_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         DirectReferenceRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "direct_reference_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;

    uint* input_value = (uint*)test_span_mapping(&device, input.span).ptr;
    *input_value = INPUT_VALUE;
    gpu::flush_mapped_span(&device, input.span)!!;
    DirectReferenceRoot* root =
        (DirectReferenceRoot*)test_span_mapping(
            &device,
            root_owned.span,
        ).ptr;
    root.input_gpu = test_span_address(&device, input.span);
    root.output_gpu = test_span_address(&device, output.span);
    gpu::flush_mapped_span(&device, root_owned.span)!!;

    gpu::ShaderDesc shader_desc = {
        .spirv       = ROOT_REFERENCE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline =
        vk::create_compute_pipeline_with_root_abi_evidence(
            &device,
            &pipeline_desc,
        )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    dispatch_root_probe(
        &device,
        pipeline,
        test_span_address(&device, root_owned.span),
    );
    gpu::invalidate_mapped_span(&device, output.span)!!;
    uint* output_value = (uint*)test_span_mapping(&device, output.span).ptr;
    assert(*output_value == (INPUT_VALUE ^ XOR_VALUE),
        "direct reference dereferences the pushed nonzero GPU address");
}

fn void signed_root_runtime_evidence() @test {
    const uint EXPECTED = 0xc3a5_f17e;
    gpu::RuntimeDesc device_desc = {
        .contract_validation       = gpu::ContractValidation.FULL,
        .enable_vulkan_validation  = true,
        .application_name          = "gpu_signed_root_evidence",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "signed_root_output",
    )!!;
    defer free_owned_span(&device, &output)!!;

    gpu::ShaderDesc shader_desc = {
        .spirv       = ROOT_SIGNED_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline =
        vk::create_compute_pipeline_with_root_abi_evidence(
            &device,
            &pipeline_desc,
        )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    dispatch_root_probe(
        &device,
        pipeline,
        test_span_address(&device, output.span),
    );
    gpu::invalidate_mapped_span(&device, output.span)!!;
    uint* output_value = (uint*)test_span_mapping(&device, output.span).ptr;
    assert(*output_value == EXPECTED,
        "signed root preserves the nonzero address bits before conversion");
}
