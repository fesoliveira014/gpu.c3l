module gpu_test;

import gpu;

// Headless (lavapipe): pipeline-time reflection validation against the
// heap convention. Fixtures come from test/shaders/ via build_shaders.sh.

const char[*] REFLECT_HEAP_SPIRV        = $embed("shaders/heap_write.comp.spv");
const char[*] REFLECT_BUFFERS_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] REFLECT_BAD_SET_SPIRV     = $embed("shaders/bad_set.comp.spv");
const char[*] REFLECT_BAD_BINDING_SPIRV = $embed("shaders/bad_binding.comp.spv");
const char[*] REFLECT_OVERSIZED_PUSH_SPIRV = $embed("shaders/oversized_push.comp.spv");
const char[*] REFLECT_VERTEX_SPIRV      = $embed("shaders/offscreen.vert.spv");

const ZString ENTRY_POINT_OVER_CAP = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

struct ShaderDiagnosticCapture {
    uint                      calls;
    gpu::DebugMessageSeverity severity;
    gpu::DebugMessageCategory category;
    bool                      has_fault;
    fault                     public_fault;
    gpu::DebugResourceKind    resource_kind;
    bool                      has_public_identity;
    uint                      resource_index;
    uint                      resource_generation;
    int                       validation_id_number;
    char[64]                  operation;
    char[64]                  resource_name;
    char[64]                  rejected_field;
    char[128]                 invariant;
    char[128]                 backend_text;
    char[64]                  validation_id_name;
}

fn void copy_shader_diagnostic_text(char* destination, usz capacity, ZString source) {
    usz index = 0;
    if (source != null) {
        while (index + 1 < capacity && source[index] != 0) {
            destination[index] = source[index];
            index++;
        }
    }
    destination[index] = 0;
}

fn void capture_shader_diagnostic(gpu::DebugMessage* message, void* user_data) {
    ShaderDiagnosticCapture* capture = (ShaderDiagnosticCapture*)user_data;
    capture.calls++;
    capture.severity = message.severity;
    capture.category = message.category;
    capture.has_fault = message.has_fault;
    capture.public_fault = message.public_fault;
    capture.resource_kind = message.resource.kind;
    capture.has_public_identity = message.resource.has_public_identity;
    capture.resource_index = message.resource.index;
    capture.resource_generation = message.resource.generation;
    capture.validation_id_number = message.validation_id_number;
    copy_shader_diagnostic_text(&capture.operation[0], 64, message.operation);
    copy_shader_diagnostic_text(&capture.resource_name[0], 64, message.resource.debug_name);
    copy_shader_diagnostic_text(&capture.rejected_field[0], 64, message.rejected_field);
    copy_shader_diagnostic_text(&capture.invariant[0], 128, message.invariant);
    copy_shader_diagnostic_text(&capture.backend_text[0], 128, message.backend_text);
    copy_shader_diagnostic_text(&capture.validation_id_name[0], 64, message.validation_id_name);
}

fn void assert_shader_rejection(
    ShaderDiagnosticCapture* capture,
    uint before,
    fault returned,
    ZString operation,
    ZString field,
    ZString invariant,
    gpu::DebugResourceKind resource_kind,
) {
    assert(capture.calls == before + 1, "shader/pipeline rejection emits exactly one diagnostic");
    assert(((ZString)&capture.operation[0]).str_view() == operation.str_view(), "diagnostic operation is exact");
    assert(capture.has_fault && capture.public_fault == returned, "diagnostic fault matches return");
    assert(capture.resource_kind == resource_kind, "diagnostic resource kind is exact");
    assert(((ZString)&capture.rejected_field[0]).str_view() == field.str_view(), "diagnostic rejected field is exact");
    assert(((ZString)&capture.invariant[0]).str_view() == invariant.str_view(), "diagnostic invariant is exact");
}

fn gpu::Device make_diagnostic_reflection_test_device(ShaderDiagnosticCapture* capture) {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .debug_callback  = capture == null ? null : &capture_shader_diagnostic,
        .debug_user_data = capture,
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_reflection_test_device() {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_shader_reflection_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::PipelineHandle? create_reflection_compute_pipeline(
    gpu::Device* device,
    gpu::ShaderDesc* shader_desc,
) {
    gpu::ShaderCode shader = gpu::prepare_shader_code(shader_desc)!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
        .push_constant_size = gpu::RootPush::size,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc);
}

fn void heap_conforming_shader_accepted() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_HEAP_SPIRV[..], .entry_point = "main",
    };
    gpu::PipelineHandle pipeline = create_reflection_compute_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(pipeline.is_valid(), "heap-conforming shader is accepted");
}

fn void buffers_only_shader_accepted() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "main",
    };
    gpu::PipelineHandle pipeline = create_reflection_compute_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(pipeline.is_valid(), "buffers-only shader with no bindings is accepted");
}

fn void foreign_set_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BAD_SET_SPIRV[..], .entry_point = "main",
    };
    fault set_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(set_fault == gpu::SHADER_INVALID, "a set-1 binding faults SHADER_INVALID");
}

fn void convention_violating_binding_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BAD_BINDING_SPIRV[..], .entry_point = "main",
    };
    fault binding_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(binding_fault == gpu::SHADER_INVALID, "a combined-image-sampler at set-0 binding 0 faults SHADER_INVALID");
}

fn void bogus_entry_point_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "not_main",
    };
    fault entry_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(entry_fault == gpu::INVALID_ARGUMENT, "a requested entry point absent from the module faults INVALID_ARGUMENT");
}

fn void wrong_stage_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.VERTEX, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "main",
    };
    fault stage_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(stage_fault == gpu::INVALID_ARGUMENT, "requesting VERTEX for a compute module faults INVALID_ARGUMENT");
}

fn void oversized_push_constant_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_OVERSIZED_PUSH_SPIRV[..], .entry_point = "main",
    };
    fault push_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(push_fault == gpu::INVALID_ARGUMENT, "a push-constant block past the root-push ABI faults INVALID_ARGUMENT");
}

fn void entry_point_over_cap_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = ENTRY_POINT_OVER_CAP,
    };
    fault cap_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(cap_fault == gpu::INVALID_ARGUMENT, "an entry point name past SHADER_ENTRY_POINT_CAP faults INVALID_ARGUMENT");
}

fn void shader_contract_rejections_report_exact_context() @test {
    ShaderDiagnosticCapture capture;
    gpu::Device device = make_diagnostic_reflection_test_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc missing_entry = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..],
        .entry_point = "not_main", .debug_name = "missing_entry_shader",
    };
    uint before = capture.calls;
    fault entry_fault = @catch(create_reflection_compute_pipeline(&device, &missing_entry));
    assert(entry_fault == gpu::INVALID_ARGUMENT, "missing entry keeps INVALID_ARGUMENT");
    assert_shader_rejection(
        capture:       &capture,
        before:        before,
        returned:      entry_fault,
        operation:     "create_compute_pipeline",
        field:         "ComputePipelineDesc.shader",
        invariant:     "entry point must exist in SPIR-V module",
        resource_kind: gpu::DebugResourceKind.SHADER,
    );
    assert(((ZString)&capture.resource_name[0]).str_view() == "missing_entry_shader", "shader debug name is preserved");
    assert(!capture.has_public_identity, "borrowed shader code has no public identity");

    gpu::ShaderDesc wrong_stage = {
        .stage = gpu::ShaderStage.VERTEX, .spirv = REFLECT_BUFFERS_SPIRV[..],
        .entry_point = "main", .debug_name = "wrong_stage_shader",
    };
    before = capture.calls;
    fault stage_fault = @catch(create_reflection_compute_pipeline(&device, &wrong_stage));
    assert(stage_fault == gpu::INVALID_ARGUMENT, "wrong stage keeps INVALID_ARGUMENT");
    assert_shader_rejection(
        capture:       &capture,
        before:        before,
        returned:      stage_fault,
        operation:     "create_compute_pipeline",
        field:         "ComputePipelineDesc.shader",
        invariant:     "shader stage must match its pipeline role",
        resource_kind: gpu::DebugResourceKind.SHADER,
    );

    gpu::ShaderDesc bad_abi = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_OVERSIZED_PUSH_SPIRV[..],
        .entry_point = "main", .debug_name = "bad_abi_shader",
    };
    before = capture.calls;
    fault abi_fault = @catch(create_reflection_compute_pipeline(&device, &bad_abi));
    assert(abi_fault == gpu::INVALID_ARGUMENT, "oversized ABI keeps INVALID_ARGUMENT");
    assert_shader_rejection(
        capture:       &capture,
        before:        before,
        returned:      abi_fault,
        operation:     "create_compute_pipeline",
        field:         "ComputePipelineDesc.shader",
        invariant:     "push constants must fit the shader-stage root ABI",
        resource_kind: gpu::DebugResourceKind.SHADER,
    );
}

fn void wrong_pipeline_shader_context_reports_borrowed_code_once() @test {
    ShaderDiagnosticCapture capture;
    gpu::Device device = make_diagnostic_reflection_test_device(&capture);
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc vertex_desc = {
        .stage = gpu::ShaderStage.VERTEX, .spirv = REFLECT_VERTEX_SPIRV[..],
        .entry_point = "main", .debug_name = "vertex_in_compute",
    };
    gpu::ShaderCode vertex = gpu::prepare_shader_code(&vertex_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = vertex, .push_constant_size = gpu::RootPush::size };
    uint before = capture.calls;
    fault pipeline_fault = @catch(gpu::create_compute_pipeline(&device, &pipeline_desc));
    assert(pipeline_fault == gpu::INVALID_ARGUMENT, "wrong compute shader context keeps INVALID_ARGUMENT");
    assert_shader_rejection(
        capture:       &capture,
        before:        before,
        returned:      pipeline_fault,
        operation:     "create_compute_pipeline",
        field:         "ComputePipelineDesc.shader",
        invariant:     "shader stage must match its pipeline role",
        resource_kind: gpu::DebugResourceKind.SHADER,
    );
    assert(!capture.has_public_identity, "borrowed shader code has no public identity");
    assert(capture.resource_index == 0 && capture.resource_generation == 0,
        "borrowed shader diagnostics invent no handle identity");
    assert(((ZString)&capture.resource_name[0]).str_view() == "vertex_in_compute",
        "borrowed shader diagnostics preserve the caller debug name");
}

fn void shader_diagnostic_callback_preserves_fault_parity() @test {
    ShaderDiagnosticCapture capture;
    gpu::Device callback_device = make_diagnostic_reflection_test_device(&capture);
    gpu::ShaderDesc rejected = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_BUFFERS_SPIRV[..], .entry_point = "not_main",
    };
    uint before = capture.calls;
    fault callback_fault = @catch(create_reflection_compute_pipeline(&callback_device, &rejected));
    assert(capture.calls == before + 1, "callback shader rejection emits exactly once");
    destroy_test_device(&callback_device)!!;
    gpu::Device plain_device = make_diagnostic_reflection_test_device(null);
    fault plain_fault = @catch(create_reflection_compute_pipeline(&plain_device, &rejected));
    destroy_test_device(&plain_device)!!;
    assert(plain_fault == callback_fault, "callback presence does not change shader fault");
}
