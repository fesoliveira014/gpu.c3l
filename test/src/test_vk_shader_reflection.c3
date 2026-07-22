module gpu_test;

import gpu @public;
import gpu::vk @public;
import spvreflect;

// Headless (lavapipe): pipeline-time reflection validation against the
// heap convention. Fixtures come from test/shaders/ via scripts/build_shaders.py.

const char[*] REFLECT_HEAP_SPIRV        = $embed("shaders/heap_write.comp.spv");
const char[*] REFLECT_BUFFERS_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] REFLECT_BAD_SET_SPIRV     = $embed("shaders/bad_set.comp.spv");
const char[*] REFLECT_BAD_BINDING_SPIRV = $embed("shaders/bad_binding.comp.spv");
const char[*] REFLECT_OVERSIZED_PUSH_SPIRV = $embed("shaders/oversized_push.comp.spv");
const char[*] REFLECT_VERTEX_SPIRV      = $embed("shaders/offscreen.vert.spv");
const char[*] REFLECT_FRAGMENT_SPIRV    = $embed("shaders/offscreen.frag.spv");
const char[*] REFLECT_NO_PUSH_SPIRV     = $embed("shaders/no_push.comp.spv");
const char[*] REFLECT_SIGNED_SPIRV      = $embed("shaders/root_signed.comp.spv");
const char[*] REFLECT_U32_SPIRV         = $embed("shaders/root_u32.comp.spv");
const char[*] REFLECT_FLOAT64_SPIRV     = $embed("shaders/root_float64.comp.spv");
const char[*] REFLECT_VECTOR_SPIRV      = $embed("shaders/root_vector.comp.spv");
const char[*] REFLECT_MATRIX_SPIRV      = $embed("shaders/root_matrix.comp.spv");
const char[*] REFLECT_ARRAY_SPIRV       = $embed("shaders/root_array.comp.spv");
const char[*] REFLECT_NESTED_SPIRV      = $embed("shaders/root_nested.comp.spv");
const char[*] REFLECT_BOOL_SPIRV        = $embed("shaders/root_bool.comp.spv");
const char[*] REFLECT_REF_SPIRV         = $embed("shaders/root_ref.comp.spv");
const char[*] REFLECT_MEMBER_COUNT_SPIRV =
    $embed("shaders/root_two_members.comp.spv");
const char[*] REFLECT_WRONG_OFFSET_SPIRV =
    $embed("shaders/root_wrong_offset.comp.spv");
const char[*] REFLECT_MULTIPLE_BLOCKS_SPIRV =
    $embed("shaders/root_multiple_blocks.comp.spv");
const char[*] REFLECT_MULTI_ENTRY_SPIRV =
    $embed("shaders/multi_entry_root.comp.spv");
const char[*] REFLECT_REVERSED_GRAPHICS_SPIRV =
    $embed("shaders/graphics_reversed_members.vert.spv");
const char[*] REFLECT_GRAPHICS_SIZE_8_SPIRV =
    $embed("shaders/graphics_size_8.vert.spv");
const char[*] REFLECT_GRAPHICS_SECOND_OFFSET_4_SPIRV =
    $embed("shaders/graphics_second_offset_4.vert.spv");
const char[*] REFLECT_RENAMED_SPIRV     = $embed("shaders/root_renamed.comp.spv");

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
        .backend         = gpu::BackendKind.VULKAN,
        .debug_callback  = capture == null ? null : &capture_shader_diagnostic,
        .debug_user_data = capture,
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_reflection_test_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
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
        .shader = shader,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc);
}

fn void assert_shader_interface_rejected(
    gpu::Device* device,
    char[] spirv,
    ZString entry_point,
    ZString message,
) {
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(*device);
    ulong reflections_before = vk::shader_reflection_validation_count(state);
    ulong native_attempts_before = vk::pipeline_shader_create_attempt_count(state);
    gpu::ShaderDesc desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = spirv,
        .entry_point = entry_point,
    };
    fault root_fault = @catch(create_reflection_compute_pipeline(device, &desc));
    assert(root_fault == gpu::SHADER_INVALID,
        "expected SHADER_INVALID, got %s: %s", root_fault, message);
    assert(vk::shader_reflection_validation_count(state) == reflections_before + 1,
        "one malformed shader performs one reflection validation");
    assert(vk::pipeline_shader_create_attempt_count(state) == native_attempts_before,
        "reflection rejection precedes native shader creation");
}

fn void? check_fixture_root_abi(
    char[] spirv,
    gpu::ShaderStage stage,
    ZString entry_point,
) {
    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module =
        (spvreflect::ShaderModule*)&module_storage;
    if (spvreflect::create_shader_module(
        spirv.len,
        spirv.ptr,
        reflect_module,
    ) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    defer reflect_module.destroy();
    vk::check_root_push_abi(reflect_module, entry_point, stage)!;
}

fn void assert_fixture_root_flag(
    char[] spirv,
    spvreflect::TypeFlags required,
) {
    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module =
        (spvreflect::ShaderModule*)&module_storage;
    assert(spvreflect::create_shader_module(
        spirv.len,
        spirv.ptr,
        reflect_module,
    ) == spvreflect::ReflectResult.RESULT_SUCCESS,
        "the malformed root fixture must remain reflectable");
    defer reflect_module.destroy();
    uint count = 0;
    assert(reflect_module.enumerate_entry_point_push_constant_blocks(
        "main",
        &count,
        null,
    ) == spvreflect::ReflectResult.RESULT_SUCCESS && count == 1,
        "the malformed root fixture must contain one push block");
    spvreflect::BlockVariable*[1] blocks;
    assert(reflect_module.enumerate_entry_point_push_constant_blocks(
        "main",
        &count,
        &blocks[0],
    ) == spvreflect::ReflectResult.RESULT_SUCCESS,
        "the malformed root fixture block must enumerate");
    spvreflect::BlockVariable* member = blocks[0].member(0);
    assert(member != null
        && ((uint)member.type_flags() & (uint)required) != 0,
        "the malformed root fixture must expose its required type flag");
}

fn void assert_fixture_root_block_shape(
    char[] spirv,
    uint expected_size,
    uint expected_members,
) {
    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module =
        (spvreflect::ShaderModule*)&module_storage;
    assert(spvreflect::create_shader_module(
        spirv.len,
        spirv.ptr,
        reflect_module,
    ) == spvreflect::ReflectResult.RESULT_SUCCESS,
        "the member-count fixture must remain reflectable");
    defer reflect_module.destroy();
    uint count = 0;
    assert(reflect_module.enumerate_entry_point_push_constant_blocks(
        "main",
        &count,
        null,
    ) == spvreflect::ReflectResult.RESULT_SUCCESS && count == 1,
        "the member-count fixture must contain one push block");
    spvreflect::BlockVariable*[1] blocks;
    assert(reflect_module.enumerate_entry_point_push_constant_blocks(
        "main",
        &count,
        &blocks[0],
    ) == spvreflect::ReflectResult.RESULT_SUCCESS,
        "the member-count fixture block must enumerate");
    assert(blocks[0].byte_size() == expected_size
        && blocks[0].member_count() == expected_members,
        "the member-count fixture must isolate the count mismatch");
}

fn void assert_reflected_shader_rejected(
    gpu::Device* device,
    char[] spirv,
    gpu::ShaderStage stage,
    ZString message,
) {
    gpu::ShaderDesc desc = {
        .stage       = stage,
        .spirv       = spirv,
        .entry_point = "main",
    };
    gpu::ShaderCode code = gpu::prepare_shader_code(&desc)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(*device);
    ulong reflections_before = vk::shader_reflection_validation_count(state);
    ulong native_attempts_before = vk::pipeline_shader_create_attempt_count(state);

    fault shader_fault = @catch(vk::create_pipeline_shader_module(
        state:          state,
        code:           &code,
        expected_stage: stage,
        operation:      "test_shader_reflection",
        rejected_field: "ShaderCode",
    ));

    assert(shader_fault == gpu::SHADER_INVALID,
        "expected SHADER_INVALID, got %s: %s", shader_fault, message);
    assert(vk::shader_reflection_validation_count(state) == reflections_before + 1,
        "one malformed shader performs one reflection validation");
    assert(vk::pipeline_shader_create_attempt_count(state) == native_attempts_before,
        "reflection rejection precedes native shader creation");
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

fn void shader_without_push_block_accepted() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_NO_PUSH_SPIRV[..], .entry_point = "main",
    };
    gpu::PipelineHandle pipeline = create_reflection_compute_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(pipeline.is_valid(), "a selected entry without a push block is accepted");
}

fn void malformed_root_member_shapes_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_SIGNED_SPIRV[..],
        entry_point: "main",
        message:     "a signed 64-bit root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_U32_SPIRV[..],
        entry_point: "main",
        message:     "a 32-bit root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_FLOAT64_SPIRV[..],
        entry_point: "main",
        message:     "a 64-bit floating root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_VECTOR_SPIRV[..],
        entry_point: "main",
        message:     "a vector root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_MATRIX_SPIRV[..],
        entry_point: "main",
        message:     "a matrix root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_ARRAY_SPIRV[..],
        entry_point: "main",
        message:     "an array root member faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_NESTED_SPIRV[..],
        entry_point: "main",
        message:     "a nested root member faults SHADER_INVALID",
    );
    assert_fixture_root_flag(
        REFLECT_BOOL_SPIRV[..],
        spvreflect::TypeFlags.TYPE_FLAG_BOOL,
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_BOOL_SPIRV[..],
        entry_point: "main",
        message:     "a bool root member faults SHADER_INVALID",
    );
    assert_fixture_root_flag(
        REFLECT_REF_SPIRV[..],
        spvreflect::TypeFlags.TYPE_FLAG_REF,
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_REF_SPIRV[..],
        entry_point: "main",
        message:     "a buffer-reference root member faults SHADER_INVALID",
    );
    assert_fixture_root_block_shape(
        REFLECT_MEMBER_COUNT_SPIRV[..],
        8,
        2,
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_MEMBER_COUNT_SPIRV[..],
        entry_point: "main",
        message:     "an exact-size two-member root faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_WRONG_OFFSET_SPIRV[..],
        entry_point: "main",
        message:     "a nonzero root block offset faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_MULTIPLE_BLOCKS_SPIRV[..],
        entry_point: "main",
        message:     "multiple root blocks fault SHADER_INVALID",
    );
}

fn void selected_entry_interfaces_are_isolated() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong reflections_before = vk::shader_reflection_validation_count(state);
    ulong native_attempts_before = vk::pipeline_shader_create_attempt_count(state);

    gpu::ShaderDesc good_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = REFLECT_MULTI_ENTRY_SPIRV[..],
        .entry_point = "good",
    };
    gpu::PipelineHandle pipeline =
        create_reflection_compute_pipeline(&device, &good_desc)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(vk::shader_reflection_validation_count(state) == reflections_before + 1,
        "the selected valid entry is reflected once");
    assert(vk::pipeline_shader_create_attempt_count(state) == native_attempts_before + 1,
        "unrelated invalid entries do not block native creation");

    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_MULTI_ENTRY_SPIRV[..],
        entry_point: "bad_descriptor",
        message:     "the selected entry's foreign descriptor set faults SHADER_INVALID",
    );
    assert_shader_interface_rejected(
        device:      &device,
        spirv:       REFLECT_MULTI_ENTRY_SPIRV[..],
        entry_point: "bad_push",
        message:     "the selected entry's wrong root shape faults SHADER_INVALID",
    );
}

fn void invalid_deduplicated_shader_is_reflected_once_transactionally() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = REFLECT_SIGNED_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc[2] descs = {
        { .shader = shader },
        { .shader = shader },
    };
    gpu::PipelineHandle[2] outputs = {
        { .owner = 1, .index = 2, .generation = 3 },
        { .owner = 4, .index = 5, .generation = 6 },
    };
    gpu::PipelineHandle[2] before = outputs;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong reflections_before = vk::shader_reflection_validation_count(state);
    ulong native_attempts_before = vk::pipeline_shader_create_attempt_count(state);
    uint cache_entries_before = vk::pipeline_cache_entry_count(state);

    fault batch_fault = @catch(gpu::create_compute_pipelines(
        &device,
        descs[..],
        outputs[..],
    ));

    assert(batch_fault == gpu::SHADER_INVALID,
        "one invalid deduplicated shader identity faults SHADER_INVALID");
    assert(outputs[0] == before[0] && outputs[1] == before[1],
        "invalid deduplicated reflection leaves every output unchanged");
    assert(vk::pipeline_cache_entry_count(state) == cache_entries_before,
        "invalid deduplicated reflection publishes no cache entry");
    assert(vk::shader_reflection_validation_count(state) == reflections_before + 1,
        "one invalid deduplicated shader identity is reflected once");
    assert(vk::pipeline_shader_create_attempt_count(state) == native_attempts_before,
        "invalid deduplicated reflection performs no native shader-create attempt");
}

fn void exact_compute_vertex_and_fragment_roots_accepted() @test {
    check_fixture_root_abi(
        spirv:       REFLECT_BUFFERS_SPIRV[..],
        stage:       gpu::ShaderStage.COMPUTE,
        entry_point: "main",
    )!!;
    check_fixture_root_abi(
        spirv:       REFLECT_VERTEX_SPIRV[..],
        stage:       gpu::ShaderStage.VERTEX,
        entry_point: "main",
    )!!;
    check_fixture_root_abi(
        spirv:       REFLECT_FRAGMENT_SPIRV[..],
        stage:       gpu::ShaderStage.FRAGMENT,
        entry_point: "main",
    )!!;
    check_fixture_root_abi(
        spirv:       REFLECT_RENAMED_SPIRV[..],
        stage:       gpu::ShaderStage.COMPUTE,
        entry_point: "main",
    )!!;
}

fn void reversed_graphics_member_order_rejected() @test {
    fault order_fault = @catch(check_fixture_root_abi(
        spirv:       REFLECT_REVERSED_GRAPHICS_SPIRV[..],
        stage:       gpu::ShaderStage.VERTEX,
        entry_point: "main",
    ));
    assert(order_fault == gpu::SHADER_INVALID,
        "graphics root members with reversed offsets fault SHADER_INVALID");
}

fn void graphics_size_8_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    assert_reflected_shader_rejected(
        device:  &device,
        spirv:   REFLECT_GRAPHICS_SIZE_8_SPIRV[..],
        stage:   gpu::ShaderStage.VERTEX,
        message: "an 8-byte graphics root faults SHADER_INVALID",
    );
}

fn void graphics_second_member_offset_4_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    assert_reflected_shader_rejected(
        device:  &device,
        spirv:   REFLECT_GRAPHICS_SECOND_OFFSET_4_SPIRV[..],
        stage:   gpu::ShaderStage.VERTEX,
        message: "a graphics second member at offset 4 faults SHADER_INVALID",
    );
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
    assert(entry_fault == gpu::SHADER_INVALID,
        "a requested entry point absent from the module faults SHADER_INVALID");
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

fn void reflected_stage_mismatch_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong native_attempts_before = vk::pipeline_shader_create_attempt_count(state);

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_VERTEX_SPIRV[..], .entry_point = "main",
    };
    fault stage_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(stage_fault == gpu::SHADER_INVALID,
        "a reflected execution model mismatch faults SHADER_INVALID");
    assert(vk::pipeline_shader_create_attempt_count(state) == native_attempts_before,
        "reflected stage rejection precedes native shader creation");
}

fn void vertex_module_reflected_as_fragment_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    assert_reflected_shader_rejected(
        device:  &device,
        spirv:   REFLECT_VERTEX_SPIRV[..],
        stage:   gpu::ShaderStage.FRAGMENT,
        message: "a vertex module declared as fragment faults SHADER_INVALID",
    );
}

fn void fragment_module_reflected_as_vertex_rejected_before_native_creation() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    assert_reflected_shader_rejected(
        device:  &device,
        spirv:   REFLECT_FRAGMENT_SPIRV[..],
        stage:   gpu::ShaderStage.VERTEX,
        message: "a fragment module declared as vertex faults SHADER_INVALID",
    );
}

fn void oversized_push_constant_rejected() @test {
    gpu::Device device = make_reflection_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = REFLECT_OVERSIZED_PUSH_SPIRV[..], .entry_point = "main",
    };
    fault push_fault = @catch(create_reflection_compute_pipeline(&device, &desc));
    assert(push_fault == gpu::SHADER_INVALID,
        "a push-constant block past the root-push ABI faults SHADER_INVALID");
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
    assert(entry_fault == gpu::SHADER_INVALID, "missing entry faults SHADER_INVALID");
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
    assert(abi_fault == gpu::SHADER_INVALID, "oversized ABI faults SHADER_INVALID");
    assert_shader_rejection(
        capture:       &capture,
        before:        before,
        returned:      abi_fault,
        operation:     "create_compute_pipeline",
        field:         "ComputePipelineDesc.shader",
        invariant:     "selected-entry push constants must exactly match the root ABI",
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
    gpu::ComputePipelineDesc pipeline_desc = { .shader = vertex };
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
