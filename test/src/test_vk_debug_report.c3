module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem;
import std::thread;

// Headless (lavapipe): leak reporters, debug-name storage, stats counts, and
// command labels. The leak test intentionally prints leak lines to stderr.

const char[*] DBG_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

struct DebugCapture {
    uint              calls;
    gpu::DebugMessage last;
    bool              validation_strings_match;
    bool              span_strings_match;
    bool              device_name_matches;
    bool              messenger_failure_strings_match;
}

struct SamplerNameCapture {
    uint           calls;
    vk::ObjectType object_type;
    ulong          object_handle;
    bool           first_name_matches;
}

SamplerNameCapture* sampler_name_capture;

fn vk::Result capture_sampler_name(
    vk::Device device,
    vk::DebugUtilsObjectNameInfoEXT* info,
) {
    SamplerNameCapture* capture = sampler_name_capture;
    if (capture == null || device == null || info == null) {
        return vk::Result.ERROR_UNKNOWN;
    }
    capture.calls++;
    capture.object_type = info.object_type;
    capture.object_handle = info.object_handle;
    capture.first_name_matches = info.p_object_name != null
        && ((ZString)info.p_object_name).str_view() == "first_sampler";
    return vk::Result.SUCCESS;
}

fn void capture_debug_message(gpu::DebugMessage* message, void* user_data) {
    DebugCapture* capture = (DebugCapture*)user_data;
    capture.calls++;
    capture.last = *message;
    if (message.validation_id_name != null
        && message.validation_id_name.str_view() == "VUID-Test-123") {
        capture.validation_strings_match = message.backend_text != null
            && message.backend_text.str_view() == "fabricated validation message"
            && message.resource.debug_name != null
            && message.resource.debug_name.str_view() == "named_gpu_span";
    }
    if (message.operation != null && message.operation.str_view() == "cmd_buffer_barrier") {
        capture.span_strings_match = message.resource.debug_name != null
            && message.resource.debug_name.str_view() == "debug_span"
            && message.rejected_field != null
            && message.rejected_field.str_view() == "BufferBarrier.span"
            && message.invariant != null
            && message.invariant.str_view() == "span range must fit its live owner";
    }
    if (message.resource.kind == gpu::DebugResourceKind.DEVICE && message.resource.debug_name != null) {
        capture.device_name_matches = message.resource.debug_name.str_view() == "named_device";
    }
    if (message.backend_text != null && message.backend_text.str_view() == "fabricated messenger failure") {
        capture.messenger_failure_strings_match = message.operation != null
            && message.operation.str_view() == "create_device";
    }
}

struct LeakCapture {
    uint total;
    uint allocations;
    uint gpu_spans;
    uint textures;
    uint pipelines;
    uint shaders;

    uint texture_descriptors;
    uint samplers;
    uint public_identities;
    bool metadata_valid;
    gpu::PipelineHandle expected_pipeline_a;
    gpu::PipelineHandle expected_pipeline_b;
    bool pipeline_identity_valid;
    bool private_span_identity_hidden;
    bool private_span_wording_valid;
    bool named_resources_valid;
}

fn void capture_leak_message(gpu::DebugMessage* message, void* user_data) {
    LeakCapture* capture = (LeakCapture*)user_data;
    capture.total++;
    bool operation_valid = message.operation != null && message.operation.str_view() == "destroy_device";
    bool invariant_valid = message.invariant != null
        && message.invariant.str_view() == "resource remains live at device destruction";
    capture.metadata_valid = capture.metadata_valid
        && message.severity == gpu::DebugMessageSeverity.WARNING
        && message.category.resource_lifetime
        && !message.has_fault
        && operation_valid
        && invariant_valid;
    if (message.resource.has_public_identity) {
        capture.public_identities++;
        capture.metadata_valid = capture.metadata_valid
            && message.resource.generation != 0;
    }
    switch (message.resource.kind) {
        case gpu::DebugResourceKind.ALLOCATION:
            capture.allocations++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_allocation";
        case gpu::DebugResourceKind.GPU_SPAN:
            capture.gpu_spans++;
            capture.private_span_identity_hidden = !message.resource.has_public_identity
                && message.resource.index == 0
                && message.resource.generation == 0;
            capture.private_span_wording_valid = message.backend_text != null
                && message.backend_text.str_view() == "GPU span storage remains live";
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_gpu_span_storage";
        case gpu::DebugResourceKind.TEXTURE:
            capture.textures++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_texture";
        case gpu::DebugResourceKind.PIPELINE:
            capture.pipelines++;
            bool first = message.resource.index == capture.expected_pipeline_a.index
                && message.resource.generation == capture.expected_pipeline_a.generation;
            bool second = message.resource.index == capture.expected_pipeline_b.index
                && message.resource.generation == capture.expected_pipeline_b.generation;
            capture.pipeline_identity_valid = message.resource.has_public_identity
                && (first || second);
        case gpu::DebugResourceKind.SHADER:             capture.shaders++;

        case gpu::DebugResourceKind.TEXTURE_DESCRIPTOR: capture.texture_descriptors++;
        case gpu::DebugResourceKind.SAMPLER:            capture.samplers++;
        default: capture.metadata_valid = false;
    }
}

struct ConcurrentDebugCapture {
    Mutex mutex;
    uint  calls;
    uint  warnings;
    uint  errors;
}

struct DebugRouteArgs {
    vk::VkDeviceState* state;
    vk::DebugUtilsMessageSeverityFlagsEXT severity;
    uint messages;
}

fn void capture_concurrent_debug_message(gpu::DebugMessage* message, void* user_data) {
    ConcurrentDebugCapture* capture = (ConcurrentDebugCapture*)user_data;
    capture.mutex.lock()!!;
    defer capture.mutex.unlock();
    capture.calls++;
    switch (message.severity) {
        case gpu::DebugMessageSeverity.WARNING: capture.warnings++;
        case gpu::DebugMessageSeverity.ERROR:   capture.errors++;
        default:
    }
}

fn int route_debug_messages_worker(void* user_data) {
    DebugRouteArgs* args = (DebugRouteArgs*)user_data;
    vk::DebugUtilsMessengerCallbackDataEXT data = vk::debug_utils_messenger_callback_data_ext();
    data.p_message_id_name = (char*)"VUID-Concurrent-Test";
    data.p_message = (char*)"concurrent fabricated message";
    for (uint i = 0; i < args.messages; i++) {
        data.message_id_number = (int)i;
        uint result = vk::route_debug_message(
            state:        args.state,
            severity:     args.severity,
            message_type: (vk::DebugUtilsMessageTypeFlagsEXT)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT,
            data:         &data,
        );
        if (result != vk::FALSE) return 1;
    }
    return 0;
}

fn void vulkan_debug_message_delivery_uses_consumer_synchronization() @test {
    const uint MESSAGES_PER_THREAD = 32;
    ConcurrentDebugCapture capture;
    capture.mutex.init()!!;
    defer (void)capture.mutex.destroy();
    vk::VkDeviceState state = {
        .debug_callback  = &capture_concurrent_debug_message,
        .debug_user_data = &capture,
    };
    DebugRouteArgs[2] args = {
        { .state = &state, .severity = (vk::DebugUtilsMessageSeverityFlagsEXT)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT, .messages = MESSAGES_PER_THREAD },
        { .state = &state, .severity = (vk::DebugUtilsMessageSeverityFlagsEXT)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT, .messages = MESSAGES_PER_THREAD },
    };
    Thread[2] threads;
    threads[0].create(&route_debug_messages_worker, &args[0])!!;
    threads[1].create(&route_debug_messages_worker, &args[1])!!;

    assert(threads[0].join()!! == 0 && threads[1].join()!! == 0, "both routing workers return VK_FALSE throughout");
    assert(capture.calls == 2 * MESSAGES_PER_THREAD, "consumer-owned mutex preserves every concurrent delivery");
    assert(capture.warnings == MESSAGES_PER_THREAD, "warning deliveries remain exact under concurrency");
    assert(capture.errors == MESSAGES_PER_THREAD, "error deliveries remain exact under concurrency");
}

fn void vulkan_debug_message_routes_structured_fields() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };
    vk::DebugUtilsObjectNameInfoEXT object = vk::debug_utils_object_name_info_ext();
    object.object_type = vk::ObjectType.BUFFER;
    object.object_handle = 0x1234;
    object.p_object_name = (char*)"named_gpu_span";
    vk::DebugUtilsMessengerCallbackDataEXT data = vk::debug_utils_messenger_callback_data_ext();
    data.p_message_id_name = (char*)"VUID-Test-123";
    data.message_id_number = 2718;
    data.p_message = (char*)"fabricated validation message";
    data.object_count = 1;
    data.p_objects = &object;

    uint severity = (uint)vk::DebugUtilsMessageSeverityFlagBitsEXT.ERROR_BIT_EXT;
    uint types = (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT
        | (uint)vk::DebugUtilsMessageTypeFlagBitsEXT.PERFORMANCE_BIT_EXT;
    uint result = vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)severity,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)types,
        data:         &data,
    );

    assert(result == vk::FALSE, "debug callback never requests Vulkan abort");
    assert(capture.calls == 1, "structured callback receives exactly one message");
    assert(capture.last.severity == gpu::DebugMessageSeverity.ERROR, "error severity is preserved");
    assert(capture.last.category.backend, "Vulkan diagnostics are backend messages");
    assert(capture.last.category.validation, "validation category is preserved");
    assert(capture.last.category.performance, "performance category is preserved");
    assert(!capture.last.category.general, "unset general category stays clear");
    assert(capture.last.validation_id_number == 2718, "numeric validation ID is preserved");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.GPU_SPAN,
        "native Vulkan buffer maps to the semantic GPU span kind");
    assert(capture.validation_strings_match, "callback observes borrowed validation ID, text, and object name while valid");
    assert(!capture.last.resource.has_public_identity, "native callback data invents no public handle identity");
    assert(capture.last.resource.index == 0 && capture.last.resource.generation == 0,
        "native object handles never leak into public index or generation fields");

    object.object_type = vk::ObjectType.DEVICE;
    object.object_handle = 0x5678;
    object.p_object_name = (char*)"named_device";
    data.p_message_id_name = (char*)"VUID-Device-Test";
    data.message_id_number = 3141;
    vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)severity,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)types,
        data:         &data,
    );
    assert(capture.calls == 2, "fabricated device message is routed");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.DEVICE, "Vulkan device maps to public device resource kind");
    assert(capture.device_name_matches, "callback observes the borrowed device name while valid");
    assert(!capture.last.resource.has_public_identity, "native device handle does not become a public identity");
}

fn void vulkan_debug_message_prefers_named_resource_and_falls_back_to_kind() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };
    vk::DebugUtilsObjectNameInfoEXT[2] objects = {
        vk::debug_utils_object_name_info_ext(),
        vk::debug_utils_object_name_info_ext(),
    };
    objects[0].object_type = vk::ObjectType.BUFFER;
    objects[0].object_handle = 0x1234;
    objects[1].object_type = vk::ObjectType.IMAGE;
    objects[1].object_handle = 0x5678;
    objects[1].p_object_name = (char*)"later_texture";
    vk::DebugUtilsMessengerCallbackDataEXT data = vk::debug_utils_messenger_callback_data_ext();
    data.p_message = (char*)"resource ordering";
    data.object_count = 2;
    data.p_objects = &objects[0];

    vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT,
        data:         &data,
    );

    assert(capture.last.resource.kind == gpu::DebugResourceKind.TEXTURE,
        "later named object wins over an earlier unnamed mapped object");
    assert(capture.last.resource.debug_name.str_view() == "later_texture",
        "selected resource keeps the first useful name");

    objects[1].p_object_name = null;
    vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)vk::DebugUtilsMessageSeverityFlagBitsEXT.WARNING_BIT_EXT,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)vk::DebugUtilsMessageTypeFlagBitsEXT.VALIDATION_BIT_EXT,
        data:         &data,
    );

    assert(capture.last.resource.kind == gpu::DebugResourceKind.GPU_SPAN,
        "all-unnamed objects fall back to the first mapped kind");
    assert(capture.last.resource.debug_name == null, "fallback resource has no invented name");
}

fn void vulkan_debug_message_handles_null_callback_data() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };

    uint result = vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)vk::DebugUtilsMessageSeverityFlagBitsEXT.INFO_BIT_EXT,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)vk::DebugUtilsMessageTypeFlagBitsEXT.GENERAL_BIT_EXT,
        data:         null,
    );

    assert(result == vk::FALSE, "null callback data still returns VK_FALSE");
    assert(capture.calls == 1, "null callback data still routes one message");
    assert(capture.last.severity == gpu::DebugMessageSeverity.INFO, "info severity is preserved");
    assert(capture.last.category.backend && capture.last.category.general, "general backend category is preserved");
    assert(capture.last.backend_text == null, "missing text remains absent");
    assert(capture.last.validation_id_name == null, "missing validation ID name remains absent");
    assert(capture.last.validation_id_number == 0, "missing numeric validation ID remains zero");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.NONE, "missing objects produce no resource kind");
    assert(!capture.last.resource.has_public_identity, "missing objects produce no public identity");
}

fn void allocation_span_diagnostic_preserves_fault() @test {
    DebugCapture capture;
    gpu::DeviceDesc device_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .debug_callback   = &capture_debug_message,
        .debug_user_data  = &capture,
        .application_name = "gpu_debug_contract_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "debug_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::BufferBarrier barrier = {
        .span          = owned.span.unchecked_subspan(48, 32),
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
    };

    uint before = capture.calls;
    fault rejected = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));

    assert(rejected == gpu::INVALID_ARGUMENT, "structured diagnostics do not change the public fault");
    assert(capture.calls == before + 1, "one public-contract message accompanies the fault");
    assert(capture.last.severity == gpu::DebugMessageSeverity.ERROR, "public rejection is an error");
    assert(capture.last.category.public_contract, "public rejection carries the contract category");
    assert(capture.last.has_fault && capture.last.public_fault == gpu::INVALID_ARGUMENT,
        "diagnostic carries the same public fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.ALLOCATION,
        "rejected allocation resource kind is identified");
    assert(capture.last.resource.has_public_identity,
        "rejected allocation preserves its public identity");
    assert(capture.last.resource.index == owned.span.index
        && capture.last.resource.generation == owned.span.generation,
        "rejected allocation preserves its slot identity");
    assert(capture.span_strings_match,
        "callback observes borrowed allocation/span metadata while valid");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void contract_failure_helper_preserves_fault_and_dispatches_once() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };

    fault returned = vk::report_contract_failure(
        state:               &state,
        operation:           "cmd_buffer_barrier",
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               7,
        generation:          9,
        debug_name:          "debug_span",
        rejected_field:      "BufferBarrier.span",
        invariant:           "span range must fit its live owner",
        backend_text:        null,
    );

    assert(returned == gpu::INVALID_ARGUMENT, "helper returns the unchanged public fault");
    assert(capture.calls == 1, "helper dispatches exactly once");
    assert(capture.last.has_fault && capture.last.public_fault == returned,
        "structured message carries the returned fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.ALLOCATION,
        "helper preserves resource kind");
    assert(capture.last.resource.has_public_identity
        && capture.last.resource.index == 7
        && capture.last.resource.generation == 9,
        "helper preserves public slot identity");
    assert(capture.span_strings_match, "helper exposes borrowed contract strings only during callback delivery");
}

fn void debug_messenger_failure_routes_backend_warning() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };

    vk::report_debug_messenger_failure(&state, "fabricated messenger failure");

    assert(capture.calls == 1, "messenger setup failure routes exactly once");
    assert(capture.last.severity == gpu::DebugMessageSeverity.WARNING, "messenger setup failure is a warning");
    assert(capture.last.category.backend, "messenger setup failure is a backend diagnostic");
    assert(!capture.last.has_fault, "best-effort messenger setup failure carries no public fault");
    assert(capture.messenger_failure_strings_match,
        "callback observes borrowed create-device operation and failure text while valid");
}

fn void debug_names_do_not_require_validation() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = false,
        .enable_debug_names = true,
        .application_name  = "gpu_debug_names_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    assert(state.debug_names, "backend retains the debug-name opt-in without validation");

    vk::BufferDesc buffer_desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "names_without_validation",
    };
    vk::BufferHandle buffer = vk::vk_create_buffer(&device, &buffer_desc)!!;
    vk::vk_destroy_buffer(&device, buffer)!!;
}

fn void interned_sampler_keeps_first_debug_name() @test {
    gpu::DeviceDesc device_desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .enable_debug_names = true,
        .application_name   = "gpu_sampler_name_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    SamplerNameCapture capture;
    sampler_name_capture = &capture;
    vk::PFN_vkSetDebugUtilsObjectNameEXT original =
        state.device_dispatch.debug.set_object_name;
    state.device_dispatch.debug.set_object_name = &capture_sampler_name;
    defer {
        state.device_dispatch.debug.set_object_name = original;
        sampler_name_capture = null;
    }

    gpu::SamplerDesc first_desc = { .debug_name = "first_sampler" };
    gpu::Sampler first = gpu::intern_sampler(&device, &first_desc)!!;
    gpu::SamplerDesc duplicate_desc = first_desc;
    duplicate_desc.debug_name = "later_sampler";
    gpu::Sampler duplicate = gpu::intern_sampler(&device, &duplicate_desc)!!;

    assert(first == duplicate, "debug names do not change sampler identity");
    assert(capture.calls == 1, "only the first interned sampler assigns a native name");
    assert(capture.object_type == vk::ObjectType.SAMPLER,
        "the native name identifies a Vulkan sampler");
    assert(capture.object_handle != 0, "the native name identifies the live object");
    assert(capture.first_name_matches, "the first semantic description owns the debug name");
}

fn gpu::Device make_dbg_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_debug_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn void leak_reporters_count_leaks() @test {
    gpu::Device device = make_dbg_device();
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    vk::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "leaky_gpu_span_storage",
    };
    vk::BufferHandle buffer = vk::vk_create_buffer(&device, &buf_desc)!!;

    gpu::AllocationDesc allocation_desc = {
        .size         = 256,
        .alignment    = 64,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "leaky_allocation",
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(
        &device,
        &allocation_desc,
    )!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "leaky_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DBG_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &comp_sd)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = 8 };
    gpu::PipelineHandle first_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    gpu::PipelineHandle second_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;


    gpu::TextureIndex texture_index = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex sampler_index = gpu::publish_sampler(&device, sampler)!!;

    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
        .expected_pipeline_a   = first_alias,
        .expected_pipeline_b   = second_alias,
    };
    state.debug_callback = &capture_leak_message;
    state.debug_user_data = &capture;

    assert(vk::report_buffer_leaks(state) == 1, "one private GPU span backing leak counted");
    assert(vk::report_allocation_leaks(state) == 1,
        "one leaked allocation counted");
    assert(vk::report_texture_leaks(state) == 1, "one leaked texture counted");
    assert(vk::report_pipeline_leaks(state) == 1, "one leaked cache entry counted");
    assert(vk::report_shader_leaks(state) == 1, "one leaked shader counted");

    vk::report_descriptor_leaks(state);

    assert(capture.total == 6, "each leaked resource or owned descriptor emits one warning");
    assert(capture.allocations == 1 && capture.gpu_spans == 1
        && capture.textures == 1 && capture.pipelines == 1,
        "allocation, GPU span, texture, and pipeline leak kinds are exact");
    assert(capture.shaders == 1,
        "private recording pools are not public resource leaks");
    assert(capture.texture_descriptors == 1 && capture.samplers == 0,
        "device-owned samplers are not descriptor leaks");
    assert(capture.public_identities == 5,
        "only strict public resources carry slot identities");
    assert(capture.private_span_identity_hidden,
        "private backing storage exposes no private handle identity");
    assert(capture.private_span_wording_valid,
        "private backing storage uses semantic GPU span wording");
    assert(capture.pipeline_identity_valid, "pipeline cache leak identifies one live public alias");
    assert(capture.metadata_valid, "leak diagnostics are warning, resource-lifetime, faultless destroy-device messages");
    assert(capture.named_resources_valid, "slot-owned storage and texture names are observed during the callback");

    // Slot data feeding the report lines.
    vk::TextureSlot* tex_slot = &state.textures.slots[texture.index].value;
    assert(tex_slot.allocation_size > 0, "texture leak line carries the allocation size");

    gpu::destroy_texture_descriptor(&device, texture_index)!!;
    gpu::destroy_pipeline(&device, first_alias)!!;
    gpu::destroy_pipeline(&device, second_alias)!!;
    gpu::destroy_shader(&device, shader)!!;
    gpu::destroy_texture(&device, texture)!!;
    vk::vk_destroy_buffer(&device, buffer)!!;
    gpu::free_allocation(&device, &allocation)!!;

    assert(vk::report_resource_leaks(state) == 0, "clean state reports zero leaks");
    assert(capture.total == 6, "clean leak report emits nothing");
    gpu::destroy_device(&device)!!;
}

fn void debug_name_copy_is_independent() @test {
    gpu::Device device = make_dbg_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    char[16] volatile_name;
    char[7] source = { 'v', 'o', 'l', 'a', 't', 'i', 'l' };
    for (uint i = 0; i < 7; i++) volatile_name[i] = source[i];
    volatile_name[7] = 0;

    vk::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = (ZString)&volatile_name[0],
    };
    vk::BufferHandle buffer = vk::vk_create_buffer(&device, &desc)!!;
    defer vk::vk_destroy_buffer(&device, buffer)!!;

    foreach (&b : volatile_name) *b = 'X';

    char* stored = &state.buffers.slots[buffer.index].value.debug_name[0];
    assert(stored[0] == 'v' && stored[6] == 'l' && stored[7] == 0,
        "the slot holds its own copy, unaffected by the caller's buffer");
}

fn void stats_cover_public_allocations_textures_and_budgets() @test {
    gpu::Device device = make_dbg_device();
    defer destroy_test_device(&device)!!;

    gpu::MemoryStats baseline = gpu::get_memory_stats(&device)!!;
    gpu::GpuAllocation[3] allocations;
    for (uint i = 0; i < 3; i++) {
        gpu::AllocationDesc desc = {
            .size         = 128,
            .alignment    = 16,
            .memory_class = gpu::MemoryClass.GPU_PRIVATE,
            .access       = { .graphics, .compute, .transfer },
        };
        allocations[i] = gpu::allocate_memory(&device, &desc)!!;
    }
    gpu::MemoryStats allocated = gpu::get_memory_stats(&device)!!;
    assert(allocated.live_allocation_count
        >= baseline.live_allocation_count + (ulong)allocations.len,
        "public allocations contribute to live allocation statistics");

    gpu::TextureHandle[2] textures;
    for (uint i = 0; i < 2; i++) {
        gpu::TextureDesc desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .access    = { .graphics, .compute, .transfer },
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled },
        };
        textures[i] = gpu::create_texture(&device, &desc)!!;
    }

    gpu::MemoryStats stats = gpu::get_memory_stats(&device)!!;
    assert(stats.texture_count == baseline.texture_count + (ulong)textures.len,
        "public texture count reflects live textures");
    assert(stats.heap_count > 0 && stats.heaps[0].budget > 0,
        "memory stats include a valid heap budget");

    foreach (texture : textures) gpu::destroy_texture(&device, texture)!!;
    for (uint i = 0; i < allocations.len; i++) {
        gpu::free_allocation(&device, &allocations[i])!!;
    }
    stats = gpu::get_memory_stats(&device)!!;
    assert(stats.texture_count == baseline.texture_count,
        "destroyed public textures leave no live texture count");
}

fn void labels_record_validation_clean() @test {
    gpu::Device device = make_dbg_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::cmd_begin_label(&cmd, "resource_setup")!!;
    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::cmd_end_label(&cmd)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_begin_label(&cmd, "main_pass", { 0.2f, 0.8f, 0.2f, 1.0f })!!;
    gpu::cmd_begin_label(&cmd, "nested")!!;
    gpu::cmd_end_label(&cmd)!!;
    gpu::cmd_end_label(&cmd)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    fault ended = @catch(gpu::cmd_begin_label(&cmd, "too_late"));
    assert(ended == gpu::INVALID_HANDLE, "labels on a consumed list fault INVALID_HANDLE");
    fault end_stale = @catch(gpu::cmd_end_label(&cmd));
    assert(end_stale == gpu::INVALID_HANDLE, "ending a label on a consumed list faults INVALID_HANDLE");
}

fn void live_backend_resource_is_reported_before_device_teardown() @test {
    gpu::Device device = make_dbg_device();
    vk::BufferDesc desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "canary_buffer",
    };
    vk::BufferHandle buffer = vk::vk_create_buffer(&device, &desc)!!;
    assert(buffer.is_valid(), "private backend storage is live");
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 1, "the live private backing is reported");
    gpu::destroy_device(&device)!!;
}

fn void callback_reports_leaks_without_validation() @test {
    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
    };
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .debug_callback    = &capture_leak_message,
        .debug_user_data   = &capture,
        .application_name  = "gpu_callback_leak_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&device_desc)!!;
    vk::BufferDesc buffer_desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "leaky_gpu_span_storage",
    };
    vk::BufferHandle buffer = vk::vk_create_buffer(&device, &buffer_desc)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 1, "the live private backing is reported");

    assert(capture.total == 1 && capture.gpu_spans == 1,
        "callback receives one private-storage warning without validation");
    assert(capture.metadata_valid && capture.named_resources_valid,
        "the report preserves structured leak metadata");
    assert(capture.private_span_identity_hidden,
        "private backing storage exposes no private handle identity");
    assert(capture.private_span_wording_valid,
        "private backing storage uses semantic GPU span wording");
    vk::vk_destroy_buffer(&device, buffer)!!;
    gpu::destroy_device(&device)!!;
}
