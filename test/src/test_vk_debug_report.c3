module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
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
    bool              barrier_strings_match;
    bool              device_name_matches;
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
    if (message.operation != null && message.operation.str_view() == "cmd_barrier") {
        capture.barrier_strings_match = message.resource.debug_name == null
            && message.rejected_field != null
            && message.rejected_field.str_view() == "Barrier.before"
            && message.invariant != null
            && message.invariant.str_view()
                == "stage mask must be nonempty and noncontradictory";
    }
    if (message.resource.kind == gpu::DebugResourceKind.DEVICE && message.resource.debug_name != null) {
        capture.device_name_matches = message.resource.debug_name.str_view() == "named_device";
    }
}

struct LeakCapture {
    uint total;
    uint allocations;
    uint textures;
    uint pipelines;
    uint shaders;
    uint command_allocators;
    uint timestamp_pools;
    uint acceleration_structures;
    uint acceleration_structure_views;

    uint texture_descriptors;
    uint samplers;
    uint public_identities;
    bool metadata_valid;
    gpu::PipelineHandle expected_pipeline_a;
    gpu::PipelineHandle expected_pipeline_b;
    gpu::AccelerationStructureHandle expected_acceleration_structure_a;
    gpu::AccelerationStructureHandle expected_acceleration_structure_b;
    uint expected_acceleration_structure_view_index;
    uint expected_acceleration_structure_view_generation;
    bool pipeline_identity_valid;
    bool acceleration_structure_identity_valid;
    bool acceleration_structure_a_seen;
    bool acceleration_structure_b_seen;
    bool acceleration_structure_view_identity_valid;
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
        case gpu::DebugResourceKind.COMMAND_ALLOCATOR:
            capture.command_allocators++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_command_allocator";
        case gpu::DebugResourceKind.TIMESTAMP_POOL:
            capture.timestamp_pools++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_timestamp_pool";
        case gpu::DebugResourceKind.ACCELERATION_STRUCTURE:
            capture.acceleration_structures++;
            bool first_acceleration_structure =
                message.resource.index
                    == capture.expected_acceleration_structure_a.index
                && message.resource.generation
                    == capture.expected_acceleration_structure_a.generation;
            bool second_acceleration_structure =
                message.resource.index
                    == capture.expected_acceleration_structure_b.index
                && message.resource.generation
                    == capture.expected_acceleration_structure_b.generation;
            capture.acceleration_structure_a_seen =
                capture.acceleration_structure_a_seen
                || first_acceleration_structure;
            capture.acceleration_structure_b_seen =
                capture.acceleration_structure_b_seen
                || second_acceleration_structure;
            capture.acceleration_structure_identity_valid =
                capture.acceleration_structure_identity_valid
                && message.resource.has_public_identity
                && (first_acceleration_structure
                    || second_acceleration_structure);
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view()
                    == "leaky_acceleration_structure";
        case gpu::DebugResourceKind.ACCELERATION_STRUCTURE_VIEW:
            capture.acceleration_structure_views++;
            capture.acceleration_structure_view_identity_valid =
                message.resource.has_public_identity
                && message.resource.index
                    == capture.expected_acceleration_structure_view_index
                && message.resource.generation
                    == capture.expected_acceleration_structure_view_generation;

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

fn void runtime_debug_messenger_failure_routes_structured_warning() @test {
    DebugCapture capture;
    vk::VkRuntimeState state = {
        .config = {
            .debug_callback  = &capture_debug_message,
            .debug_user_data = &capture,
        },
    };

    vk::report_runtime_debug_messenger_failure(&state);

    assert(capture.calls == 1,
        "runtime messenger failure reaches the configured callback exactly once");
    assert(capture.last.severity == gpu::DebugMessageSeverity.WARNING,
        "runtime messenger failure is a warning");
    assert(capture.last.category.backend,
        "runtime messenger failure carries the backend category");
    assert(capture.last.operation != null
        && capture.last.operation.str_view() == "create_runtime",
        "runtime messenger failure identifies create_runtime");
    assert(capture.last.backend_text != null
        && capture.last.backend_text.str_view()
            == "debug-utils messenger unavailable",
        "runtime messenger failure preserves its backend diagnostic");
    assert(!capture.last.has_fault,
        "runtime messenger failure does not invent a public fault");
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
    object.object_type = vk::ObjectType.ACCELERATION_STRUCTURE_KHR;
    object.object_handle = 0x9abc;
    object.p_object_name = (char*)"named_acceleration_structure";
    data.p_message_id_name = (char*)"VUID-AccelerationStructure-Test";
    data.message_id_number = 1618;
    vk::route_debug_message(
        state:        &state,
        severity:     (vk::DebugUtilsMessageSeverityFlagsEXT)severity,
        message_type: (vk::DebugUtilsMessageTypeFlagsEXT)types,
        data:         &data,
    );
    assert(capture.calls == 3,
        "fabricated acceleration-structure message is routed");
    assert(capture.last.resource.kind
            == gpu::DebugResourceKind.ACCELERATION_STRUCTURE,
        "native Vulkan acceleration structure maps to its semantic kind");
    assert(capture.last.resource.debug_name != null
            && capture.last.resource.debug_name.str_view()
                == "named_acceleration_structure",
        "callback observes the borrowed acceleration-structure name");
    assert(!capture.last.resource.has_public_identity,
        "native acceleration-structure handle does not become a public identity");
    assert(capture.last.resource.index == 0
            && capture.last.resource.generation == 0,
        "native acceleration-structure identity remains hidden");
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

fn void barrier_mask_diagnostic_preserves_fault() @test {
    DebugCapture capture;
    gpu::RuntimeDesc device_desc = {
        .contract_validation = gpu::ContractValidation.FULL,
        .debug_callback      = &capture_debug_message,
        .debug_user_data     = &capture,
        .application_name    = "gpu_debug_contract_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = {
            .all = true,
            .compute = true,
        },
        .after = { .compute = true },
    };

    uint before = capture.calls;
    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));

    assert(rejected == gpu::INVALID_ARGUMENT,
        "structured diagnostics do not change the public fault");
    assert(capture.calls == before + 1,
        "one public-contract message accompanies the fault");
    assert(capture.last.severity == gpu::DebugMessageSeverity.ERROR,
        "public rejection is an error");
    assert(capture.last.category.public_contract,
        "public rejection carries the contract category");
    assert(capture.last.has_fault
        && capture.last.public_fault == gpu::INVALID_ARGUMENT,
        "diagnostic carries the same public fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.NONE,
        "global barrier rejection carries no resource kind");
    assert(!capture.last.resource.has_public_identity,
        "global barrier rejection carries no public identity");
    assert(capture.barrier_strings_match,
        "callback observes borrowed semantic barrier metadata while valid");

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
        operation:           "cmd_barrier",
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      "Barrier.before",
        invariant:           "stage mask must be nonempty and noncontradictory",
        backend_text:        null,
    );

    assert(returned == gpu::INVALID_ARGUMENT,
        "helper returns the unchanged public fault");
    assert(capture.calls == 1, "helper dispatches exactly once");
    assert(capture.last.has_fault && capture.last.public_fault == returned,
        "structured message carries the returned fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.NONE,
        "helper preserves the absent resource kind");
    assert(!capture.last.resource.has_public_identity,
        "helper preserves the absent resource identity");
    assert(capture.barrier_strings_match,
        "helper exposes borrowed contract strings only during callback delivery");
}


fn void debug_names_do_not_require_validation() @test {
    gpu::RuntimeDesc desc = {
        .enable_debug_names = true,
        .application_name   = "gpu_debug_names_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;

    assert(state.debug_names, "backend retains the debug-name opt-in without validation");

    gpu::AllocationDesc buffer_desc = {
        .size        = 64,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "names_without_validation",
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &buffer_desc)!!;
    gpu::free_allocation(&device, &allocation)!!;
}

fn void interned_sampler_keeps_first_debug_name() @test {
    gpu::RuntimeDesc device_desc = {
        .enable_debug_names = true,
        .application_name   = "gpu_sampler_name_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
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
    gpu::SamplerIndex first = gpu::intern_sampler(&device, &first_desc)!!;
    gpu::SamplerDesc duplicate_desc = first_desc;
    duplicate_desc.debug_name = "later_sampler";
    gpu::SamplerIndex duplicate = gpu::intern_sampler(&device, &duplicate_desc)!!;

    assert(first == duplicate, "debug names do not change sampler interning");
    assert(capture.calls == 1, "only the first interned sampler assigns a native name");
    assert(capture.object_type == vk::ObjectType.SAMPLER,
        "the native name identifies a Vulkan sampler");
    assert(capture.object_handle != 0, "the native name identifies the live object");
    assert(capture.first_name_matches, "the first semantic description owns the debug name");
}

fn gpu::Device make_dbg_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_debug_test",
    };
    return create_test_device(&desc)!!;
}

fn void leak_reporters_count_leaks() @test {
    gpu::Device device = make_dbg_device();
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;

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
        .access    = { .graphics, .compute, .transfer },
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "leaky_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;

    gpu::ShaderDesc comp_sd = { .spirv = DBG_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc shader = comp_sd;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader };
    gpu::PipelineHandle first_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    gpu::PipelineHandle second_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .debug_name = "leaky_command_allocator",
    };
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    gpu::TimestampPoolDesc timestamp_desc = {
        .capacity   = 2,
        .debug_name = "leaky_timestamp_pool",
    };
    gpu::TimestampPoolHandle timestamp_pool =
        gpu::create_timestamp_pool(&device, &timestamp_desc)!!;

    gpu::TextureView texture_index_view = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureIndex texture_index = texture_index_view.index;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex sampler_index = gpu::intern_sampler(&device, &sampler_desc)!!;
    assert(texture_index.is_valid() && sampler_index.is_valid(),
        "published shader indices are valid");

    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
        .expected_pipeline_a   = first_alias,
        .expected_pipeline_b   = second_alias,
    };
    state.debug_callback = &capture_leak_message;
    state.debug_user_data = &capture;

    assert(vk::report_allocation_leaks(state) == 1,
        "one leaked allocation counted");
    assert(vk::report_texture_leaks(state) == 1, "one leaked texture counted");
    assert(vk::report_pipeline_leaks(state) == 1, "one leaked cache entry counted");
    assert(vk::report_command_allocator_leaks(state) == 1,
        "one leaked command allocator counted");
    assert(vk::report_timestamp_pool_leaks(state) == 1,
        "one leaked timestamp pool counted");

    vk::report_descriptor_leaks(state);

    assert(capture.total == 6, "each leaked resource or owned descriptor emits one warning");
    assert(capture.allocations == 1
        && capture.textures == 1 && capture.pipelines == 1
        && capture.command_allocators == 1
        && capture.timestamp_pools == 1,
        "allocation, texture, pipeline, command allocator, and timestamp leak kinds are exact");
    assert(capture.shaders == 0,
        "borrowed shader input is not a live resource");
    assert(capture.texture_descriptors == 1 && capture.samplers == 0,
        "device-owned samplers are not descriptor leaks");
    assert(state.textures.get(texture)!!.live_descriptor_views == 1,
        "leak reporting does not decrement owner counts during table sweeps");
    assert(capture.public_identities == 6,
        "only exact public resources carry slot identities");
    assert(capture.pipeline_identity_valid, "pipeline cache leak identifies one live public alias");
    assert(capture.metadata_valid, "leak diagnostics are warning, resource-lifetime, faultless destroy-device messages");
    assert(capture.named_resources_valid, "slot-owned storage and texture names are observed during the callback");

    // Slot data feeding the report lines.
    vk::TextureSlot* tex_slot = &state.textures.slots[texture.index].value;
    assert(tex_slot.allocation_size > 0, "texture leak line carries the allocation size");

    gpu::destroy_texture_view(&device, texture_index_view)!!;
    gpu::destroy_timestamp_pool(&device, timestamp_pool)!!;
    gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::destroy_pipeline(&device, first_alias)!!;
    gpu::destroy_pipeline(&device, second_alias)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::free_allocation(&device, &allocation)!!;

    assert(vk::report_resource_leaks(state) == 0, "clean state reports zero leaks");
    assert(capture.total == 6, "clean leak report emits nothing");
    destroy_test_device(&device)!!;
}

fn void acceleration_structure_leaks_route_semantic_diagnostics() @test {
    vk::VkDeviceState state;
    state.acceleration_structures.slots =
        mem::new_array(vk::AccelerationStructureCell, 3);
    defer mem::free(state.acceleration_structures.slots.ptr);
    state.acceleration_structures.next_new = 3;

    vk::AccelerationStructureCell* first =
        &state.acceleration_structures.slots[0];
    first.used = true;
    first.generation = 17;
    vk::copy_debug_name(
        &first.value.debug_name[0],
        "leaky_acceleration_structure",
    );
    vk::AccelerationStructureCell* second =
        &state.acceleration_structures.slots[2];
    second.used = true;
    second.generation = 23;
    vk::copy_debug_name(
        &second.value.debug_name[0],
        "leaky_acceleration_structure",
    );

    vk::DescriptorHeapState* heap = &state.shader_heap.descriptors;
    heap.acceleration_structure_cells =
        mem::new_array(vk::AccelerationStructureDescriptorCell, 1);
    defer mem::free(heap.acceleration_structure_cells.ptr);
    heap.acceleration_structure_next_new = 1;
    vk::AccelerationStructureDescriptorCell* view =
        &heap.acceleration_structure_cells[0];
    view.used = true;
    view.generation = 31;
    view.owner = { .index = 2, .generation = 23 };

    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
        .expected_acceleration_structure_a = {
            .index      = 0,
            .generation = 17,
        },
        .expected_acceleration_structure_b = {
            .index      = 2,
            .generation = 23,
        },
        .expected_acceleration_structure_view_index = 0,
        .expected_acceleration_structure_view_generation = 31,
        .acceleration_structure_identity_valid = true,
    };
    state.debug_callback = &capture_leak_message;
    state.debug_user_data = &capture;

    assert(vk::report_resource_leaks(&state) == 2,
        "every live acceleration-structure owner is counted across gaps");
    vk::report_descriptor_leaks(&state);
    assert(capture.total == 3
            && capture.acceleration_structures == 2
            && capture.acceleration_structure_views == 1,
        "owners and the descriptor view each emit one semantic warning");
    assert(capture.public_identities == 3,
        "owner and view leaks carry their public slot identities");
    assert(capture.acceleration_structure_identity_valid
            && capture.acceleration_structure_a_seen
            && capture.acceleration_structure_b_seen
            && capture.acceleration_structure_view_identity_valid,
        "owner and view diagnostics preserve their exact slot identities");
    assert(capture.metadata_valid && capture.named_resources_valid,
        "acceleration-structure leaks preserve standard metadata and names");
}

fn void full_validation_teardown_reports_live_acceleration_structure() @test {
    gpu::Runtime runtime;
    gpu::Device device = create_ray_query_test_device(&runtime)!!;
    gpu::AccelerationStructureDesc desc = {
        .kind               = gpu::AccelerationStructureKind.TOP_LEVEL,
        .max_instance_count = 1,
        .debug_name         = "leaky_acceleration_structure",
    };
    gpu::AccelerationStructureHandle acceleration_structure =
        gpu::create_acceleration_structure(&device, &desc)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
        .expected_acceleration_structure_a = acceleration_structure,
        .acceleration_structure_identity_valid = true,
    };
    state.debug_callback = &capture_leak_message;
    state.debug_user_data = &capture;

    gpu::internal::mark_device_lost(&device);
    destroy_test_device(&device)!!;
    runtime = {};

    assert(!device.is_valid(),
        "device-loss teardown consumes the device with a live structure");
    assert(capture.total == 1
            && capture.acceleration_structures == 1
            && capture.acceleration_structure_a_seen,
        "FULL teardown routes the live acceleration-structure owner once");
    assert(capture.acceleration_structure_identity_valid
            && capture.public_identities == 1,
        "FULL teardown preserves the exact live owner identity");
    assert(capture.metadata_valid && capture.named_resources_valid,
        "FULL teardown preserves standard leak metadata and name");
}

fn void debug_name_copy_is_independent() @test {
    gpu::Device device = make_dbg_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;

    char[16] volatile_name;
    char[7] source = { 'v', 'o', 'l', 'a', 't', 'i', 'l' };
    for (uint i = 0; i < 7; i++) volatile_name[i] = source[i];
    volatile_name[7] = 0;

    gpu::AllocationDesc desc = {
        .size        = 64,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = (ZString)&volatile_name[0],
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    defer gpu::free_allocation(&device, &allocation)!!;

    foreach (&b : volatile_name) *b = 'X';

    char* stored = &state.allocations.slots[allocation.index].value.debug_name[0];
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
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::cmd_begin_label(&cmd, "resource_setup")!!;
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::cmd_end_label(&cmd)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
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

fn void callback_reports_leaks_without_validation() @test {
    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
    };
    gpu::RuntimeDesc device_desc = {
        .debug_callback   = &capture_leak_message,
        .debug_user_data  = &capture,
        .application_name = "gpu_callback_leak_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    gpu::AllocationDesc buffer_desc = {
        .size        = 128,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access      = { .graphics, .compute, .transfer },
        .debug_name  = "leaky_allocation",
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &buffer_desc)!!;
    capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
    };
    assert(vk::report_resource_leaks(state) == 1, "the live allocation is reported");

    assert(capture.total == 1 && capture.allocations == 1,
        "callback receives one allocation warning without validation");
    assert(capture.metadata_valid && capture.named_resources_valid,
        "the report preserves structured leak metadata");
    gpu::free_allocation(&device, &allocation)!!;
    destroy_test_device(&device)!!;
}
