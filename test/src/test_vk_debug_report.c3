module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;
import std::core::mem;
import std::thread;

// Headless (lavapipe): leak reporters, debug-name storage, stats counts, and
// command labels. The leak test intentionally prints leak lines to stderr.

const char[*] DBG_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

struct DebugCapture {
    uint              calls;
    gpu::compat::DebugMessage last;
    bool              validation_strings_match;
    bool              buffer_strings_match;
    bool              device_name_matches;
    bool              messenger_failure_strings_match;
}

fn void capture_debug_message(gpu::compat::DebugMessage* message, void* user_data) {
    DebugCapture* capture = (DebugCapture*)user_data;
    capture.calls++;
    capture.last = *message;
    if (message.validation_id_name != null
        && message.validation_id_name.str_view() == "VUID-Test-123") {
        capture.validation_strings_match = message.backend_text != null
            && message.backend_text.str_view() == "fabricated validation message"
            && message.resource.debug_name != null
            && message.resource.debug_name.str_view() == "named_buffer";
    }
    if (message.operation != null && message.operation.str_view() == "create_buffer") {
        capture.buffer_strings_match = message.resource.debug_name != null
            && message.resource.debug_name.str_view() == "zero_buffer"
            && message.rejected_field != null
            && message.rejected_field.str_view() == "BufferDesc.size"
            && message.invariant != null
            && message.invariant.str_view() == "must be greater than zero";
    }
    if (message.resource.kind == gpu::compat::DebugResourceKind.DEVICE && message.resource.debug_name != null) {
        capture.device_name_matches = message.resource.debug_name.str_view() == "named_device";
    }
    if (message.backend_text != null && message.backend_text.str_view() == "fabricated messenger failure") {
        capture.messenger_failure_strings_match = message.operation != null
            && message.operation.str_view() == "create_device";
    }
}

struct LeakCapture {
    uint total;
    uint buffers;
    uint textures;
    uint pipelines;
    uint shaders;
    uint semaphores;
    uint contexts;
    uint texture_descriptors;
    uint samplers;
    uint persistent_spans;
    uint public_identities;
    bool metadata_valid;
    gpu::compat::PipelineHandle expected_pipeline_a;
    gpu::compat::PipelineHandle expected_pipeline_b;
    bool pipeline_identity_valid;
    bool named_resources_valid;
}

fn void capture_leak_message(gpu::compat::DebugMessage* message, void* user_data) {
    LeakCapture* capture = (LeakCapture*)user_data;
    capture.total++;
    bool operation_valid = message.operation != null && message.operation.str_view() == "destroy_device";
    bool invariant_valid = message.invariant != null
        && message.invariant.str_view() == "resource remains live at device destruction";
    capture.metadata_valid = capture.metadata_valid
        && message.severity == gpu::compat::DebugMessageSeverity.WARNING
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
        case gpu::compat::DebugResourceKind.BUFFER:
            capture.buffers++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_buffer";
        case gpu::compat::DebugResourceKind.TEXTURE:
            capture.textures++;
            capture.named_resources_valid = capture.named_resources_valid
                && message.resource.debug_name != null
                && message.resource.debug_name.str_view() == "leaky_texture";
        case gpu::compat::DebugResourceKind.PIPELINE:
            capture.pipelines++;
            bool first = message.resource.index == capture.expected_pipeline_a.index
                && message.resource.generation == capture.expected_pipeline_a.generation;
            bool second = message.resource.index == capture.expected_pipeline_b.index
                && message.resource.generation == capture.expected_pipeline_b.generation;
            capture.pipeline_identity_valid = message.resource.has_public_identity
                && (first || second);
        case gpu::compat::DebugResourceKind.SHADER:             capture.shaders++;
        case gpu::compat::DebugResourceKind.SEMAPHORE:          capture.semaphores++;
        case gpu::compat::DebugResourceKind.RECORDING_CONTEXT:  capture.contexts++;
        case gpu::compat::DebugResourceKind.TEXTURE_DESCRIPTOR: capture.texture_descriptors++;
        case gpu::compat::DebugResourceKind.SAMPLER:            capture.samplers++;
        case gpu::compat::DebugResourceKind.PERSISTENT_SPAN:    capture.persistent_spans++;
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

fn void capture_concurrent_debug_message(gpu::compat::DebugMessage* message, void* user_data) {
    ConcurrentDebugCapture* capture = (ConcurrentDebugCapture*)user_data;
    capture.mutex.lock()!!;
    defer capture.mutex.unlock();
    capture.calls++;
    switch (message.severity) {
        case gpu::compat::DebugMessageSeverity.WARNING: capture.warnings++;
        case gpu::compat::DebugMessageSeverity.ERROR:   capture.errors++;
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
    object.p_object_name = (char*)"named_buffer";
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
    assert(capture.last.severity == gpu::compat::DebugMessageSeverity.ERROR, "error severity is preserved");
    assert(capture.last.category.backend, "Vulkan diagnostics are backend messages");
    assert(capture.last.category.validation, "validation category is preserved");
    assert(capture.last.category.performance, "performance category is preserved");
    assert(!capture.last.category.general, "unset general category stays clear");
    assert(capture.last.validation_id_number == 2718, "numeric validation ID is preserved");
    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.BUFFER, "named Vulkan buffer maps to public resource kind");
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
    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.DEVICE, "Vulkan device maps to public device resource kind");
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

    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.TEXTURE,
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

    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.BUFFER,
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
    assert(capture.last.severity == gpu::compat::DebugMessageSeverity.INFO, "info severity is preserved");
    assert(capture.last.category.backend && capture.last.category.general, "general backend category is preserved");
    assert(capture.last.backend_text == null, "missing text remains absent");
    assert(capture.last.validation_id_name == null, "missing validation ID name remains absent");
    assert(capture.last.validation_id_number == 0, "missing numeric validation ID remains zero");
    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.NONE, "missing objects produce no resource kind");
    assert(!capture.last.resource.has_public_identity, "missing objects produce no public identity");
}

fn void zero_size_buffer_diagnostic_preserves_fault() @test {
    DebugCapture capture;
    gpu::compat::DeviceDesc device_desc = {
        .backend         = gpu::compat::BackendKind.VULKAN,
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
        .frames_in_flight = 2,
        .application_name = "gpu_debug_contract_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    uint before = capture.calls;
    gpu::compat::BufferDesc desc = {
        .size        = 0,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "zero_buffer",
    };
    fault rejected = @catch(gpu::compat::create_buffer(&device, &desc));

    assert(rejected == gpu::compat::INVALID_ARGUMENT, "structured diagnostics do not change the public fault");
    assert(capture.calls == before + 1, "one public-contract message accompanies the fault");
    assert(capture.last.severity == gpu::compat::DebugMessageSeverity.ERROR, "public rejection is an error");
    assert(capture.last.category.public_contract, "public rejection carries the contract category");
    assert(capture.last.has_fault && capture.last.public_fault == gpu::compat::INVALID_ARGUMENT,
        "diagnostic carries the same public fault");
    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.BUFFER, "rejected resource kind is identified");
    assert(!capture.last.resource.has_public_identity, "rejected create has no invented public identity");
    assert(capture.buffer_strings_match, "callback observes borrowed operation metadata and debug name while valid");
}

fn void contract_failure_helper_preserves_fault_and_dispatches_once() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };

    fault returned = vk::report_contract_failure(
        state:               &state,
        operation:           "create_buffer",
        public_fault:        gpu::compat::INVALID_ARGUMENT,
        resource_kind:       gpu::compat::DebugResourceKind.BUFFER,
        has_public_identity: true,
        index:               7,
        generation:          9,
        debug_name:          "zero_buffer",
        rejected_field:      "BufferDesc.size",
        invariant:           "must be greater than zero",
        backend_text:        null,
    );

    assert(returned == gpu::compat::INVALID_ARGUMENT, "helper returns the unchanged public fault");
    assert(capture.calls == 1, "helper dispatches exactly once");
    assert(capture.last.has_fault && capture.last.public_fault == returned,
        "structured message carries the returned fault");
    assert(capture.last.resource.kind == gpu::compat::DebugResourceKind.BUFFER,
        "helper preserves resource kind");
    assert(capture.last.resource.has_public_identity
        && capture.last.resource.index == 7
        && capture.last.resource.generation == 9,
        "helper preserves public slot identity");
    assert(capture.buffer_strings_match, "helper exposes borrowed contract strings only during callback delivery");
}

fn void debug_messenger_failure_routes_backend_warning() @test {
    DebugCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
    };

    vk::report_debug_messenger_failure(&state, "fabricated messenger failure");

    assert(capture.calls == 1, "messenger setup failure routes exactly once");
    assert(capture.last.severity == gpu::compat::DebugMessageSeverity.WARNING, "messenger setup failure is a warning");
    assert(capture.last.category.backend, "messenger setup failure is a backend diagnostic");
    assert(!capture.last.has_fault, "best-effort messenger setup failure carries no public fault");
    assert(capture.messenger_failure_strings_match,
        "callback observes borrowed create-device operation and failure text while valid");
}

fn void debug_names_do_not_require_validation() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = false,
        .enable_debug_names = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_debug_names_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    assert(state.debug_names, "backend retains the debug-name opt-in without validation");

    gpu::compat::BufferDesc buffer_desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "names_without_validation",
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &buffer_desc)!!;
    gpu::compat::destroy_buffer(&device, buffer)!!;
}

fn gpu::compat::Device make_dbg_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_debug_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn void leak_reporters_count_leaks() @test {
    gpu::compat::Device device = make_dbg_device();
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    gpu::compat::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "leaky_buffer",
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &buf_desc)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = 8,
        .height     = 8,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "leaky_texture",
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;

    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = DBG_COMP_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle shader = gpu::compat::create_shader(&device, &comp_sd)!!;
    gpu::compat::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = 8 };
    gpu::compat::PipelineHandle first_alias = gpu::compat::create_compute_pipeline(&device, &pipe_desc)!!;
    gpu::compat::PipelineHandle second_alias = gpu::compat::create_compute_pipeline(&device, &pipe_desc)!!;

    gpu::compat::SemaphoreDesc sem_desc = { .debug_name = "leaky_semaphore" };
    gpu::compat::SemaphoreHandle semaphore = gpu::compat::create_semaphore(&device, &sem_desc)!!;
    gpu::compat::RecordingContextHandle ctx = gpu::compat::create_recording_context(&device)!!;

    gpu::compat::TextureIndex texture_index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex sampler = gpu::compat::create_sampler(&device, &sampler_desc)!!;

    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
        .expected_pipeline_a   = first_alias,
        .expected_pipeline_b   = second_alias,
    };
    state.debug_callback = &capture_leak_message;
    state.debug_user_data = &capture;

    assert(vk::report_buffer_leaks(state) == 1, "one leaked buffer counted");
    assert(vk::report_texture_leaks(state) == 1, "one leaked texture counted");
    assert(vk::report_pipeline_leaks(state) == 1, "one leaked cache entry counted");
    assert(vk::report_shader_leaks(state) == 1, "one leaked shader counted");
    assert(vk::report_semaphore_leaks(state) == 1, "one leaked semaphore counted");
    assert(vk::report_context_leaks(state) == 1, "one leaked context counted (default and helper exempt)");
    vk::report_descriptor_leaks(state);

    assert(capture.total == 8, "each live resource or descriptor emits exactly one structured warning");
    assert(capture.buffers == 1 && capture.textures == 1 && capture.pipelines == 1,
        "buffer, texture, and pipeline leak kinds are exact");
    assert(capture.shaders == 1 && capture.semaphores == 1 && capture.contexts == 1,
        "shader, semaphore, and context leak kinds are exact");
    assert(capture.texture_descriptors == 1 && capture.samplers == 1,
        "descriptor leak kinds are exact");
    assert(capture.public_identities == 8,
        "every slot-backed leak carries one public identity");
    assert(capture.pipeline_identity_valid, "pipeline cache leak identifies one live public alias");
    assert(capture.metadata_valid, "leak diagnostics are warning, resource-lifetime, faultless destroy-device messages");
    assert(capture.named_resources_valid, "slot-owned buffer and texture names are observed during the callback");

    // Slot data feeding the report lines.
    vk::TextureSlot* tex_slot = &state.textures.slots[texture.index].value;
    assert(tex_slot.allocation_size > 0, "texture leak line carries the allocation size");

    gpu::compat::destroy_texture_descriptor(&device, texture_index)!!;
    gpu::compat::destroy_sampler(&device, sampler)!!;
    gpu::compat::destroy_recording_context(&device, ctx)!!;
    gpu::compat::destroy_semaphore(&device, semaphore)!!;
    gpu::compat::destroy_pipeline(&device, first_alias)!!;
    gpu::compat::destroy_pipeline(&device, second_alias)!!;
    gpu::compat::destroy_shader(&device, shader)!!;
    gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::PersistentAllocDesc persistent_desc = {
        .size        = 64,
        .align       = 16,
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "leaky_persistent_span",
    };
    gpu::compat::GpuSpan persistent = gpu::compat::alloc_persistent_span(&device, &persistent_desc)!!;
    assert(vk::report_resource_leaks(state) == 1, "one persistent span leak counted");
    assert(capture.total == 9 && capture.persistent_spans == 1,
        "persistent allocation summary emits one structured warning");
    assert(capture.metadata_valid, "persistent warning uses the same resource-lifetime contract");
    gpu::compat::free_persistent_span(&device, persistent)!!;

    assert(vk::report_resource_leaks(state) == 0, "clean state reports zero leaks");
    assert(capture.total == 9, "clean leak report emits nothing");
    gpu::compat::destroy_device(&device)!!;
}

fn void debug_name_copy_is_independent() @test {
    gpu::compat::Device device = make_dbg_device();
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    char[16] volatile_name;
    char[7] source = { 'v', 'o', 'l', 'a', 't', 'i', 'l' };
    for (uint i = 0; i < 7; i++) volatile_name[i] = source[i];
    volatile_name[7] = 0;

    gpu::compat::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = (ZString)&volatile_name[0],
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    foreach (&b : volatile_name) *b = 'X';

    char* stored = &state.buffers.slots[buffer.index].value.debug_name[0];
    assert(stored[0] == 'v' && stored[6] == 'l' && stored[7] == 0,
        "the slot holds its own copy, unaffected by the caller's buffer");
}

fn void stats_counts_populated() @test {
    gpu::compat::Device device = make_dbg_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferHandle[3] buffers;
    for (uint i = 0; i < 3; i++) {
        gpu::compat::BufferDesc desc = { .size = 128, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
        buffers[i] = gpu::compat::create_buffer(&device, &desc)!!;
    }
    gpu::compat::TextureHandle[2] textures;
    for (uint i = 0; i < 2; i++) {
        gpu::compat::TextureDesc desc = {
            .dimension = gpu::compat::TextureDimension.TEX_2D,
            .width     = 4,
            .height    = 4,
            .format    = gpu::compat::Format.RGBA8_UNORM,
            .usage     = { .sampled },
        };
        textures[i] = gpu::compat::create_texture(&device, &desc)!!;
    }

    gpu::compat::MemoryStats stats = gpu::compat::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 3, "buffer count reflects live buffers");
    assert(stats.texture_count == 2, "texture count reflects live textures");
    assert(stats.heap_count > 0 && stats.heaps[0].budget > 0, "stats include a VMA budget");

    foreach (t : textures) gpu::compat::destroy_texture(&device, t)!!;
    foreach (b : buffers) gpu::compat::destroy_buffer(&device, b)!!;
    stats = gpu::compat::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 0 && stats.texture_count == 0, "counts drop to zero after destroy");
}

fn void labels_record_validation_clean() @test {
    gpu::compat::Device device = make_dbg_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc rt_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::compat::TextureHandle target = gpu::compat::create_texture(&device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::cmd_begin_label(&cmd, "frame_setup")!!;
    gpu::compat::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::compat::cmd_end_label(&cmd)!!;

    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_begin_label(&cmd, "main_pass", { 0.2f, 0.8f, 0.2f, 1.0f })!!;
    gpu::compat::cmd_begin_label(&cmd, "nested")!!;
    gpu::compat::cmd_end_label(&cmd)!!;
    gpu::compat::cmd_end_label(&cmd)!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;

    fault ended = @catch(gpu::compat::cmd_begin_label(&cmd, "too_late"));
    assert(ended == gpu::compat::INVALID_HANDLE, "labels on a consumed list fault INVALID_HANDLE");
    fault end_stale = @catch(gpu::compat::cmd_end_label(&cmd));
    assert(end_stale == gpu::compat::INVALID_HANDLE, "ending a label on a consumed list faults INVALID_HANDLE");
}

fn void destroy_reports_leaks_before_sweeping() @test {
    gpu::compat::Device device = make_dbg_device();
    gpu::compat::BufferDesc desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "canary_buffer",
    };
    (void)gpu::compat::create_buffer(&device, &desc)!!;
    // Teardown must print the canary's leak line (visible in test stderr)
    // and still destroy everything without faulting.
    gpu::compat::destroy_device(&device)!!;
}

fn void callback_reports_leaks_without_validation() @test {
    LeakCapture capture = {
        .metadata_valid        = true,
        .named_resources_valid = true,
    };
    gpu::compat::DeviceDesc device_desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .debug_callback    = &capture_leak_message,
        .debug_user_data   = &capture,
        .frames_in_flight  = 2,
        .application_name  = "gpu_callback_leak_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    gpu::compat::BufferDesc buffer_desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "leaky_buffer",
    };
    (void)gpu::compat::create_buffer(&device, &buffer_desc)!!;

    gpu::compat::destroy_device(&device)!!;

    assert(capture.total == 1 && capture.buffers == 1,
        "callback receives one leaked-buffer warning without validation");
    assert(capture.metadata_valid && capture.named_resources_valid,
        "validation-disabled teardown preserves structured leak metadata");
}
