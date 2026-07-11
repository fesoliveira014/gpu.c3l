module gpu_test;

import gpu;
import gpu::vk;
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
    bool              buffer_strings_match;
    bool              device_name_matches;
    bool              messenger_failure_strings_match;
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
    if (message.resource.kind == gpu::DebugResourceKind.DEVICE && message.resource.debug_name != null) {
        capture.device_name_matches = message.resource.debug_name.str_view() == "named_device";
    }
    if (message.backend_text != null && message.backend_text.str_view() == "fabricated messenger failure") {
        capture.messenger_failure_strings_match = message.operation != null
            && message.operation.str_view() == "create_device";
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
    assert(capture.last.severity == gpu::DebugMessageSeverity.ERROR, "error severity is preserved");
    assert(capture.last.category.backend, "Vulkan diagnostics are backend messages");
    assert(capture.last.category.validation, "validation category is preserved");
    assert(capture.last.category.performance, "performance category is preserved");
    assert(!capture.last.category.general, "unset general category stays clear");
    assert(capture.last.validation_id_number == 2718, "numeric validation ID is preserved");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.BUFFER, "named Vulkan buffer maps to public resource kind");
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

fn void zero_size_buffer_diagnostic_preserves_fault() @test {
    DebugCapture capture;
    gpu::DeviceDesc device_desc = {
        .backend         = gpu::BackendKind.VULKAN,
        .debug_callback  = &capture_debug_message,
        .debug_user_data = &capture,
        .frames_in_flight = 2,
        .application_name = "gpu_debug_contract_test",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    uint before = capture.calls;
    gpu::BufferDesc desc = {
        .size        = 0,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "zero_buffer",
    };
    fault rejected = @catch(gpu::create_buffer(&device, &desc));

    assert(rejected == gpu::INVALID_ARGUMENT, "structured diagnostics do not change the public fault");
    assert(capture.calls == before + 1, "one public-contract message accompanies the fault");
    assert(capture.last.severity == gpu::DebugMessageSeverity.ERROR, "public rejection is an error");
    assert(capture.last.category.public_contract, "public rejection carries the contract category");
    assert(capture.last.has_fault && capture.last.public_fault == gpu::INVALID_ARGUMENT,
        "diagnostic carries the same public fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.BUFFER, "rejected resource kind is identified");
    assert(!capture.last.resource.has_public_identity, "rejected create has no invented public identity");
    assert(capture.buffer_strings_match, "callback observes borrowed operation metadata and debug name while valid");
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
        .frames_in_flight  = 2,
        .application_name  = "gpu_debug_names_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    assert(state.debug_names, "backend retains the debug-name opt-in without validation");

    gpu::BufferDesc buffer_desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "names_without_validation",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buffer_desc)!!;
    gpu::destroy_buffer(&device, buffer)!!;
}

fn gpu::Device make_dbg_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_debug_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void leak_reporters_count_leaks() @test {
    gpu::Device device = make_dbg_device();
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "leaky_buffer",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buf_desc)!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
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

    gpu::SemaphoreDesc sem_desc = { .debug_name = "leaky_semaphore" };
    gpu::SemaphoreHandle semaphore = gpu::create_semaphore(&device, &sem_desc)!!;
    gpu::RecordingContextHandle ctx = gpu::create_recording_context(&device)!!;

    assert(vk::report_buffer_leaks(state) == 1, "one leaked buffer counted");
    assert(vk::report_texture_leaks(state) == 1, "one leaked texture counted");
    assert(vk::report_pipeline_leaks(state) == 1, "one leaked cache entry counted");
    assert(vk::report_shader_leaks(state) == 1, "one leaked shader counted");
    assert(vk::report_semaphore_leaks(state) == 1, "one leaked semaphore counted");
    assert(vk::report_context_leaks(state) == 1, "one leaked context counted (default and helper exempt)");

    // Slot data feeding the report lines.
    vk::TextureSlot* tex_slot = &state.textures.slots[texture.index].value;
    assert(tex_slot.allocation_size > 0, "texture leak line carries the allocation size");

    gpu::destroy_recording_context(&device, ctx)!!;
    gpu::destroy_semaphore(&device, semaphore)!!;
    gpu::destroy_pipeline(&device, first_alias)!!;
    gpu::destroy_pipeline(&device, second_alias)!!;
    gpu::destroy_shader(&device, shader)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::destroy_buffer(&device, buffer)!!;

    assert(vk::report_resource_leaks(state) == 0, "clean state reports zero leaks");
    gpu::destroy_device(&device)!!;
}

fn void debug_name_copy_is_independent() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    char[16] volatile_name;
    char[7] source = { 'v', 'o', 'l', 'a', 't', 'i', 'l' };
    for (uint i = 0; i < 7; i++) volatile_name[i] = source[i];
    volatile_name[7] = 0;

    gpu::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = (ZString)&volatile_name[0],
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    foreach (&b : volatile_name) *b = 'X';

    char* stored = &state.buffers.slots[buffer.index].value.debug_name[0];
    assert(stored[0] == 'v' && stored[6] == 'l' && stored[7] == 0,
        "the slot holds its own copy, unaffected by the caller's buffer");
}

fn void stats_counts_populated() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferHandle[3] buffers;
    for (uint i = 0; i < 3; i++) {
        gpu::BufferDesc desc = { .size = 128, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
    }
    gpu::TextureHandle[2] textures;
    for (uint i = 0; i < 2; i++) {
        gpu::TextureDesc desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled },
        };
        textures[i] = gpu::create_texture(&device, &desc)!!;
    }

    gpu::MemoryStats stats = gpu::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 3, "buffer count reflects live buffers");
    assert(stats.texture_count == 2, "texture count reflects live textures");
    assert(stats.heap_count > 0 && stats.heaps[0].budget > 0, "stats include a VMA budget");

    foreach (t : textures) gpu::destroy_texture(&device, t)!!;
    foreach (b : buffers) gpu::destroy_buffer(&device, b)!!;
    stats = gpu::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 0 && stats.texture_count == 0, "counts drop to zero after destroy");
}

fn void labels_record_validation_clean() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::cmd_begin_label(&cmd, "frame_setup")!!;
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

    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;

    fault ended = @catch(gpu::cmd_begin_label(&cmd, "too_late"));
    assert(ended == gpu::INVALID_HANDLE, "labels on a consumed list fault INVALID_HANDLE");
    fault end_stale = @catch(gpu::cmd_end_label(&cmd));
    assert(end_stale == gpu::INVALID_HANDLE, "ending a label on a consumed list faults INVALID_HANDLE");
}

fn void destroy_reports_leaks_before_sweeping() @test {
    gpu::Device device = make_dbg_device();
    gpu::BufferDesc desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "canary_buffer",
    };
    (void)gpu::create_buffer(&device, &desc)!!;
    // Teardown must print the canary's leak line (visible in test stderr)
    // and still destroy everything without faulting.
    gpu::destroy_device(&device)!!;
}
