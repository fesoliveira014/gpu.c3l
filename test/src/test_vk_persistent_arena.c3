module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;
import std::io;

struct PersistentDiagnosticCapture {
    uint                      calls;
    gpu::compat::DebugMessageSeverity severity;
    gpu::compat::DebugMessageCategory category;
    bool                      has_fault;
    fault                     public_fault;
    gpu::compat::DebugResourceKind    resource_kind;
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

fn void copy_persistent_diagnostic_text(char* destination, usz capacity, ZString source) {
    usz index = 0;
    if (source != null) {
        while (index + 1 < capacity && source[index] != 0) {
            destination[index] = source[index];
            index++;
        }
    }
    destination[index] = 0;
}

fn void capture_persistent_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    PersistentDiagnosticCapture* capture = (PersistentDiagnosticCapture*)user_data;
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
    copy_persistent_diagnostic_text(&capture.operation[0], 64, message.operation);
    copy_persistent_diagnostic_text(&capture.resource_name[0], 64, message.resource.debug_name);
    copy_persistent_diagnostic_text(&capture.rejected_field[0], 64, message.rejected_field);
    copy_persistent_diagnostic_text(&capture.invariant[0], 128, message.invariant);
    copy_persistent_diagnostic_text(&capture.backend_text[0], 128, message.backend_text);
    copy_persistent_diagnostic_text(&capture.validation_id_name[0], 64, message.validation_id_name);
}

fn void assert_persistent_rejection(
    PersistentDiagnosticCapture* capture,
    uint before,
    fault returned,
    ZString field,
    ZString invariant,
    ZString debug_name,
) {
    assert(returned == gpu::compat::INVALID_ARGUMENT, "persistent rejection keeps INVALID_ARGUMENT");
    assert(capture.calls == before + 1, "persistent rejection emits exactly one diagnostic");
    assert(((ZString)&capture.operation[0]).str_view() == "alloc_persistent_span", "persistent operation is exact");
    assert(capture.has_fault && capture.public_fault == returned, "diagnostic fault matches return");
    assert(capture.resource_kind == gpu::compat::DebugResourceKind.PERSISTENT_SPAN, "persistent resource kind is exact");
    assert(!capture.has_public_identity, "rejected allocation invents no identity");
    assert(((ZString)&capture.resource_name[0]).str_view() == debug_name.str_view(), "persistent debug name is preserved");
    assert(((ZString)&capture.rejected_field[0]).str_view() == field.str_view(), "persistent rejected field is exact");
    assert(((ZString)&capture.invariant[0]).str_view() == invariant.str_view(), "persistent invariant is exact");
}

// Headless (lavapipe): persistent arena alloc/free/reuse, alignment, stable
// addresses, stats, and usage honoring (index/indirect/transfer spans usable,
// vertex-bit requests fault).

const char[*] PA_DOUBLE_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn gpu::compat::PipelineHandle make_persistent_cross_family_pipeline(
    gpu::compat::Device* device,
    gpu::compat::ShaderHandle* out_shader,
) {
    gpu::compat::ShaderDesc shader_desc = {
        .stage       = gpu::compat::ShaderStage.COMPUTE,
        .spirv       = PA_DOUBLE_SPIRV[..],
        .entry_point = "main",
    };
    *out_shader = gpu::compat::create_shader(device, &shader_desc)!!;
    gpu::compat::ComputePipelineDesc pipeline_desc = {
        .shader             = *out_shader,
        .push_constant_size = gpu::compat::RootPush::size,
    };
    return gpu::compat::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::compat::GpuSpan alloc_persistent_indirect_args(gpu::compat::Device* device) {
    gpu::compat::PersistentAllocDesc desc = {
        .size        = gpu::compat::DispatchIndirectCommand::size,
        .align       = 4,
        .usage       = { .indirect },
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "persistent_cross_family_args",
    };
    gpu::compat::GpuSpan span = gpu::compat::alloc_persistent_span(device, &desc)!!;
    gpu::compat::DispatchIndirectCommand* args = (gpu::compat::DispatchIndirectCommand*)span.cpu;
    *args = { .x = 1, .y = 1, .z = 1 };
    gpu::compat::flush_buffer(device, span.buffer, span.offset, span.size)!!;
    return span;
}

fn gpu::compat::GpuSpan alloc_frame_empty_root(gpu::compat::FrameToken* frame) {
    gpu::compat::GpuSpan span = gpu::compat::alloc_frame_span(frame, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)span.cpu;
    *root = { .count = 0 };
    return span;
}

fn void persistent_span_crosses_compute_and_graphics_families_when_available() @test {
    gpu::compat::Device device = make_persistent_usage_device();
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    if (state.graphics_family == state.compute_family) {
        io::eprintfn(
            "persistent_span_crosses_compute_and_graphics: N/A graphics=%d compute=%d",
            state.graphics_family,
            state.compute_family,
        );
        return;
    }

    gpu::compat::ShaderHandle shader;
    gpu::compat::PipelineHandle pipeline = make_persistent_cross_family_pipeline(&device, &shader);
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;
    defer gpu::compat::destroy_shader(&device, shader)!!;
    gpu::compat::SemaphoreDesc semaphore_desc = { .debug_name = "persistent_cross_family" };
    gpu::compat::SemaphoreHandle semaphore = gpu::compat::create_semaphore(&device, &semaphore_desc)!!;
    defer gpu::compat::destroy_semaphore(&device, semaphore)!!;
    gpu::compat::GpuSpan args = alloc_persistent_indirect_args(&device);

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan compute_root = alloc_frame_empty_root(&frame);
    gpu::compat::GpuSpan graphics_root = alloc_frame_empty_root(&frame);

    gpu::compat::CommandList compute = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     compute_root.gpu,
        args:     args,
    )!!;
    gpu::compat::end_commands(&compute)!!;
    gpu::compat::CommandList[1] compute_lists = { compute };
    gpu::compat::SemaphoreSignal[1] signals = {{
        .semaphore = semaphore,
        .value     = (gpu::compat::SemaphoreValue)1,
        .stage     = gpu::compat::Stage.INDIRECT_COMMAND,
    }};
    gpu::compat::SubmitDesc compute_submit = {
        .command_lists = compute_lists[..],
        .signals       = signals[..],
    };
    gpu::compat::submit(&device, &compute_submit)!!;

    gpu::compat::CommandList graphics = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     graphics_root.gpu,
        args:     args,
    )!!;
    gpu::compat::end_commands(&graphics)!!;
    gpu::compat::CommandList[1] graphics_lists = { graphics };
    gpu::compat::SemaphoreWait[1] waits = {{
        .semaphore = semaphore,
        .value     = (gpu::compat::SemaphoreValue)1,
        .stage     = gpu::compat::Stage.INDIRECT_COMMAND,
    }};
    gpu::compat::SubmitDesc graphics_submit = {
        .command_lists = graphics_lists[..],
        .waits         = waits[..],
    };
    gpu::compat::submit(&device, &graphics_submit)!!;
    gpu::compat::end_frame(&frame)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::free_persistent_span(&device, args)!!;
    io::eprintfn(
        "persistent_span_crosses_compute_and_graphics: PASS graphics=%d compute=%d",
        state.graphics_family,
        state.compute_family,
    );
}

fn void persistent_span_reused_offset_switches_family_when_available() @test {
    gpu::compat::DeviceDesc desc = {
        .backend               = gpu::compat::BackendKind.VULKAN,
        .enable_validation     = true,
        .frames_in_flight      = 1,
        .persistent_arena_size = gpu::compat::DispatchIndirectCommand::size,
        .application_name      = "persistent_reuse_cross_family",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    if (state.graphics_family == state.compute_family) {
        io::eprintfn(
            "persistent_span_reused_offset_switches_family: N/A graphics=%d compute=%d",
            state.graphics_family,
            state.compute_family,
        );
        return;
    }

    gpu::compat::ShaderHandle shader;
    gpu::compat::PipelineHandle pipeline = make_persistent_cross_family_pipeline(&device, &shader);
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;
    defer gpu::compat::destroy_shader(&device, shader)!!;

    gpu::compat::GpuSpan compute_args = alloc_persistent_indirect_args(&device);
    assert(compute_args.offset == 0, "first persistent allocation uses offset zero");
    gpu::compat::BufferHandle backing = compute_args.buffer;
    gpu::compat::GpuAddress address = compute_args.gpu;
    gpu::compat::FrameToken compute_frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan compute_root = alloc_frame_empty_root(&compute_frame);
    gpu::compat::CommandList compute = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     compute_root.gpu,
        args:     compute_args,
    )!!;
    gpu::compat::end_commands(&compute)!!;
    gpu::compat::CommandList[1] compute_lists = { compute };
    gpu::compat::SubmitDesc compute_submit = { .command_lists = compute_lists[..] };
    gpu::compat::submit(&device, &compute_submit)!!;
    gpu::compat::end_frame(&compute_frame)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::free_persistent_span(&device, compute_args)!!;

    gpu::compat::GpuSpan graphics_args = alloc_persistent_indirect_args(&device);
    assert(graphics_args.buffer == backing, "reallocation keeps the persistent backing buffer");
    assert(graphics_args.offset == 0, "retired persistent allocation reuses offset zero");
    assert(graphics_args.gpu == address, "retired persistent allocation reuses its GPU address");
    gpu::compat::FrameToken graphics_frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan graphics_root = alloc_frame_empty_root(&graphics_frame);
    gpu::compat::CommandList graphics = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     graphics_root.gpu,
        args:     graphics_args,
    )!!;
    gpu::compat::end_commands(&graphics)!!;
    gpu::compat::CommandList[1] graphics_lists = { graphics };
    gpu::compat::SubmitDesc graphics_submit = { .command_lists = graphics_lists[..] };
    gpu::compat::submit(&device, &graphics_submit)!!;
    gpu::compat::end_frame(&graphics_frame)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::free_persistent_span(&device, graphics_args)!!;
    io::eprintfn(
        "persistent_span_reused_offset_switches_family: PASS graphics=%d compute=%d",
        state.graphics_family,
        state.compute_family,
    );
}

fn void persistent_arena_desc_has_exact_backing_contract() @test {
    vk::VkDeviceState state = {
        .graphics_family = 4,
        .compute_family  = 4,
        .transfer_family = 4,
    };
    vk::PersistentArenaCreatePlan plan = vk::persistent_arena_create_plan(&state, 4096);
    gpu::compat::BufferUsage expected_usage = {
        .transfer_src,
        .transfer_dst,
        .uniform,
        .storage,
        .addressable,
        .indirect,
        .index,
        .shared_queues,
    };

    assert(plan.desc.size == 4096, "persistent arena descriptor keeps the requested size");
    assert((uint)plan.desc.usage == (uint)expected_usage, "persistent arena descriptor uses the exact backing superset");
    assert(plan.desc.memory_kind == gpu::compat::MemoryKind.PERSISTENT_UPLOAD, "persistent arena descriptor uses persistent upload memory");
    assert(plan.desc.debug_name == "persistent_arena", "persistent arena descriptor keeps its debug name");
}

fn void persistent_arena_create_info_is_exclusive_when_families_alias() @test {
    vk::VkDeviceState state = {
        .graphics_family = 4,
        .compute_family  = 4,
        .transfer_family = 4,
    };
    vk::PersistentArenaCreatePlan plan = vk::persistent_arena_create_plan(&state, 4096);
    vk::BufferCreateInfo info = vk::make_buffer_create_info(&plan.desc, plan.families[:plan.family_count])!!;

    assert(plan.family_count == 1, "aliased queues produce one persistent arena family");
    assert(plan.families[0] == 4, "aliased persistent arena keeps the graphics family");
    assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "one family keeps persistent arena sharing exclusive");
    assert(info.queue_family_index_count == 0, "exclusive persistent arena creation publishes no family count");
    assert(info.p_queue_family_indices == null, "exclusive persistent arena creation publishes no family pointer");
}

fn void persistent_arena_create_info_uses_two_exact_families() @test {
    uint[3][3] topologies = {
        { 4, 5, 5 },
        { 4, 4, 7 },
        { 4, 5, 4 },
    };
    uint[3] expected_second = { 5, 7, 5 };
    foreach (index, topology : topologies) {
        vk::VkDeviceState state = {
            .graphics_family = topology[0],
            .compute_family  = topology[1],
            .transfer_family = topology[2],
        };
        vk::PersistentArenaCreatePlan plan = vk::persistent_arena_create_plan(&state, 4096);
        vk::BufferCreateInfo info = vk::make_buffer_create_info(&plan.desc, plan.families[:plan.family_count])!!;

        assert(plan.family_count == 2, "two distinct queues produce two persistent arena families");
        assert(plan.families[0] == 4, "two-family persistent arena keeps graphics first");
        assert(plan.families[1] == expected_second[index], "two-family persistent arena keeps the exact second family");
        assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "two families use concurrent persistent arena sharing");
        assert(info.queue_family_index_count == 2, "concurrent persistent arena creation publishes two families");
        assert(info.p_queue_family_indices[0] == 4, "persistent arena keeps graphics first");
        assert(info.p_queue_family_indices[1] == expected_second[index], "create info keeps the exact second family");
    }
}

fn void persistent_arena_create_info_uses_three_exact_families() @test {
    vk::VkDeviceState state = {
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    vk::PersistentArenaCreatePlan plan = vk::persistent_arena_create_plan(&state, 4096);
    vk::BufferCreateInfo info = vk::make_buffer_create_info(&plan.desc, plan.families[:plan.family_count])!!;

    assert(plan.family_count == 3, "three distinct queues produce three persistent arena families");
    assert(plan.families[0] == 4, "tri-family persistent arena keeps graphics first");
    assert(plan.families[1] == 5, "tri-family persistent arena keeps compute second");
    assert(plan.families[2] == 7, "tri-family persistent arena keeps transfer third");
    assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "three families use concurrent persistent arena sharing");
    assert(info.queue_family_index_count == 3, "concurrent persistent arena creation publishes three families");
    assert(info.p_queue_family_indices[0] == 4, "persistent arena keeps graphics first");
    assert(info.p_queue_family_indices[1] == 5, "persistent arena keeps compute second");
    assert(info.p_queue_family_indices[2] == 7, "persistent arena keeps transfer third");
}

fn void persistent_alloc_free_stats() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_persistent_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::PersistentArenaStats empty = gpu::compat::get_persistent_stats(&device)!!;
    assert(empty.used_bytes == 0, "arena starts empty");
    assert(empty.capacity_bytes == gpu::compat::DEFAULT_PERSISTENT_ARENA_SIZE, "capacity is the arena size");

    gpu::compat::PersistentAllocDesc desc_a = {
        .size = 256, .align = 64, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "a",
    };
    gpu::compat::GpuSpan a = gpu::compat::alloc_persistent_span(&device, &desc_a)!!;
    assert(a.cpu != null, "persistent span is mapped");
    assert((ulong)a.gpu != 0, "non-zero GPU address");
    assert(a.offset % 64 == 0, "offset honors alignment");
    ulong a_gpu = (ulong)a.gpu;

    gpu::compat::PersistentArenaStats after_a = gpu::compat::get_persistent_stats(&device)!!;
    assert(after_a.used_bytes >= 256, "used bytes rose");
    assert(after_a.allocation_count == 1, "one live allocation");

    gpu::compat::PersistentAllocDesc desc_b = { .size = 512, .align = 256, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    gpu::compat::GpuSpan b = gpu::compat::alloc_persistent_span(&device, &desc_b)!!;
    assert((ulong)b.gpu != a_gpu, "B gets a distinct address");
    assert(gpu::compat::get_persistent_stats(&device)!!.allocation_count == 2, "two live allocations");

    // Free B; A's address is unchanged, count drops.
    gpu::compat::free_persistent_span(&device, b)!!;
    gpu::compat::PersistentArenaStats after_free_b = gpu::compat::get_persistent_stats(&device)!!;
    assert(after_free_b.allocation_count == 1, "back to one after free");
    assert(after_free_b.used_bytes < after_a.used_bytes + 512, "used dropped after free");
    assert((ulong)a.gpu == a_gpu, "A's GPU address is stable");

    // Reuse: C allocates into freed space, A still distinct/stable.
    gpu::compat::PersistentAllocDesc desc_c = { .size = 128, .align = 16, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    gpu::compat::GpuSpan c = gpu::compat::alloc_persistent_span(&device, &desc_c)!!;
    assert((ulong)c.gpu != a_gpu, "C distinct from live A");

    gpu::compat::free_persistent_span(&device, a)!!;
    gpu::compat::free_persistent_span(&device, c)!!;
    gpu::compat::PersistentArenaStats drained = gpu::compat::get_persistent_stats(&device)!!;
    assert(drained.allocation_count == 0, "arena empty after freeing all");
    assert(drained.used_bytes == 0, "used back to 0");

    // Wrong memory kind faults.
    gpu::compat::PersistentAllocDesc bad_kind = { .size = 64, .align = 16, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    fault kind_fault = @catch(gpu::compat::alloc_persistent_span(&device, &bad_kind));
    assert(kind_fault == gpu::compat::INVALID_ARGUMENT, "non-persistent memory kind faults INVALID_ARGUMENT");

    // Bogus free faults.
    gpu::compat::GpuSpan bogus = { .buffer = a.buffer, .offset = 999_999 };
    fault free_fault = @catch(gpu::compat::free_persistent_span(&device, bogus));
    assert(free_fault == gpu::compat::INVALID_HANDLE, "freeing a bogus span faults INVALID_HANDLE");
}

fn void persistent_allocation_rejections_report_contract_context() @test {
    PersistentDiagnosticCapture capture;
    gpu::compat::DeviceDesc device_desc = {
        .backend = gpu::compat::BackendKind.VULKAN, .debug_callback = &capture_persistent_diagnostic,
        .debug_user_data = &capture, .frames_in_flight = 2,
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::PersistentAllocDesc bad_size = {
        .size = 0, .align = 16, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "bad_size",
    };
    uint before = capture.calls;
    fault size_fault = @catch(gpu::compat::alloc_persistent_span(&device, &bad_size));
    assert_persistent_rejection(
        capture:  &capture,
        before:   before,
        returned: size_fault,
        field:    "PersistentAllocDesc.size",
        invariant: "must be greater than zero",
        debug_name: "bad_size",
    );

    gpu::compat::PersistentAllocDesc bad_align = {
        .size = 64, .align = 3, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "bad_align",
    };
    before = capture.calls;
    fault align_fault = @catch(gpu::compat::alloc_persistent_span(&device, &bad_align));
    assert_persistent_rejection(
        capture:    &capture,
        before:     before,
        returned:   align_fault,
        field:      "PersistentAllocDesc.align",
        invariant:  "must be zero or a power of two",
        debug_name: "bad_align",
    );

    gpu::compat::PersistentAllocDesc bad_kind = {
        .size = 64, .align = 16, .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "bad_kind",
    };
    before = capture.calls;
    fault kind_fault = @catch(gpu::compat::alloc_persistent_span(&device, &bad_kind));
    assert_persistent_rejection(
        capture:    &capture,
        before:     before,
        returned:   kind_fault,
        field:      "PersistentAllocDesc.memory_kind",
        invariant:  "must be PERSISTENT_UPLOAD",
        debug_name: "bad_kind",
    );

    gpu::compat::PersistentAllocDesc bad_usage = {
        .size = 64, .align = 16, .usage = { .vertex },
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "bad_usage",
    };
    before = capture.calls;
    fault usage_fault = @catch(gpu::compat::alloc_persistent_span(&device, &bad_usage));
    assert_persistent_rejection(
        capture:    &capture,
        before:     before,
        returned:   usage_fault,
        field:      "PersistentAllocDesc.usage",
        invariant:  "contains unsupported persistent-arena flags",
        debug_name: "bad_usage",
    );
}

fn void persistent_diagnostic_callback_preserves_fault_parity() @test {
    PersistentDiagnosticCapture capture;
    gpu::compat::DeviceDesc callback_desc = {
        .backend = gpu::compat::BackendKind.VULKAN, .debug_callback = &capture_persistent_diagnostic,
        .debug_user_data = &capture, .frames_in_flight = 2,
    };
    gpu::compat::Device callback_device = gpu::compat::create_device(&callback_desc)!!;
    gpu::compat::PersistentAllocDesc rejected_desc = {
        .size = 0, .align = 16, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };
    uint before = capture.calls;
    fault callback_fault = @catch(gpu::compat::alloc_persistent_span(&callback_device, &rejected_desc));
    assert(callback_fault == gpu::compat::INVALID_ARGUMENT, "callback path keeps INVALID_ARGUMENT");
    assert(capture.calls == before + 1, "callback path emits exactly once");
    gpu::compat::destroy_device(&callback_device)!!;

    gpu::compat::DeviceDesc plain_desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device plain_device = gpu::compat::create_device(&plain_desc)!!;
    fault plain_fault = @catch(gpu::compat::alloc_persistent_span(&plain_device, &rejected_desc));
    gpu::compat::destroy_device(&plain_device)!!;
    assert(plain_fault == callback_fault, "callback presence does not change persistent allocation fault");
}

fn void custom_persistent_arena_size_reported() @test {
    gpu::compat::DeviceDesc desc = {
        .backend               = gpu::compat::BackendKind.VULKAN,
        .enable_validation     = true,
        .frames_in_flight      = 2,
        .persistent_arena_size = 128 << 20,
        .application_name      = "gpu_persistent_size_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::PersistentArenaStats stats = gpu::compat::get_persistent_stats(&device)!!;
    assert(stats.capacity_bytes == 128 << 20, "custom persistent capacity reported in stats");
}

fn gpu::compat::Device make_persistent_usage_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_persistent_usage_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

<* A transfer-usage persistent span is a real cmd_copy_buffer endpoint: two
   spans backed by the same arena buffer, copied disjointly. *>
fn void transfer_usage_persistent_span_copies() @test {
    gpu::compat::Device device = make_persistent_usage_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::PersistentAllocDesc src_desc = {
        .size = 64, .align = 16, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "src",
    };
    gpu::compat::GpuSpan src = gpu::compat::alloc_persistent_span(&device, &src_desc)!!;
    defer gpu::compat::free_persistent_span(&device, src)!!;
    gpu::compat::PersistentAllocDesc dst_desc = {
        .size = 64, .align = 16, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "dst",
    };
    gpu::compat::GpuSpan dst = gpu::compat::alloc_persistent_span(&device, &dst_desc)!!;
    defer gpu::compat::free_persistent_span(&device, dst)!!;

    char* src_bytes = (char*)src.cpu;
    for (usz i = 0; i < 64; i++) src_bytes[i] = (char)(i % 251);
    gpu::compat::flush_buffer(&device, src.buffer, src.offset, src.size)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferCopyDesc copy = { .src = src.buffer, .dst = dst.buffer, .src_offset = src.offset, .dst_offset = dst.offset, .size = 64 };
    gpu::compat::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = dst.buffer, .offset = dst.offset, .size = dst.size,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;

    gpu::compat::invalidate_buffer(&device, dst.buffer, dst.offset, dst.size)!!;
    char* dst_bytes = (char*)dst.cpu;
    for (usz i = 0; i < 64; i++) {
        assert(dst_bytes[i] == (char)(i % 251), "transfer-usage persistent span copies intact bytes");
    }
}

<* resolve_indirect_span rejects buffers without the indirect bit, so this
   fails pre-fix and passes only once the arena carries the superset. *>
fn void indirect_usage_persistent_span_dispatches() @test {
    gpu::compat::Device device = make_persistent_usage_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc shader_desc = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = PA_DOUBLE_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle shader = gpu::compat::create_shader(&device, &shader_desc)!!;
    defer gpu::compat::destroy_shader(&device, shader)!!;
    gpu::compat::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    const uint COUNT = 4;
    gpu::compat::PersistentAllocDesc io_desc = {
        .size = COUNT * 4 * 2, .align = 16, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "io",
    };
    gpu::compat::GpuSpan io = gpu::compat::alloc_persistent_span(&device, &io_desc)!!;
    defer gpu::compat::free_persistent_span(&device, io)!!;
    float* values = (float*)io.cpu;
    for (uint i = 0; i < COUNT; i++) values[i] = (float)i;

    gpu::compat::PersistentAllocDesc root_desc = {
        .size = ComputeRoot::size, .align = 16, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "root",
    };
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_persistent_span(&device, &root_desc)!!;
    defer gpu::compat::free_persistent_span(&device, root_span)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = io.gpu;
    root.output_gpu = (gpu::compat::GpuAddress)((ulong)io.gpu + COUNT * 4);
    root.count      = COUNT;

    gpu::compat::PersistentAllocDesc args_desc = {
        .size = gpu::compat::DispatchIndirectCommand::size, .align = 4, .usage = { .indirect }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "dispatch_args",
    };
    gpu::compat::GpuSpan args_span = gpu::compat::alloc_persistent_span(&device, &args_desc)!!;
    defer gpu::compat::free_persistent_span(&device, args_span)!!;
    gpu::compat::DispatchIndirectCommand* args = (gpu::compat::DispatchIndirectCommand*)args_span.cpu;
    *args = { .x = 1, .y = 1, .z = 1 };
    gpu::compat::flush_buffer(&device, args_span.buffer, args_span.offset, args_span.size)!!;
    gpu::compat::flush_buffer(&device, root_span.buffer, root_span.offset, root_span.size)!!;
    gpu::compat::flush_buffer(&device, io.buffer, io.offset, io.size)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        args:     args_span,
    )!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = io.buffer, .offset = io.offset, .size = io.size,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;

    gpu::compat::invalidate_buffer(&device, io.buffer, io.offset, io.size)!!;
    float* doubled = &values[COUNT];
    for (uint i = 0; i < COUNT; i++) {
        assert(doubled[i] == (float)i * 2.0f, "indirect-usage persistent span drove cmd_dispatch_indirect");
    }
}

<* Usage bits outside the arena's superset fault at allocation time instead
   of surfacing as a Vulkan validation error at bind time. *>
fn void unsupported_usage_persistent_span_faults() @test {
    gpu::compat::Device device = make_persistent_usage_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::PersistentAllocDesc vertex_desc = {
        .size = 64, .align = 16, .usage = { .vertex }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };
    fault vertex_fault = @catch(gpu::compat::alloc_persistent_span(&device, &vertex_desc));
    assert(vertex_fault == gpu::compat::INVALID_ARGUMENT, "vertex-bit usage request faults INVALID_ARGUMENT");

    gpu::compat::PersistentAllocDesc shared_desc = {
        .size = 64, .align = 16, .usage = { .shared_queues }, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };
    gpu::compat::GpuSpan shared_span = gpu::compat::alloc_persistent_span(&device, &shared_desc)!!;
    gpu::compat::free_persistent_span(&device, shared_span)!!;

    // Empty usage stays valid (storage-style default).
    gpu::compat::PersistentAllocDesc empty_desc = { .size = 64, .align = 16, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    gpu::compat::GpuSpan empty_span = gpu::compat::alloc_persistent_span(&device, &empty_desc)!!;
    gpu::compat::free_persistent_span(&device, empty_span)!!;
}

fn void persistent_free_rejections_report_exact_context() @test {
    PersistentDiagnosticCapture capture;
    gpu::compat::DeviceDesc device_desc = {
        .backend          = gpu::compat::BackendKind.VULKAN,
        .debug_callback   = &capture_persistent_diagnostic,
        .debug_user_data  = &capture,
        .frames_in_flight = 2,
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::PersistentAllocDesc alloc_desc = {
        .size        = 64,
        .align       = 16,
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };
    gpu::compat::GpuSpan span = gpu::compat::alloc_persistent_span(&device, &alloc_desc)!!;

    gpu::compat::GpuSpan wrong_buffer = span;
    wrong_buffer.buffer = gpu::compat::BUFFER_HANDLE_INVALID;
    uint before = capture.calls;
    fault wrong_fault = @catch(gpu::compat::free_persistent_span(&device, wrong_buffer));
    assert(wrong_fault == gpu::compat::INVALID_ARGUMENT, "wrong persistent backing keeps INVALID_ARGUMENT");
    assert(capture.calls == before + 1, "wrong persistent backing emits exactly once");
    assert(capture.has_fault && capture.public_fault == wrong_fault, "wrong backing diagnostic keeps its fault");
    assert(capture.resource_kind == gpu::compat::DebugResourceKind.PERSISTENT_SPAN,
        "wrong backing diagnostic identifies a persistent span");
    assert(!capture.has_public_identity, "wrong backing diagnostic invents no identity");
    assert(((ZString)&capture.operation[0]).str_view() == "free_persistent_span",
        "wrong backing diagnostic names free_persistent_span");
    assert(((ZString)&capture.rejected_field[0]).str_view() == "GpuSpan.buffer",
        "wrong backing diagnostic identifies GpuSpan.buffer");

    gpu::compat::free_persistent_span(&device, span)!!;
    before = capture.calls;
    fault double_free = @catch(gpu::compat::free_persistent_span(&device, span));
    assert(double_free == gpu::compat::INVALID_HANDLE, "double persistent free keeps INVALID_HANDLE");
    assert(capture.calls == before + 1, "double persistent free emits exactly once");
    assert(capture.has_fault && capture.public_fault == double_free, "double-free diagnostic keeps its fault");
    assert(!capture.has_public_identity, "double-free diagnostic invents no identity");
    assert(((ZString)&capture.rejected_field[0]).str_view() == "GpuSpan.offset",
        "double-free diagnostic identifies GpuSpan.offset");
}

fn void persistent_capacity_failure_reports_backend_result() @test {
    PersistentDiagnosticCapture capture;
    gpu::compat::DeviceDesc device_desc = {
        .backend               = gpu::compat::BackendKind.VULKAN,
        .persistent_arena_size = 64,
        .debug_callback        = &capture_persistent_diagnostic,
        .debug_user_data       = &capture,
        .frames_in_flight      = 2,
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::PersistentAllocDesc full_desc = {
        .size        = 64,
        .align       = 1,
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };
    gpu::compat::GpuSpan full = gpu::compat::alloc_persistent_span(&device, &full_desc)!!;
    defer gpu::compat::free_persistent_span(&device, full)!!;
    gpu::compat::PersistentAllocDesc overflow_desc = {
        .size        = 1,
        .align       = 1,
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
    };

    uint before = capture.calls;
    fault full_fault = @catch(gpu::compat::alloc_persistent_span(&device, &overflow_desc));
    assert(full_fault == gpu::compat::ARENA_FULL, "persistent capacity keeps ARENA_FULL");
    assert(capture.calls == before + 1, "persistent capacity emits exactly once");
    assert(capture.category.backend, "persistent capacity is a backend diagnostic");
    assert(capture.has_fault && capture.public_fault == full_fault,
        "persistent capacity diagnostic keeps its fault");
    assert(((ZString)&capture.operation[0]).str_view() == "alloc_persistent_span",
        "persistent capacity diagnostic names alloc_persistent_span");
    assert(capture.validation_id_number == (int)vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
        "persistent capacity carries the raw backend result");
}
