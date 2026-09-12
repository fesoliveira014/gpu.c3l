module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;

fn gpu::Device make_unified_device(VulkanValidationCapture* capture = null) {
    gpu::RuntimeDesc desc = {
        .enable_vulkan_validation = true,
        .application_name         = "gpu_unified_layouts_test",
        .debug_callback           = capture == null ? null : &capture_vulkan_validation_error,
        .debug_user_data          = capture,
    };
    gpu::DeviceDesc device_desc = { .unified_layouts = true };
    return create_test_device(&desc, &device_desc)!!;
}

fn gpu::TextureHandle make_unified_heap_texture(gpu::Device* device, ZString name) {
    gpu::TextureDesc desc = {
        .access     = { .graphics, .compute, .transfer },
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = name,
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void unified_layouts_caps_and_general_rejection() @test {
    gpu::RuntimeDesc runtime_desc = {
        .enable_vulkan_validation = true,
        .application_name         = "gpu_unified_layouts_test",
    };
    gpu::Device explicit_device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&explicit_device)!!;
    gpu::DeviceCaps explicit_caps = gpu::get_device_caps(&explicit_device)!!;
    assert(explicit_caps.unified_layouts || !explicit_caps.unified_layouts_optimal,
        "the optimal cap needs the mode");
    gpu::TextureHandle texture = make_unified_heap_texture(&explicit_device, "explicit_general");
    defer gpu::destroy_texture(&explicit_device, texture)!!;
    gpu::Queue queue = gpu::get_queue(&explicit_device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&explicit_device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier general = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = { .layout = gpu::TextureLayout.GENERAL, .stages = { .compute }, .access = { .write } },
    };
    if (explicit_caps.unified_layouts) gpu::cmd_texture_barrier(&cmd, &general)!!;

    gpu::Device unified = make_unified_device();
    defer destroy_test_device(&unified)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&unified)!!;
    assert(caps.unified_layouts, "the mode is active when requested");
    assert(!caps.unified_layouts_optimal || caps.unified_layouts,
        "the optimal cap implies the mode");
    gpu::DeviceDesc requested = { .unified_layouts = true };
    gpu::DeviceSupport support = gpu::supports_device_desc(
        &gpu::internal::device_slots[unified.index].data.adapter,
        &requested,
    )!!;
    assert(support.supported, "the mode is satisfiable on every device");
}

fn void unified_barrier_lowers_to_general() @test {
    gpu::Device device = make_unified_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_unified_heap_texture(&device, "unified_barrier");
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    TextureBarrierEmitCapture capture;
    vk::BarrierEmitOps ops = {
        .user_data = &capture,
        .emit      = &capture_texture_barrier_emit,
    };
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  gpu::sampled_at({ .fragment_shader }),
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    vk::vk_cmd_texture_barrier_with_ops(vk::resolve_command(&cmd)!!, &barrier, &ops)!!;
    assert(capture.calls == 1
        && capture.native.old_layout == vk::ImageLayout.GENERAL
        && capture.native.new_layout == vk::ImageLayout.GENERAL,
        "both sides lower to GENERAL");
    assert(capture.native.src_stage_mask == (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
        && capture.native.src_access_mask == (vk::AccessFlags2)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT
        && capture.native.dst_stage_mask == (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        && capture.native.dst_access_mask == (vk::AccessFlags2)vk::AccessFlagBits2.ACCESS_2_MEMORY_WRITE_BIT,
        "stages come from the texture states and access from the read/write intent");

    gpu::TextureBarrier discard = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    vk::vk_cmd_texture_barrier_with_ops(vk::resolve_command(&cmd)!!, &discard, &ops)!!;
    assert(capture.calls == 2
        && capture.native.old_layout == vk::ImageLayout.UNDEFINED
        && capture.native.new_layout == vk::ImageLayout.GENERAL,
        "UNDEFINED keeps discard semantics");

    gpu::TextureBarrier sampled_write = gpu::texture_transition(
        texture: texture,
        before:  gpu::storage_at({ .compute }, { .write }),
        after:   { .layout = gpu::TextureLayout.SAMPLED, .stages = { .compute }, .access = { .read, .write } },
    )!!;
    vk::vk_cmd_texture_barrier_with_ops(vk::resolve_command(&cmd)!!, &sampled_write, &ops)!!;
    assert(capture.calls == 3, "layout-specific access rules are lifted in the mode");
    assert(capture.native.dst_access_mask
            == ((vk::AccessFlags2)vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT
                | (vk::AccessFlags2)vk::AccessFlagBits2.ACCESS_2_MEMORY_WRITE_BIT),
        "access follows the stated intent, not the ignored layout label");
}

fn void unified_storage_write_then_sampled_read() @test {
    VulkanValidationCapture validation;
    gpu::Device device = make_unified_device(&validation);
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
    gpu::TextureHandle texture = make_unified_heap_texture(&device, "unified_heap_target");
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView view = gpu::create_texture_view(&device, texture, null)!!;
    defer gpu::destroy_texture_view(&device, view)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         TEX_SIZE * TEX_SIZE * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "unified_sample_output",
    )!!;
    defer free_owned_span(&device, &output)!!;
    gpu::ShaderDesc write_shader = { .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc sample_shader = { .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ComputePipelineDesc write_desc = { .shader = write_shader };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_desc)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_desc = { .shader = sample_shader };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;
    OwnedSpan roots = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size + SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "unified_roots",
    )!!;
    defer free_owned_span(&device, &roots)!!;
    char* mapping = test_span_mapping(&device, roots.span).ptr;
    WriteRoot* write_root = (WriteRoot*)mapping;
    *write_root = { .texture_index = view.index, .width = TEX_SIZE, .height = TEX_SIZE };
    SampleRoot* sample_root = (SampleRoot*)(mapping + WriteRoot::size);
    *sample_root = {
        .output_gpu    = test_span_address(&device, output.span),
        .texture_index = view.index,
        .sampler_index = sampler,
        .width         = TEX_SIZE,
        .height        = TEX_SIZE,
    };
    gpu::flush_mapped_span(&device, roots.span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    const uint GROUPS = (TEX_SIZE + 7) / 8;
    bind_test_compute_pipeline(&cmd, write_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, roots.span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::TextureBarrier write_to_sample = gpu::texture_transition(
        texture: texture,
        before:  gpu::storage_at({ .compute }, { .write }),
        after:   gpu::sampled_at({ .compute }),
    )!!;
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!!;
    bind_test_compute_pipeline(&cmd, sample_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, roots.span.checked_subspan(WriteRoot::size, SampleRoot::size)!!),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::Barrier to_host = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&cmd, &to_host)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
    assert_heap_pattern(&device, output.span, TEX_SIZE);
    assert(validation.errors == 0, "storage write then sampled read in one interval is validation clean");
}

fn void unified_textures_initialize_at_submit() @test {
    VulkanValidationCapture validation;
    gpu::Device device = make_unified_device(&validation);
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
    gpu::internal::vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;

    gpu::TextureHandle dropped = make_unified_heap_texture(&device, "dropped_before_submit");
    assert(state.pending_init_count == 1, "creation queues the initialization");
    gpu::destroy_texture(&device, dropped)!!;
    assert(state.pending_init_count == 0, "destruction dequeues it");

    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator graphics_allocator = gpu::create_command_allocator(&device, graphics)!!;
    defer gpu::destroy_command_allocator(&graphics_allocator)!!;
    gpu::CommandList first = gpu::begin_commands(&graphics_allocator)!!;
    defer discard_test_commands(&first)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    defer discard_test_executable(&first_executable)!!;
    gpu::ExecutableCommandList[1] first_lists = { first_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::CompletionPoint first_done = gpu::submit(graphics, &first_submit)!!;

    gpu::TextureHandle texture = make_unified_heap_texture(&device, "unified_late_texture");
    defer if (texture.is_valid()) gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView view = gpu::create_texture_view(&device, texture, null)!!;
    gpu::ShaderDesc write_shader = { .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ComputePipelineDesc write_desc = { .shader = write_shader };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_desc)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    OwnedSpan root = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "unified_late_root",
    )!!;
    defer free_owned_span(&device, &root)!!;
    *(WriteRoot*)test_span_mapping(&device, root.span).ptr =
        { .texture_index = view.index, .width = TEX_SIZE, .height = TEX_SIZE };
    gpu::flush_mapped_span(&device, root.span)!!;

    gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator compute_allocator = gpu::create_command_allocator(&device, compute)!!;
    defer gpu::destroy_command_allocator(&compute_allocator)!!;
    gpu::CommandList second = gpu::begin_commands(&compute_allocator)!!;
    defer discard_test_commands(&second)!!;
    bind_test_compute_pipeline(&second, write_pipe);
    const uint GROUPS = (TEX_SIZE + 7) / 8;
    gpu::cmd_dispatch(
        commands: &second,
        root:     test_span_address(&device, root.span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    defer discard_test_executable(&second_executable)!!;
    gpu::ExecutableCommandList[1] second_lists = { second_executable };
    gpu::CompletionWait[1] waits = {{ .point = first_done, .before = { .compute } }};
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..], .completion_waits = waits[..] };
    gpu::CompletionPoint second_done = gpu::submit(compute, &second_submit)!!;
    assert(state.pending_init_count == 0, "the submit drained the pending list");
    gpu::destroy_texture_view(&device, view)!!;
    assert(@catch(gpu::destroy_texture(&device, texture)) == gpu::RESOURCE_IN_USE,
        "the pending initialization retains the texture until it retires");
    gpu::wait_completion(second_done)!!;
    gpu::wait_completion(first_done)!!;
    gpu::destroy_texture(&device, texture)!!;
    texture = {};
    assert(validation.errors == 0, "a texture created between submits is initialized by its first submit");
}

fn vk::Result fail_native_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submit_count;
    (void)submits;
    (void)fence;
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}

fn void unified_failed_submit_keeps_textures_pending() @test {
    gpu::Device device = make_unified_device();
    defer destroy_test_device(&device)!!;
    gpu::internal::vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    gpu::TextureHandle texture = make_unified_heap_texture(&device, "unified_retry_texture");
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)!! };
    defer discard_test_executable(&lists[0])!!;
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    assert(@catch(gpu::internal::vk::vk_submit_with(state, queue, &desc, &fail_native_submit))
        == gpu::OUT_OF_HOST_MEMORY, "the native fault propagates");
    assert(state.pending_init_count == 1, "a failed submit returns the texture to the pending list");
    gpu::internal::vk::CommandAllocatorSlot* internal =
        state.queue_completions[queue.id].internal_allocator;
    assert(internal.live_executables == 0 && internal.live_recordings == 0,
        "the failed submit discards its internal lists");
    gpu::CompletionPoint point = gpu::submit(queue, &desc)!!;
    assert(state.pending_init_count == 0, "the retry drains the pending list");
    gpu::wait_completion(point)!!;
}
