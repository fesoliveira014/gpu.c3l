module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;
import std::thread;
import std::io;
import std::core::mem::allocators;

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
    assert(state.pending_inits.count() == 1, "creation queues the initialization");
    gpu::destroy_texture(&device, dropped)!!;
    assert(state.pending_inits.count() == 0, "destruction dequeues it");

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
    assert(state.pending_inits.count() == 0, "the submit drained the pending list");
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
    assert(state.pending_inits.count() == 1, "a failed submit returns the texture to the pending list");
    gpu::internal::vk::CommandAllocatorSlot* internal =
        state.queue_completions[queue.id].internal_allocator;
    assert(internal.live_executables == 0 && internal.live_recordings == 0,
        "the failed submit discards its internal lists");
    gpu::CompletionPoint point = gpu::submit(queue, &desc)!!;
    assert(state.pending_inits.count() == 0, "the retry drains the pending list");
    gpu::wait_completion(point)!!;
}

const uint INIT_CAPTURE_CALLS = 16;
const uint INIT_CAPTURE_WAITS = 8;

struct CapturedSubmit {
    vk::Queue                         queue;
    uint                              command_buffers;
    uint                              wait_count;
    vk::Semaphore[INIT_CAPTURE_WAITS] wait_semaphores;
    ulong[INIT_CAPTURE_WAITS]         wait_values;
    vk::PipelineStageFlags2[INIT_CAPTURE_WAITS] wait_stages;
    ulong                             signal_value;
}

struct SubmitCapture {
    Atomic{uint}                      calls;
    CapturedSubmit[INIT_CAPTURE_CALLS] submits;
    // When set, the first captured call on this queue blocks until released.
    vk::Queue                         gated_queue;
    Atomic{bool}                      gate_entered;
    Atomic{bool}                      gate_released;
}

struct InitSubmitWorker {
    vk::VkDeviceState* state;
    gpu::Queue         queue;
    gpu::CompletionPoint point;
    Atomic{bool}       done;
}

SubmitCapture submit_capture;

fn void reset_submit_capture() {
    submit_capture.calls.store(0);
    submit_capture.submits = {};
    submit_capture.gated_queue = (vk::Queue)null;
    submit_capture.gate_entered.store(false);
    submit_capture.gate_released.store(false);
}

<* Records each native submit and forwards it to the driver. *>
fn vk::Result capture_init_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    uint index = submit_capture.calls.add(1);
    if (index < INIT_CAPTURE_CALLS && submit_count == 1) {
        CapturedSubmit* captured = &submit_capture.submits[index];
        captured.queue = queue;
        captured.command_buffers = submits[0].command_buffer_info_count;
        captured.wait_count = submits[0].wait_semaphore_info_count;
        for (uint i = 0; i < captured.wait_count && i < INIT_CAPTURE_WAITS; i++) {
            captured.wait_semaphores[i] = submits[0].p_wait_semaphore_infos[i].semaphore;
            captured.wait_values[i] = submits[0].p_wait_semaphore_infos[i].value;
            captured.wait_stages[i] = submits[0].p_wait_semaphore_infos[i].stage_mask;
        }
        uint signals = submits[0].signal_semaphore_info_count;
        if (signals > 0) {
            captured.signal_value = submits[0].p_signal_semaphore_infos[signals - 1].value;
        }
    }
    if (queue == submit_capture.gated_queue
        && !submit_capture.gate_entered.load()) {
        submit_capture.gate_entered.store(true);
        while (!submit_capture.gate_released.load()) thread::yield();
    }
    return vk::queue_submit2(queue, submit_count, submits, fence);
}

fn bool captured_waits_on(CapturedSubmit* captured, vk::Semaphore timeline, ulong value) {
    for (uint i = 0; i < captured.wait_count && i < INIT_CAPTURE_WAITS; i++) {
        if (captured.wait_semaphores[i] == timeline
            && captured.wait_values[i] == value
            && captured.wait_stages[i]
                == (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT) {
            return true;
        }
    }
    return false;
}

fn bool captured_waits_on_timeline(CapturedSubmit* captured, vk::Semaphore timeline) {
    for (uint i = 0; i < captured.wait_count && i < INIT_CAPTURE_WAITS; i++) {
        if (captured.wait_semaphores[i] == timeline) return true;
    }
    return false;
}

<* A unified device whose graphics, compute and transfer roles sit on three
   distinct queue families, or a reported skip when the adapter has none. *>
fn gpu::Device? make_split_family_device(VulkanValidationCapture* capture) {
    gpu::RuntimeDesc desc = {
        .enable_vulkan_validation = true,
        .application_name         = "gpu_unified_layouts_test",
        .debug_callback           = &capture_vulkan_validation_error,
        .debug_user_data          = capture,
    };
    gpu::DeviceDesc device_desc = {
        .unified_layouts = true,
        .queues = {
            .required       = { .graphics, .compute, .transfer },
            .distinct_roles = { .compute, .transfer },
        },
    };
    gpu::Device? device = create_test_device(&desc, &device_desc);
    if (catch err = device) {
        if (err == gpu::UNSUPPORTED_FEATURE) {
            io::printfn("distinct compute and transfer queues unavailable; split-family initialization check skipped");
        }
        return err~;
    }
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    if (state.graphics_family == state.transfer_family
        || state.compute_family == state.transfer_family
        || state.graphics_family == state.compute_family) {
        io::printfn("graphics, compute and transfer share a queue family; split-family initialization check skipped");
        destroy_test_device(&device)!!;
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return device;
}

fn gpu::TextureHandle make_init_texture(gpu::Device* device, gpu::QueueRoles access, ZString name) {
    gpu::TextureDesc desc = {
        .access     = access,
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
        .debug_name = name,
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::CompletionPoint submit_empty_captured(vk::VkDeviceState* state, gpu::Queue queue) {
    gpu::SubmitDesc empty = {};
    return vk::vk_submit_with(
        state:     state,
        queue:     queue,
        desc:      &empty,
        submit_fn: &capture_init_submit,
    )!!;
}

fn void unified_init_skips_queues_outside_texture_access() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    validation.errors = 0;
    reset_submit_capture();

    gpu::TextureHandle unshared = make_init_texture(&device, { .graphics, .compute }, "unshared_init");
    defer gpu::destroy_texture(&device, unshared)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CompletionPoint transfer_done = submit_empty_captured(state, transfer);
    gpu::wait_completion(transfer_done)!!;
    assert(state.pending_inits.count() == 1,
        "a transfer submit leaves a graphics and compute texture pending");
    assert(submit_capture.calls.load() == 1 && submit_capture.submits[0].command_buffers == 0,
        "the transfer submit records no initialization batch");
    assert(validation.errors == 0,
        "no queue family outside the texture's sharing set touches it");

    gpu::CompletionPoint graphics_done = submit_empty_captured(state, graphics);
    assert(state.pending_inits.count() == 0, "the graphics submit takes the texture");
    assert(submit_capture.calls.load() == 3 && submit_capture.submits[1].command_buffers == 1
        && submit_capture.submits[1].wait_count == 0,
        "the initialization is a batch of its own that waits on nothing");
    assert(state.queue_completions[graphics.id].init_sequence_for[transfer.id].load() == 0,
        "a texture the transfer queue cannot access orders nothing there");
    gpu::wait_completion(graphics_done)!!;
    assert(validation.errors == 0, "the graphics initialization is validation clean");
}

fn void unified_init_orders_other_queues_sharing_the_texture() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    validation.errors = 0;
    reset_submit_capture();

    gpu::TextureHandle shared = make_init_texture(&device, { .graphics, .transfer }, "shared_init");
    defer gpu::destroy_texture(&device, shared)!!;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    vk::QueueCompletionState* graphics_state = &state.queue_completions[graphics.id];
    gpu::CompletionPoint graphics_done = submit_empty_captured(state, graphics);
    assert(submit_capture.calls.load() == 2, "graphics submits the initialization, then its own batch");
    ulong init_sequence = submit_capture.submits[0].signal_value;
    assert(init_sequence + 1 == gpu::internal::completion_point_sequence(graphics_done),
        "the initialization takes the point before the caller's batch");
    assert(graphics_state.init_sequence_for[transfer.id].load() == init_sequence,
        "the point is published for the transfer queue");

    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         TEX_SIZE * TEX_SIZE * 4,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .transfer },
        debug_name:   "shared_init_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, transfer)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::BufferTextureCopyDesc copy = { .src = staging.span, .texture = shared };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)!! };
    defer discard_test_executable(&lists[0])!!;
    gpu::SubmitDesc copy_submit = { .command_lists = lists[..] };
    gpu::CompletionPoint copy_done = vk::vk_submit_with(
        state:     state,
        queue:     transfer,
        desc:      &copy_submit,
        submit_fn: &capture_init_submit,
    )!!;
    assert(submit_capture.calls.load() == 3, "the transfer queue has nothing to initialize");
    assert(captured_waits_on(&submit_capture.submits[2], graphics_state.timeline, init_sequence),
        "the copy waits for the graphics queue's initialization at every stage");

    gpu::CompletionPoint unretired = submit_empty_captured(state, transfer);
    assert(captured_waits_on(&submit_capture.submits[3], graphics_state.timeline, init_sequence),
        "later transfer batches keep the wait until the initialization retires");

    gpu::CompletionWait[1] waits = {{ .point = graphics_done, .before = { .all } }};
    gpu::SubmitDesc waiting = { .completion_waits = waits[..] };
    gpu::CompletionPoint waited = vk::vk_submit_with(
        state:     state,
        queue:     transfer,
        desc:      &waiting,
        submit_fn: &capture_init_submit,
    )!!;
    assert(submit_capture.submits[4].wait_count == 1
        && submit_capture.submits[4].wait_values[0] == gpu::internal::completion_point_sequence(graphics_done),
        "an application wait at every stage on a later graphics point covers the initialization");

    gpu::wait_completion(graphics_done)!!;
    assert(graphics_state.retired_sequence.load() >= init_sequence);
    gpu::CompletionPoint retired = submit_empty_captured(state, transfer);
    assert(!captured_waits_on_timeline(&submit_capture.submits[5], graphics_state.timeline),
        "a retired initialization needs no wait");
    gpu::wait_completion(copy_done)!!;
    gpu::wait_completion(unretired)!!;
    gpu::wait_completion(retired)!!;
    gpu::wait_completion(waited)!!;
    assert(validation.errors == 0, "the cross-queue initialization is validation clean");
}

fn void unified_init_destroyed_while_waiting_for_a_compatible_queue() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    validation.errors = 0;
    reset_submit_capture();

    gpu::TextureHandle unshared = make_init_texture(&device, { .graphics, .compute }, "dropped_init");
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CompletionPoint transfer_done = submit_empty_captured(state, transfer);
    assert(state.pending_inits.count() == 1);
    gpu::destroy_texture(&device, unshared)!!;
    assert(state.pending_inits.count() == 0
        && !state.pending_inits.has_outstanding({ .graphics, .compute, .transfer }),
        "destruction removes the pending texture and its outstanding count");
    gpu::CompletionPoint graphics_done = submit_empty_captured(state, graphics);
    assert(submit_capture.calls.load() == 2 && submit_capture.submits[1].command_buffers == 0,
        "the next graphics submit has nothing to initialize");
    gpu::wait_completion(transfer_done)!!;
    gpu::wait_completion(graphics_done)!!;
    assert(validation.errors == 0);
}

fn vk::Result fail_second_native_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    if (submit_capture.calls.add(1) == 1) return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
    return vk::queue_submit2(queue, submit_count, submits, fence);
}

fn void unified_failed_init_restores_only_its_textures() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    validation.errors = 0;
    reset_submit_capture();

    gpu::TextureHandle unshared = make_init_texture(&device, { .graphics, .compute }, "restore_unshared");
    defer gpu::destroy_texture(&device, unshared)!!;
    gpu::TextureHandle transfer_only = make_init_texture(&device, { .transfer }, "restore_transfer");
    defer gpu::destroy_texture(&device, transfer_only)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    vk::QueueCompletionState* transfer_state = &state.queue_completions[transfer.id];

    gpu::SubmitDesc empty = {};
    fault failed_init = @catch(vk::vk_submit_with(
        state:     state,
        queue:     transfer,
        desc:      &empty,
        submit_fn: &fail_native_submit,
    ));
    assert(failed_init == gpu::OUT_OF_HOST_MEMORY, "a failed initialization batch fails the submit");
    assert(state.pending_inits.count() == 2, "the failed submit puts its texture back beside the untouched one");
    assert(state.pending_inits.compatible_count({ .transfer }) == 1
        && state.pending_inits.compatible_count({ .graphics }) == 1);
    foreach (&sequence : transfer_state.init_sequence_for) {
        assert(sequence.load() == 0, "a failed initialization publishes nothing");
    }
    assert(transfer_state.internal_allocator.live_executables == 0
        && transfer_state.internal_allocator.live_recordings == 0,
        "the failed initialization discards its list");

    reset_submit_capture();
    fault failed_batch = @catch(vk::vk_submit_with(
        state:     state,
        queue:     transfer,
        desc:      &empty,
        submit_fn: &fail_second_native_submit,
    ));
    assert(failed_batch == gpu::OUT_OF_HOST_MEMORY, "a failure of the caller's batch fails the submit");
    assert(state.pending_inits.count() == 1 && state.pending_inits.compatible_count({ .graphics }) == 1,
        "the initialization submitted before the failed batch stays done");
    assert(!state.pending_inits.has_outstanding({ .transfer }),
        "the published initialization is settled");
    gpu::CompletionPoint graphics_done = gpu::submit(graphics, &empty)!!;
    assert(state.pending_inits.count() == 0);
    gpu::wait_completion(graphics_done)!!;
    gpu::CompletionPoint transfer_done = gpu::submit(transfer, &empty)!!;
    gpu::wait_completion(transfer_done)!!;
    assert(validation.errors == 0);
}

fn int init_submit_worker(void* raw_args) {
    InitSubmitWorker* args = (InitSubmitWorker*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::SubmitDesc empty = {};
        args.point = vk::vk_submit_with(
            state:     args.state,
            queue:     args.queue,
            desc:      &empty,
            submit_fn: &capture_init_submit,
        )!!;
        args.done.store(true);
    };
    return 0;
}

fn void unified_init_publication_is_atomic_to_other_queues() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    validation.errors = 0;
    reset_submit_capture();

    gpu::TextureHandle shared = make_init_texture(&device, { .graphics, .transfer }, "racing_init");
    defer gpu::destroy_texture(&device, shared)!!;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    vk::QueueCompletionState* graphics_state = &state.queue_completions[graphics.id];
    submit_capture.gated_queue = graphics_state.queue;

    InitSubmitWorker graphics_worker = { .state = state, .queue = graphics };
    Thread graphics_thread;
    graphics_thread.create(&init_submit_worker, &graphics_worker)!!;
    while (!submit_capture.gate_entered.load()) thread::yield();

    InitSubmitWorker transfer_worker = { .state = state, .queue = transfer };
    Thread transfer_thread;
    transfer_thread.create(&init_submit_worker, &transfer_worker)!!;
    thread::sleep_ms(50)!!;
    assert(!transfer_worker.done.load(),
        "a transfer submit waits while graphics holds an unpublished initialization it shares");
    submit_capture.gate_released.store(true);
    assert(graphics_thread.join()!! == 0 && transfer_thread.join()!! == 0);

    ulong init_sequence = submit_capture.submits[0].signal_value;
    assert(submit_capture.submits[0].queue == graphics_state.queue
        && submit_capture.submits[0].command_buffers == 1,
        "graphics took the initialization");
    bool transfer_waited = false;
    for (uint i = 1; i < submit_capture.calls.load() && i < INIT_CAPTURE_CALLS; i++) {
        if (submit_capture.submits[i].queue != state.queue_completions[transfer.id].queue) continue;
        transfer_waited = captured_waits_on(&submit_capture.submits[i], graphics_state.timeline, init_sequence);
    }
    assert(transfer_waited, "the transfer batch waits for the published initialization");
    gpu::wait_completion(graphics_worker.point)!!;
    gpu::wait_completion(transfer_worker.point)!!;
    assert(validation.errors == 0);
}

struct InitStressWorker {
    gpu::Device* device;
    gpu::Queue   queue;
    bool         create_textures;
    uint         iterations;
}

fn int init_stress_worker(void* raw_args) {
    InitStressWorker* args = (InitStressWorker*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::SubmitDesc empty = {};
        gpu::CompletionPoint last;
        for (uint i = 0; i < args.iterations; i++) {
            if (args.create_textures) {
                gpu::QueueRoles access = i % 2 == 0
                    ? (gpu::QueueRoles){ .graphics, .transfer }
                    : (gpu::QueueRoles){ .graphics, .compute };
                gpu::TextureHandle texture = make_init_texture(args.device, access, "stress_init");
                last = gpu::submit(args.queue, &empty)!!;
                gpu::wait_completion(last)!!;
                gpu::destroy_texture(args.device, texture)!!;
            } else {
                last = gpu::submit(args.queue, &empty)!!;
            }
        }
        if (last.is_valid()) gpu::wait_completion(last)!!;
    };
    return 0;
}

fn void unified_init_concurrent_queues_stay_validation_clean() @test {
    VulkanValidationCapture validation;
    gpu::Device? created = make_split_family_device(&validation);
    if (catch err = created) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "only a missing capability skips the check");
        return;
    }
    gpu::Device device = created;
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    InitStressWorker creator = {
        .device          = &device,
        .queue           = graphics,
        .create_textures = true,
        .iterations      = 64,
    };
    InitStressWorker submitter = {
        .device     = &device,
        .queue      = transfer,
        .iterations = 256,
    };
    Thread creator_thread;
    Thread submitter_thread;
    creator_thread.create(&init_stress_worker, &creator)!!;
    submitter_thread.create(&init_stress_worker, &submitter)!!;
    assert(creator_thread.join()!! == 0 && submitter_thread.join()!! == 0);
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    assert(state.pending_inits.count() == 0
        && !state.pending_inits.has_outstanding({ .graphics, .compute, .transfer }),
        "every texture was initialized or removed and settled");
    assert(validation.errors == 0, "concurrent graphics and transfer submits stay validation clean");
}
