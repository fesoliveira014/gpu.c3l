module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;

// Headless (lavapipe): record a buffer copy from a host-written upload buffer to
// a readback buffer, submit signaling a timeline semaphore, wait, and verify the
// bytes. Assumes host-coherent memory (non-coherent flush/invalidate is later).

fn ulong command_submit_count(gpu::compat::Device* device) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(*device);
    return vk::submit_stats(state).queue_submits;
}


fn void command_list_is_small_owner_token() @test {
    assert(gpu::compat::CommandList::size <= 16, "CommandList stays within two machine words");
}

fn void copy_upload_to_readback() @test {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc up_desc = {
        .size = 256, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING, .debug_name = "upload",
    };
    gpu::compat::BufferHandle upload = gpu::compat::create_buffer(&device, &up_desc)!!;
    defer gpu::compat::destroy_buffer(&device, upload)!!;

    gpu::compat::BufferDesc rb_desc = {
        .size = 256, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK, .debug_name = "readback",
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;

    gpu::compat::GpuSpan up_span = gpu::compat::get_buffer_span(&device, upload)!!;
    char* src = (char*)up_span.cpu;
    for (uint i = 0; i < 256; i++) src[i] = (char)i;

    gpu::compat::SemaphoreDesc sem_desc = { .initial_value = (gpu::compat::SemaphoreValue)0 };
    gpu::compat::SemaphoreHandle sem = gpu::compat::create_semaphore(&device, &sem_desc)!!;
    defer gpu::compat::destroy_semaphore(&device, sem)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    gpu::compat::BufferCopyDesc copy = { .src = upload, .dst = readback, .src_offset = 0, .dst_offset = 0, .size = 256 };
    gpu::compat::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::compat::BufferBarrier bar = {
        .buffer = readback, .offset = 0, .size = 256,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::compat::end_commands(&cmd)!!;

    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SemaphoreSignal[1] sigs = { { .semaphore = sem, .value = (gpu::compat::SemaphoreValue)1, .stage = gpu::compat::Stage.TRANSFER } };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..], .signals = sigs[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::compat::SemaphoreValue)1,
        timeout_ns: 1_000_000_000,
    )!!;

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readback)!!;
    char* dst = (char*)rb_span.cpu;
    for (uint i = 0; i < 256; i++) assert(dst[i] == (char)i, "readback byte matches uploaded byte");

    gpu::compat::end_frame(&frame)!!;
}

fn void empty_command_list_submits() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&cmd)!!;

    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame)!!;
}

fn void command_list_owns_its_device_token_value() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::Device temporary = device;
    gpu::compat::CommandList commands = gpu::compat::begin_commands(&temporary, gpu::compat::QueueKind.GRAPHICS)!!;
    temporary = {};
    gpu::compat::end_commands(&commands)!!;

    gpu::compat::CommandList[1] lists = { commands };
    gpu::compat::SubmitDesc submit_desc = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit_desc)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void stale_command_owner_faults_invalid_handle() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::CommandList commands = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&commands)!!;
    gpu::compat::CommandList live_commands = commands;
    commands.device = {};

    gpu::compat::CommandList[1] stale_lists = { commands };
    gpu::compat::SubmitDesc stale_submit = { .command_lists = stale_lists[..] };
    fault rejected = @catch(gpu::compat::submit(&device, &stale_submit));
    assert(rejected == gpu::compat::INVALID_HANDLE, "stale command owner faults INVALID_HANDLE");

    gpu::compat::CommandList[1] live_lists = { live_commands };
    gpu::compat::SubmitDesc live_submit = { .command_lists = live_lists[..] };
    gpu::compat::submit(&device, &live_submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void copied_alias_cannot_resubmit_after_success() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::CommandList original = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&original)!!;
    gpu::compat::CommandList[1] copied = { original };
    gpu::compat::SubmitDesc first = { .command_lists = copied[..] };
    gpu::compat::submit(&device, &first)!!;
    gpu::compat::CommandList replacement = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&replacement)!!;
    assert(replacement.handle.index == original.handle.index, "the freed command slot is reused");
    assert(replacement.handle.generation != original.handle.generation,
        "slot reuse advances the command generation");


    ulong before = command_submit_count(&device);
    gpu::compat::CommandList[1] original_alias = { original };
    gpu::compat::SubmitDesc second = { .command_lists = original_alias[..] };
    fault rejected = @catch(gpu::compat::submit(&device, &second));
    assert(rejected == gpu::compat::INVALID_HANDLE, "a consumed alias faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "a consumed alias reaches no Vulkan queue submit");
    gpu::compat::CommandList[1] replacement_lists = { replacement };
    gpu::compat::SubmitDesc replacement_submit = { .command_lists = replacement_lists[..] };
    gpu::compat::submit(&device, &replacement_submit)!!;

    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void command_generation_wraps_without_poisoning_slot() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::CommandList nearly_wrapped = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&nearly_wrapped)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    state.commands.slots[nearly_wrapped.handle.index].generation = vk::HANDLE_GENERATION_MAX - 1;
    nearly_wrapped.handle.generation = vk::HANDLE_GENERATION_MAX - 1;

    gpu::compat::CommandList[1] first_lists = { nearly_wrapped };
    gpu::compat::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::compat::submit(&device, &first_submit)!!;

    gpu::compat::CommandList wrapped = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    assert(wrapped.handle.index == nearly_wrapped.handle.index, "wrapped slot is reused");
    assert(wrapped.handle.generation == 1, "command generation wraps to the first valid value");
    gpu::compat::end_commands(&wrapped)!!;
    gpu::compat::CommandList[1] wrapped_lists = { wrapped };
    gpu::compat::SubmitDesc wrapped_submit = { .command_lists = wrapped_lists[..] };
    gpu::compat::submit(&device, &wrapped_submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void duplicate_token_batch_rejects_without_consuming() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[2] duplicates = { cmd, cmd };
    gpu::compat::SubmitDesc bad = { .command_lists = duplicates[..] };
    ulong before = command_submit_count(&device);
    fault rejected = @catch(gpu::compat::submit(&device, &bad));
    assert(rejected == gpu::compat::COMMAND_RECORDING_ERROR, "duplicate tokens fault before submission");
    assert(command_submit_count(&device) == before, "duplicate-token rejection records no queue submit");

    gpu::compat::CommandList[1] one = { cmd };
    gpu::compat::SubmitDesc good = { .command_lists = one[..] };
    gpu::compat::submit(&device, &good)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void second_live_device_is_rejected() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device owner = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&owner)!!;
    fault second_device = @catch(gpu::compat::create_device(&desc));
    assert(second_device == gpu::compat::INVALID_RESOURCE_STATE, "only one device may be live");

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&owner, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&owner, &submit)!!;
    gpu::compat::wait_queue_idle(&owner, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn void abandoned_token_invalidates_when_pool_slot_resets() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 1 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList abandoned = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&abandoned)!!;
    gpu::compat::end_frame(&frame)!!;
    frame = gpu::compat::begin_frame(&device)!!;

    fault stale = @catch(gpu::compat::end_commands(&abandoned));
    assert(stale == gpu::compat::INVALID_HANDLE, "pool reset invalidates an abandoned command token");
    gpu::compat::end_frame(&frame)!!;
}

fn void live_command_record_blocks_context_destruction() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::RecordingContextHandle context = gpu::compat::create_recording_context(&device)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS, context)!!;
    fault live = @catch(gpu::compat::destroy_recording_context(&device, context));
    assert(live == gpu::compat::INVALID_RESOURCE_STATE, "a context with a live command record cannot be destroyed");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::destroy_recording_context(&device, context)!!;
}

fn void bad_command_state_rejected() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&cmd)!!;

    // Recording after end must be rejected.
    gpu::compat::GlobalBarrier gb = {
        .before_stage = gpu::compat::Stage.HOST, .after_stage = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE, .after_hazard = gpu::compat::Hazard.TRANSFER_READ,
    };
    fault rejected = @catch(gpu::compat::cmd_global_barrier(&cmd, &gb));
    assert(rejected == gpu::compat::COMMAND_RECORDING_ERROR, "recording after end returns COMMAND_RECORDING_ERROR");
    gpu::compat::end_frame(&frame)!!;
}

fn void texture_barrier_faults() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::TextureBarrier bar = {
        .texture = gpu::compat::TEXTURE_HANDLE_INVALID,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.SHADER_READ,
        .old_layout = gpu::compat::TextureLayout.TRANSFER_DST, .new_layout = gpu::compat::TextureLayout.SHADER_READ,
    };
    fault barrier_fault = @catch(gpu::compat::cmd_texture_barrier(&cmd, &bar));
    assert(barrier_fault == gpu::compat::INVALID_HANDLE, "barrier on an invalid texture handle faults INVALID_HANDLE");
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame)!!;
}

// vkWaitSemaphores with timeout 0 is a spec-guaranteed poll: it returns
// immediately, so these are deterministic on lavapipe.

fn void wait_timeout_zero_on_unreached_value_faults() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SemaphoreDesc sem_desc = { .initial_value = (gpu::compat::SemaphoreValue)0 };
    gpu::compat::SemaphoreHandle sem = gpu::compat::create_semaphore(&device, &sem_desc)!!;
    defer gpu::compat::destroy_semaphore(&device, sem)!!;

    fault timeout_fault = @catch(gpu::compat::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::compat::SemaphoreValue)1,
        timeout_ns: 0,
    ));
    assert(timeout_fault == gpu::compat::WAIT_TIMEOUT, "timeout-0 wait on an unreached value faults WAIT_TIMEOUT");
}

// Lazy pool resets: begin_frame's reset loop only pays for pools a context
// actually dirtied since that frame slot last reset.

fn void idle_context_pools_are_never_reset() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    // Never recorded on: covers the "idle context" scenario, plus the
    // built-in default/helper contexts, which stay idle too.
    gpu::compat::RecordingContextHandle idle_ctx = gpu::compat::create_recording_context(&device)!!;
    defer (void)gpu::compat::destroy_recording_context(&device, idle_ctx);

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    for (uint i = 0; i < 6; i++) {
        gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
        assert(vk::pool_reset_stats(state).reset == 0, "an idle context's pools are never reset");
        gpu::compat::end_frame(&frame)!!;
    }
}

fn void used_context_pool_resets_exactly_once_on_slot_wrap() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::RecordingContextHandle ctx = gpu::compat::create_recording_context(&device)!!;
    defer (void)gpu::compat::destroy_recording_context(&device, ctx);
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!; // slot 1
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS, ctx)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame)!!;

    frame = gpu::compat::begin_frame(&device)!!; // slot 0: never touched
    assert(vk::pool_reset_stats(state).reset == 0, "the other slot was never dirtied, so nothing resets");
    gpu::compat::end_frame(&frame)!!;

    frame = gpu::compat::begin_frame(&device)!!; // slot 1 wraps back: the one dirtied pool resets
    assert(vk::pool_reset_stats(state).reset == 1, "the recorded context's pool resets exactly once on its slot's wrap");
    gpu::compat::end_frame(&frame)!!;

    frame = gpu::compat::begin_frame(&device)!!; // slot 0 again: still clean
    assert(vk::pool_reset_stats(state).reset == 0, "slot 0 stays clean");
    gpu::compat::end_frame(&frame)!!;

    frame = gpu::compat::begin_frame(&device)!!; // slot 1 again: the bit stays cleared, nothing new recorded
    assert(vk::pool_reset_stats(state).reset == 0, "the dirty bit stays cleared until the context records again");
    gpu::compat::end_frame(&frame)!!;
}

fn void wait_timeout_zero_on_reached_value_succeeds() @test {
    gpu::compat::DeviceDesc desc = { .backend = gpu::compat::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SemaphoreDesc sem_desc = { .initial_value = (gpu::compat::SemaphoreValue)1 };
    gpu::compat::SemaphoreHandle sem = gpu::compat::create_semaphore(&device, &sem_desc)!!;
    defer gpu::compat::destroy_semaphore(&device, sem)!!;

    gpu::compat::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::compat::SemaphoreValue)1,
        timeout_ns: 0,
    )!!;
}
