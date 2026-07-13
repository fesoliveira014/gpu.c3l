module gpu_test;

import gpu @public;
import gpu::vk @public;

const uint TEST_HANDLE_GENERATION_MAX = 0xFF_FFFF;

// Headless (lavapipe): record a buffer copy from a host-written upload buffer to
// a readback buffer, submit signaling a timeline semaphore, wait, and verify the
// bytes. Assumes host-coherent memory (non-coherent flush/invalidate is later).

fn ulong command_submit_count(gpu::Device* device) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
    return vk::submit_stats(state).queue_submits;
}


fn void command_list_is_small_owner_token() @test {
    assert(gpu::CommandList::size <= 16, "CommandList stays within two machine words");
}

fn void copy_upload_to_readback() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc up_desc = {
        .size = 256, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING, .debug_name = "upload",
    };
    gpu::BufferHandle upload = gpu::create_buffer(&device, &up_desc)!!;
    defer gpu::destroy_buffer(&device, upload)!!;

    gpu::BufferDesc rb_desc = {
        .size = 256, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK, .debug_name = "readback",
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::GpuSpan up_span = gpu::get_buffer_span(&device, upload)!!;
    char* src = (char*)up_span.cpu;
    for (uint i = 0; i < 256; i++) src[i] = (char)i;

    gpu::SemaphoreDesc sem_desc = { .initial_value = (gpu::SemaphoreValue)0 };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::BufferCopyDesc copy = { .src = upload, .dst = readback, .src_offset = 0, .dst_offset = 0, .size = 256 };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier bar = {
        .buffer = readback, .offset = 0, .size = 256,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::end_commands(&cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SemaphoreSignal[1] sigs = { { .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.TRANSFER } };
    gpu::SubmitDesc submit = { .command_lists = lists[..], .signals = sigs[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::SemaphoreValue)1,
        timeout_ns: 1_000_000_000,
    )!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    char* dst = (char*)rb_span.cpu;
    for (uint i = 0; i < 256; i++) assert(dst[i] == (char)i, "readback byte matches uploaded byte");

    gpu::end_frame(&frame)!!;
}

fn void empty_command_list_submits() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame)!!;
}

fn void command_list_owns_its_device_token_value() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::Device temporary = device;
    gpu::CommandList commands = gpu::begin_commands(&temporary, gpu::QueueKind.GRAPHICS)!!;
    temporary = {};
    gpu::end_commands(&commands)!!;

    gpu::CommandList[1] lists = { commands };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    gpu::submit(&device, &submit_desc)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void copied_alias_cannot_resubmit_after_success() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::CommandList original = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&original)!!;
    gpu::CommandList[1] copied = { original };
    gpu::SubmitDesc first = { .command_lists = copied[..] };
    gpu::submit(&device, &first)!!;
    gpu::CommandList replacement = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&replacement)!!;
    assert(replacement.handle.index == original.handle.index, "the freed command slot is reused");
    assert(replacement.handle.generation != original.handle.generation,
        "slot reuse advances the command generation");


    ulong before = command_submit_count(&device);
    gpu::CommandList[1] original_alias = { original };
    gpu::SubmitDesc second = { .command_lists = original_alias[..] };
    fault rejected = @catch(gpu::submit(&device, &second));
    assert(rejected == gpu::INVALID_HANDLE, "a consumed alias faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "a consumed alias reaches no Vulkan queue submit");
    gpu::CommandList[1] replacement_lists = { replacement };
    gpu::SubmitDesc replacement_submit = { .command_lists = replacement_lists[..] };
    gpu::submit(&device, &replacement_submit)!!;

    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void command_generation_wraps_without_poisoning_slot() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::CommandList nearly_wrapped = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&nearly_wrapped)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.commands.slots[nearly_wrapped.handle.index].generation = TEST_HANDLE_GENERATION_MAX - 1;
    nearly_wrapped.handle.generation = TEST_HANDLE_GENERATION_MAX - 1;

    gpu::CommandList[1] first_lists = { nearly_wrapped };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::submit(&device, &first_submit)!!;

    gpu::CommandList wrapped = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    assert(wrapped.handle.index == nearly_wrapped.handle.index, "wrapped slot is reused");
    assert(wrapped.handle.generation == 1, "command generation wraps to the first valid value");
    gpu::end_commands(&wrapped)!!;
    gpu::CommandList[1] wrapped_lists = { wrapped };
    gpu::SubmitDesc wrapped_submit = { .command_lists = wrapped_lists[..] };
    gpu::submit(&device, &wrapped_submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void duplicate_token_batch_rejects_without_consuming() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[2] duplicates = { cmd, cmd };
    gpu::SubmitDesc bad = { .command_lists = duplicates[..] };
    ulong before = command_submit_count(&device);
    fault rejected = @catch(gpu::submit(&device, &bad));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR, "duplicate tokens fault before submission");
    assert(command_submit_count(&device) == before, "duplicate-token rejection records no queue submit");

    gpu::CommandList[1] one = { cmd };
    gpu::SubmitDesc good = { .command_lists = one[..] };
    gpu::submit(&device, &good)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void second_live_device_is_rejected() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device owner = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&owner)!!;
    fault second_device = @catch(gpu::create_device(&desc));
    assert(second_device == gpu::INVALID_RESOURCE_STATE, "only one device may be live");

    gpu::CommandList cmd = gpu::begin_commands(&owner, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&owner, &submit)!!;
    gpu::wait_queue_idle(&owner, gpu::QueueKind.GRAPHICS)!!;
}

fn void abandoned_token_invalidates_when_pool_slot_resets() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 1 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList abandoned = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&abandoned)!!;
    gpu::end_frame(&frame)!!;
    frame = gpu::begin_frame(&device)!!;

    fault stale = @catch(gpu::end_commands(&abandoned));
    assert(stale == gpu::INVALID_HANDLE, "pool reset invalidates an abandoned command token");
    gpu::end_frame(&frame)!!;
}

fn void live_command_record_blocks_context_destruction() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    gpu::RecordingContextHandle context = gpu::create_recording_context(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS, context)!!;
    fault live = @catch(gpu::destroy_recording_context(&device, context));
    assert(live == gpu::INVALID_RESOURCE_STATE, "a context with a live command record cannot be destroyed");

    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::destroy_recording_context(&device, context)!!;
}

fn void bad_command_state_rejected() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&cmd)!!;

    // Recording after end must be rejected.
    gpu::GlobalBarrier gb = {
        .before_stage = gpu::Stage.HOST, .after_stage = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE, .after_hazard = gpu::Hazard.TRANSFER_READ,
    };
    fault rejected = @catch(gpu::cmd_global_barrier(&cmd, &gb));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR, "recording after end returns COMMAND_RECORDING_ERROR");
    gpu::end_frame(&frame)!!;
}

fn void texture_barrier_faults() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureBarrier bar = {
        .texture = gpu::TEXTURE_HANDLE_INVALID,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        .old_layout = gpu::TextureLayout.TRANSFER_DST, .new_layout = gpu::TextureLayout.SHADER_READ,
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &bar));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on an invalid texture handle faults INVALID_HANDLE");
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame)!!;
}

// vkWaitSemaphores with timeout 0 is a spec-guaranteed poll: it returns
// immediately, so these are deterministic on lavapipe.

fn void wait_timeout_zero_on_unreached_value_faults() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::SemaphoreDesc sem_desc = { .initial_value = (gpu::SemaphoreValue)0 };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    fault timeout_fault = @catch(gpu::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::SemaphoreValue)1,
        timeout_ns: 0,
    ));
    assert(timeout_fault == gpu::WAIT_TIMEOUT, "timeout-0 wait on an unreached value faults WAIT_TIMEOUT");
}

// Lazy pool resets: begin_frame's reset loop only pays for pools a context
// actually dirtied since that frame slot last reset.

fn void idle_context_pools_are_never_reset() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    // Never recorded on: covers the "idle context" scenario, plus the
    // built-in default/helper contexts, which stay idle too.
    gpu::RecordingContextHandle idle_ctx = gpu::create_recording_context(&device)!!;
    defer (void)gpu::destroy_recording_context(&device, idle_ctx);

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    for (uint i = 0; i < 6; i++) {
        gpu::FrameToken frame = gpu::begin_frame(&device)!!;
        assert(vk::pool_reset_stats(state).reset == 0, "an idle context's pools are never reset");
        gpu::end_frame(&frame)!!;
    }
}

fn void used_context_pool_resets_exactly_once_on_slot_wrap() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::RecordingContextHandle ctx = gpu::create_recording_context(&device)!!;
    defer (void)gpu::destroy_recording_context(&device, ctx);
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::FrameToken frame = gpu::begin_frame(&device)!!; // slot 1
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS, ctx)!!;
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame)!!;

    frame = gpu::begin_frame(&device)!!; // slot 0: never touched
    assert(vk::pool_reset_stats(state).reset == 0, "the other slot was never dirtied, so nothing resets");
    gpu::end_frame(&frame)!!;

    frame = gpu::begin_frame(&device)!!; // slot 1 wraps back: the one dirtied pool resets
    assert(vk::pool_reset_stats(state).reset == 1, "the recorded context's pool resets exactly once on its slot's wrap");
    gpu::end_frame(&frame)!!;

    frame = gpu::begin_frame(&device)!!; // slot 0 again: still clean
    assert(vk::pool_reset_stats(state).reset == 0, "slot 0 stays clean");
    gpu::end_frame(&frame)!!;

    frame = gpu::begin_frame(&device)!!; // slot 1 again: the bit stays cleared, nothing new recorded
    assert(vk::pool_reset_stats(state).reset == 0, "the dirty bit stays cleared until the context records again");
    gpu::end_frame(&frame)!!;
}

fn void wait_timeout_zero_on_reached_value_succeeds() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::SemaphoreDesc sem_desc = { .initial_value = (gpu::SemaphoreValue)1 };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    gpu::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::SemaphoreValue)1,
        timeout_ns: 0,
    )!!;
}
